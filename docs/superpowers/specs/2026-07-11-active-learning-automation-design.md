# Autonomous active-learning pipeline for YOLO + DINOv2

## Context

Today's active-learning loop (introduced [2026-07-09](../../handoff-2026-07-09.md)) is five manual steps threaded through automatic capture:

1. `maybe_capture()` ([active_learning_capture.py](../../../src/deploy/active_learning_capture.py)) — automatic, always on. Stages a `/predict` frame + sidecar JSON under `SMARTCART_CAPTURE_DIR` whenever any detection is below `SMARTCART_CAPTURE_CONF_THRESHOLD` or there are zero detections.
2. **Manual:** run `push_captures_to_label_studio.py` to push staged captures into a Label Studio project.
3. **Manual (human):** review/correct annotations in Label Studio.
4. **Manual:** run `retrain_from_label_studio.py` → pulls corrections, rebuilds catalog+gallery, resynthesizes scenes, retrains YOLO, audits against an absolute `mAP50 >= 0.5` gate ([training_pipeline.py](../../../src/pipeline/training_pipeline.py)). Never touches live weights.
5. **Manual:** if the audit passes, a human edits `SMARTCART_WEIGHTS_PATH` in `.env` and restarts the API to promote.

This spec automates all five steps end to end, using a local Qwen2.5-VL-3B-Instruct-8bit model served by LM Studio as an independent second opinion in place of the human review step, and a mechanical metrics-based rule in place of human promotion approval. The explicit goal (confirmed during design) is to minimize required human intervention as far as the tooling reasonably allows, while degrading safely (skip, don't guess) wherever the automation isn't confident — not to preserve today's human-in-the-loop review as a checkpoint.

### A correctness gap found in the existing pipeline, fixed as part of this work

`ModelTrainingPipeline.rebuild_catalog()` ([training_pipeline.py:29-32](../../../src/pipeline/training_pipeline.py#L29-L32)) writes `catalog_prices.csv`/`gallery_index.npy`/`gallery_meta.csv` straight to `./artifacts/` — the same path the live API reads via `get_gallery_index_path()`/`get_gallery_meta_path()` ([detector.py](../../../src/deploy/detector.py)). Unlike YOLO weights (which land in an isolated `runs/detect/retrain_<run_name>/`, requiring manual promotion), the gallery rebuild happens *unconditionally, before the audit even runs*. A human running the pipeline manually today notices a FAILED printout before restarting the API; once this loop is unattended, that safety net disappears, so this spec isolates gallery/catalog writes into the same candidate-then-promote flow the weights already use.

## Goals

- Fully autonomous loop: capture → label → retrain → gate → promote → reload, with no required human action for existing catalog categories.
- Improve both YOLO detection (mAP50) and DINOv2 variant identification (a new held-out accuracy metric — nothing today measures this).
- Never let automation failure modes (VLM unavailable, ambiguous label, failed gate) corrupt the dataset or degrade the live model. Every failure mode skips and waits for more data, never guesses.
- Preserve, don't remove, the existing Label Studio integration — it stays available for manual override or genuinely new-product cold-start onboarding, it's just not on the default automated path.

## Non-goals

- Onboarding brand-new products with no existing catalog entry. The VLM is prompted with existing catalog category names; a product outside that set still needs the existing manual cold-start flow (`dataset/raw_photos/README.md`) — new name, price, seed reference photos.
- Live/canary traffic validation before promotion. Promotion is gated on proxy metrics (mAP50 against synthetic scenes, held-out accuracy against gallery photos) — there is no shadow-traffic or live A/B mechanism in this design. This is a known, accepted trade-off of removing the human promotion checkpoint, not an oversight.
- Any change to the live `/predict` request path or its latency. Everything in this spec is asynchronous background automation.

## Architecture

```
/predict (live, unchanged)
   │
   ├─► maybe_capture() [existing, automatic] → staging dir (image + sidecar JSON, per-detection confidences)
   │
   ▼
al_scheduler_check.py   ◄── launchd, every SMARTCART_AL_SCHEDULER_INTERVAL_SEC
   │
   ├─ [A] VLM auto-importer (NEW)                    — see "VLM auto-importer" below
   │      every unconsumed staged capture is either auto-imported into the dataset tree,
   │      or discarded (never left pending for a human)
   │
   └─ [B] Retrain trigger
          once newly-auto-imported crops since last run ≥ SMARTCART_AL_RETRAIN_TRIGGER_COUNT:
          → ModelTrainingPipeline.run(), writing to an isolated candidate dir (NEW — see "Candidate staging")
          → YOLO mAP50 audit (existing, unchanged)
          → DINOv2 held-out variant-accuracy audit (NEW)
          → writes artifacts/candidates/<run_name>/report.json
          → auto-promote (NEW) if both gates pass and both metrics ≥ the currently-promoted run's own
          → launchctl kickstart -k com.smartcart.api (NEW)

artifacts/al_pipeline_state.json (NEW) tracks last-processed counts and the currently-promoted
run's recorded metrics, so each candidate's report diffs against what's actually live.
```

## VLM auto-importer (new — `src/models/vlm_verifier.py` + a new import path in `src/data/annotation_import.py`)

**LM Studio integration**, confirmed working against the running local server:

```
POST http://localhost:1234/api/v1/chat
{
  "model": "qwen2.5-vl-3b-instruct",
  "input": [
    {"type": "text", "content": "<prompt>"},
    {"type": "image", "data_url": "data:image/jpeg;base64,<...>"}
  ]
}
```
Response text is read from `output[0].content`. (The server also exposes a standard OpenAI-compatible `/v1/chat/completions` with the conventional `messages`/`image_url` shape — confirmed working too — but `/api/v1/chat` is the integration target per the running setup.) Base URL and model name are configurable (`SMARTCART_VLM_BASE_URL`, `SMARTCART_VLM_MODEL`) so this isn't hardcoded to one local setup.

**Per capture, read the sidecar JSON's `detections` list:**

- **Mid-confidence detections** (every detection's confidence in `[SMARTCART_AUTOLABEL_MIN_CONF, SMARTCART_CAPTURE_CONF_THRESHOLD)`): crop the box, ask the VLM which catalog category (from a supplied candidate list) it shows. Compare the answer to YOLO's predicted class using the same longest-keyword-match logic `classify_category()` ([annotation_import.py](../../../src/data/annotation_import.py)) already uses — reused, not reimplemented.
  - **Agree** → crop is auto-imported into the dataset tree via the existing `variant_subpath()`/`classify_category()` routing, exactly where a human-reviewed Label Studio export would land.
  - **Disagree, or VLM says none/unsure** → discard the capture. Two independent signals failing to agree is treated as "not confident enough to add to the dataset," not as a coin flip to resolve.
- **Zero-detection frames**: ask the VLM for a category match plus a grounding-style box over the whole frame (Qwen2.5-VL supports box-formatted output when prompted for it — to be confirmed empirically against the running instance during implementation, the same way the chat schema was confirmed in this session).
  - Box parses and passes a sanity check (in-bounds, plausible size — not >90% or <1% of the frame) → use it.
  - Box missing/invalid but category match found → fall back to a whole-frame box for that category. This fits the app's own capture pattern: `CameraFeed.tsx`'s "freeze and detect" is already one held-up item per frame, so whole-frame-as-box is a reasonable default here, not a blind guess.
  - No confident category match at all → discard.
- Every consumed capture (imported or discarded) is marked so it's never reprocessed — mirrors how `mark_pushed()` already avoids reprocessing pushed captures, image files are never moved/deleted, only tracked.
- **VLM unreachable or errors** → treated identically to "disagree/unsure" → discard that capture, log it, continue. Never blocks the scheduler tick.

**Passive audit trail:** at rate `SMARTCART_AL_AUDIT_SAMPLE_RATE`, sampled auto-import decisions (image + crop + VLM answer + YOLO answer + accept/discard) are written to `artifacts/al_review/` for optional, non-blocking human spot-checking. Nothing reads this directory automatically; it exists purely for a human to look at if they choose to.

## Candidate staging (fixes the gallery-overwrite gap)

`ModelTrainingPipeline` gains a `candidate_dir` parameter (e.g. `artifacts/candidates/<run_name>/`). `rebuild_catalog()` writes `catalog_prices.csv`/`gallery_index.npy`/`gallery_meta.csv` there instead of to `./artifacts/`. YOLO weights already land safely isolated under `runs/detect/retrain_<run_name>/weights/best.pt` — unchanged.

## DINOv2 held-out variant-accuracy audit (new)

Mirrors the existing YOLO mAP50 audit, for the gallery side — nothing today measures whether a retrain actually improved variant identification.

- New function (alongside `CheckoutModelAuditor` in [auditor.py](../../../src/models/auditor.py), or a small new `variant_auditor.py`): for every catalog variant with ≥2 reference photos, hold out ~20% of its photos as a query set, build the gallery from the rest, and run the *same* cosine-similarity nearest-neighbor lookup `VariantResolver.resolve()` ([variant_resolver.py](../../../src/models/variant_resolver.py)) already implements — reused, not reimplemented. Reports top-1 accuracy across all held-out photos.
- Variants with only 1 reference photo (common right after onboarding a new SKU) are excluded from the accuracy computation; the excluded count is recorded in the report for visibility.

## Promotion (mechanical, auto-restart)

`artifacts/candidates/<run_name>/report.json` records: both gate results, both metrics vs. the currently-promoted run's own recorded metrics (from `al_pipeline_state.json`), and the auto-import provenance split (auto-imported vs. any manually-imported counts, for visibility even though no human needs to read this to act).

Auto-promotion rule: candidate mAP50 ≥ live mAP50 **and** candidate variant-accuracy ≥ live variant-accuracy (both gates from `training_pipeline.py`/the new variant auditor must also independently pass their own absolute floors, `SMARTCART_RETRAIN_MIN_MAP50` / `SMARTCART_RETRAIN_MIN_VARIANT_ACC`). On promotion: copy candidate gallery/catalog files into live `./artifacts/`, update `SMARTCART_WEIGHTS_PATH` in `.env` to the candidate's weights path, update `al_pipeline_state.json` with the new live run's metrics, then `launchctl kickstart -k com.smartcart.api` to restart the API process so its `lru_cache`d `get_detector()`/`get_variant_resolver()` singletons pick up the new files (no hot-reload endpoint is added — a restart-to-load is kept, matching today's behavior, just automated instead of manual).

If the rule doesn't pass: no promotion, no restart. The candidate directory is left in place (not deleted) for inspection; the next scheduler tick continues normally once more data accumulates.

## Process management (launchd)

Two services:

- **`com.smartcart.api`** — runs `main_api_server.py` as a persistent service (replaces manually running `uv run python main_api_server.py`), `KeepAlive` for crash recovery.
- **`com.smartcart.al-scheduler`** — runs `al_scheduler_check.py` on a `StartInterval` of `SMARTCART_AL_SCHEDULER_INTERVAL_SEC`.

Both are plain plist files; not unit-testable, verified manually once during implementation (documented as a manual verification step, not a pytest case).

## Config (new `SMARTCART_*` env vars, following the existing convention)

| Var | Purpose |
|---|---|
| `SMARTCART_VLM_BASE_URL` | LM Studio server base URL (default `http://localhost:1234`) |
| `SMARTCART_VLM_MODEL` | Model identifier as loaded in LM Studio (e.g. `qwen2.5-vl-3b-instruct`) |
| `SMARTCART_AUTOLABEL_MIN_CONF` | Lower bound of the mid-confidence auto-label band |
| `SMARTCART_AL_RETRAIN_TRIGGER_COUNT` | Auto-imported crops needed since last retrain to trigger the next one |
| `SMARTCART_RETRAIN_MIN_VARIANT_ACC` | Absolute floor for the new DINOv2 held-out accuracy gate |
| `SMARTCART_AL_AUDIT_SAMPLE_RATE` | Fraction of auto-import decisions written to the passive review log |
| `SMARTCART_AL_SCHEDULER_INTERVAL_SEC` | launchd `StartInterval` for the checker script |

Existing Label Studio env vars (`SMARTCART_LS_PROJECT_ID`, `LABEL_STUDIO_API_KEY`, etc.) are untouched and remain valid — that integration isn't deleted, just bypassed by the default automated path, so it's still available for manual override or cold-start onboarding.

## Error handling summary

Every new failure mode degrades to "skip and wait for more data," never to "block the scheduler" or "guess":

| Failure | Behavior |
|---|---|
| LM Studio unreachable/times out | Discard that capture, log, continue |
| VLM and YOLO disagree | Discard that capture |
| VLM gives no confident category match | Discard that capture |
| Zero-detection frame, no parseable box, no category match | Discard that capture |
| Zero-detection frame, category match but no valid box | Fall back to whole-frame box |
| YOLO mAP50 gate fails | No promotion, candidate kept for inspection |
| DINOv2 variant-accuracy gate fails | No promotion, candidate kept for inspection |
| Candidate metrics don't beat live metrics | No promotion, candidate kept for inspection |

## Testing

- `vlm_verifier.py` — mocked via `httpx` monkeypatching, matching the existing Label Studio test pattern (see the `httpx`-module-identity gotcha already documented in `CLAUDE.md`).
- Auto-importer agree/disagree/discard logic, and the zero-detection box-parse/fallback/discard logic — unit tests with synthetic detection + VLM-response fixtures, no real LM Studio/YOLO/DINO calls.
- DINOv2 held-out accuracy auditor — unit tests against a small synthetic gallery (a handful of variants, known nearest-neighbor structure), asserting the accuracy computation and the ≥2-photos exclusion rule.
- `promote_candidate` file-copy and state-update logic — unit tests against tmp directories.
- launchd plists — manual verification (start both services, confirm the API responds, confirm the checker script runs on schedule), not part of the automated test suite.
