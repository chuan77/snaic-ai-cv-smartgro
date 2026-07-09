# Week 4 Demo Script — SmartCart AI Checkout Assistant

Target length: **2:30** (hard ceiling 3:00). Speaking pace ~140 wpm — the word counts below are a guide for rehearsing to time, not something to read verbatim.

## Before you start (do this before recording/presenting, not on camera)

- [ ] Backend running: `uv run python main_api_server.py` — confirm `http://localhost:8888/health` (or your configured port) returns `{"status":"ok"}`
- [ ] Frontend running: `cd frontend && npm run dev` — open it in the browser, camera permission already granted
- [ ] 2-3 real physical items on hand that map to catalog classes (e.g. an apple, a carrot, a milk carton) — pick items you know detect well, not edge cases
- [ ] A terminal tab ready with the retrain output pre-scrolled to the `Retrain PASSED: mAP50=0.7870...` line (see Segment 4) — don't run it live, just have the evidence visible
- [ ] Browser tab on the Label Studio project (optional, only if doing Segment 4 visually)
- [ ] Close anything else on screen — Slack, notifications, unrelated tabs

## Segment 1 — The problem (0:00–0:20, ~45 words)

**On screen:** README or a plain terminal, nothing running yet.

**Say:**
> "The dataset I started with — Marcus Klasson's GroceryStoreDataset — has thousands of grocery photos, but zero bounding boxes. Just classification labels: 'this photo is an apple.' To train an object detector, you normally need boxes around every object. I didn't have that."

## Segment 2 — The trick: synthetic scenes (0:20–0:50, ~60 words)

**On screen:** switch to a synthetic training image if you have one open (`synthetic_dataset/train/images/`), or just keep talking over the terminal.

**Say:**
> "So instead of hand-labeling thousands of photos, I wrote a pipeline that crops products out of those classification photos and programmatically composites them onto generated backgrounds — at random positions, scales, and counts. Every synthetic scene comes with a perfect bounding box for free, because I placed it there. That gave me a fully-labeled YOLO dataset with no manual annotation at all, which I used to train a YOLO11 detector."

## Segment 3 — Live demo: the checkout flow (0:50–1:50, ~100 words + live actions)

**On screen:** the React frontend, webcam live.

**Say (while doing):**
> "Here's the actual checkout app. It's a live webcam feed talking to a FastAPI backend running that trained detector."

- **Do:** hold up item #1, click **Freeze & Detect**.
> "It detects the item, matches it against the catalog, and prices it — this isn't a hardcoded barcode lookup, it's genuinely running inference on the frame right now."
- **Do:** wait ~1s for auto-resume to live, hold up item #2, **Freeze & Detect** again.
> "It keeps a running cart across multiple scans —"
- **Do:** click **Checkout**.
> "— and closes out with a receipt, the same way a real self-checkout kiosk would."
- **Do:** click **Acknowledge & Close**.

*(If you have time and a fine-grained item like a specific chocolate bar or noodle brand: mention the two-stage detection — YOLO finds the coarse category, then a second DINOv2 nearest-neighbor lookup resolves the specific brand/variant, without needing to retrain YOLO on every SKU.)*

## Segment 4 — Beyond a static model: active learning (1:50–2:20, ~70 words)

**On screen:** the pre-scrolled terminal with the retrain result, or the Label Studio tab.

**Say:**
> "The model isn't frozen after training. When it's uncertain — low confidence, or it misses something entirely — that frame gets automatically flagged and staged for review. I push those into Label Studio, correct the ones that are wrong, and pull the corrections back into an automated retrain-and-audit pipeline."

- **Do:** point at the terminal output.
> "This ran for real this week — 31 corrected production frames, folded into the existing 120-class catalog, retrained, and the new model passed its accuracy gate at 0.79 mAP before I promoted it to serve live traffic. It's a closed feedback loop, not a one-shot training run."

## Segment 5 — Close (2:20–2:30, ~30 words)

**Say:**
> "So: no manual bounding-box labeling, a real-time detection-and-pricing checkout flow, and a production feedback loop that keeps the model improving from its own mistakes. Thanks."

---

## Timing cheat sheet

| Segment | Time | Cumulative |
|---|---|---|
| 1. Problem | 0:20 | 0:20 |
| 2. Synthetic data trick | 0:30 | 0:50 |
| 3. Live checkout demo | 1:00 | 1:50 |
| 4. Active learning loop | 0:30 | 2:20 |
| 5. Close | 0:10 | 2:30 |

**If you're running long:** cut the fine-grained variant aside in Segment 3 first, then shorten Segment 2 to one sentence ("I generate fully-labeled synthetic training scenes instead of hand-labeling") — the live demo (Segment 3) is the part that should never get cut, it's the only part that *shows* rather than tells.

**If you're running short / have time to spare:** add one line in Segment 3 about the cart's running total, or in Segment 4 mention the audit gate exists specifically so a bad retrain can never silently replace a good one in production.

## Scope note (in case of Q&A)

This is visual-product-class matching (e.g. "chocolate bar → Cadbury Dairy Milk"), not barcode-level SKU recognition — prices are editable demo catalog data, not a live pricing feed. If asked "does it tell two flavors of the same brand apart," the honest answer is: only where there's enough real reference photos per variant to support it; the project intentionally didn't chase full SKU-level granularity everywhere.
