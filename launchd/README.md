# launchd services

Two services replace running the API server and the active-learning scheduler manually.

## Install

Both plists use `__REPO_ROOT__` as a placeholder. Replace it with this repo's absolute path, then load:

    sed "s|__REPO_ROOT__|$(pwd)|g" launchd/com.smartcart.api.plist > ~/Library/LaunchAgents/com.smartcart.api.plist
    sed "s|__REPO_ROOT__|$(pwd)|g" launchd/com.smartcart.al-scheduler.plist > ~/Library/LaunchAgents/com.smartcart.al-scheduler.plist
    mkdir -p artifacts/logs

Before loading the al-scheduler service, seed the pipeline state with the currently-live
model's real metrics -- otherwise the first autonomous promotion only has to clear the
absolute floors (`SMARTCART_RETRAIN_MIN_MAP50`/`SMARTCART_RETRAIN_MIN_VARIANT_ACC`) rather
than actually beat the model already in production:

    uv run python seed_al_baseline.py

Then load both services:

    launchctl load ~/Library/LaunchAgents/com.smartcart.api.plist
    launchctl load ~/Library/LaunchAgents/com.smartcart.al-scheduler.plist

## Verify

    curl http://localhost:8000/health          # API server responding
    tail -f artifacts/logs/al-scheduler.log    # scheduler running on its StartInterval

## Uninstall

    launchctl unload ~/Library/LaunchAgents/com.smartcart.api.plist
    launchctl unload ~/Library/LaunchAgents/com.smartcart.al-scheduler.plist
    rm ~/Library/LaunchAgents/com.smartcart.{api,al-scheduler}.plist

## Change the scheduler interval

Edit `StartInterval` (seconds) in `com.smartcart.al-scheduler.plist` before installing, or edit the installed copy under `~/Library/LaunchAgents/` and reload.
