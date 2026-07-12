# launchd services

Two services replace running the API server and the active-learning scheduler manually.

## Install

Both plists use `__REPO_ROOT__` as a placeholder. Replace it with this repo's absolute path, then load:

    sed "s|__REPO_ROOT__|$(pwd)|g" launchd/com.smartcart.api.plist > ~/Library/LaunchAgents/com.smartcart.api.plist
    sed "s|__REPO_ROOT__|$(pwd)|g" launchd/com.smartcart.al-scheduler.plist > ~/Library/LaunchAgents/com.smartcart.al-scheduler.plist
    mkdir -p artifacts/logs
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
