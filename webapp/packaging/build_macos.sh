#!/usr/bin/env bash
# Build Player Tracking.app and wrap it in an installer DMG.
set -euo pipefail
cd "$(dirname "$0")/.."                       # webapp/
PY=${PY:-python3}
DIST=${DIST:-dist}
APP="$DIST/Player Tracking.app"
case "$(uname -m)" in
  arm64) ARCH=arm64 ;;          # Apple Silicon
  *)     ARCH=intel ;;          # x86_64
esac
DMG="$DIST/PlayerTracking-macOS-$ARCH.dmg"

"$PY" tools/make_icons.py
"$PY" tools/make_dmg_bg.py
"$PY" -m PyInstaller packaging/player_tracking.spec --noconfirm --distpath "$DIST" --workpath build

# Apple Silicon refuses to execute code with no signature at all; the ad-hoc
# signature below is free, needs no account, and is NOT notarization — the user
# still clears Gatekeeper once by hand (steps are on the DMG background).
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP" && echo "ad-hoc signature ok"

STAGE=$(mktemp -d)
cp -R "$APP" "$STAGE/"
rm -f "$DMG"
create-dmg \
  --volname "Player Tracking" \
  --background packaging/dmg_bg.png \
  --window-pos 200 120 --window-size 760 560 \
  --icon-size 112 \
  --icon "Player Tracking.app" 180 200 \
  --app-drop-link 580 200 \
  --hide-extension "Player Tracking.app" \
  --no-internet-enable \
  "$DMG" "$STAGE"
rm -rf "$STAGE"
echo "built: $DMG"
