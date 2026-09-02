#!/usr/bin/env bash
#
# Prepares the Flutter project for an iOS build. Run by both Codemagic
# workflows before building.
#
# The ios/ folder is generated here rather than committed. That way you never
# need a Mac to change anything about the iOS side, and the Info.plist keys
# live in one place instead of drifting between the repo and CI.
#
# Expects to be run from the folder containing pubspec.yaml.

set -euo pipefail

ORG="${ORG:-com.kda}"
MIN_IOS="${MIN_IOS:-13.0}"
PB=/usr/libexec/PlistBuddy

echo "==> Preparing iOS project (org $ORG, min iOS $MIN_IOS)"

# ---------------------------------------------------------------------------
# 1. Generate the iOS project if it isn't in the repo
# ---------------------------------------------------------------------------
if [ ! -d "ios" ]; then
  echo "No ios/ folder found — generating one"
  flutter create --platforms=ios --org "$ORG" .
else
  echo "ios/ folder already present"
fi

PLIST="ios/Runner/Info.plist"

# ---------------------------------------------------------------------------
# 2. Info.plist keys
#
# Delete-then-add for the collection types, so this stays correct whether the
# ios/ folder was just generated or committed to the repo.
# ---------------------------------------------------------------------------
set_string () {
  $PB -c "Set :$1 '$2'" "$PLIST" 2>/dev/null \
    || $PB -c "Add :$1 string '$2'" "$PLIST"
}

# Without this the app is terminated rather than shown a permission prompt.
set_string NSMicrophoneUsageDescription \
  "Live Translate uses the microphone to hear speech and translate it."

# iOS 14 and later block local network access until the user consents. The
# server is on the LAN, so without this the socket times out with no clue why.
set_string NSLocalNetworkUsageDescription \
  "Live Translate connects to your translation server on this network."

# App Transport Security blocks cleartext, which includes ws:// to a local
# address. This permits it for local destinations only, leaving the block in
# place for the public internet. The iOS counterpart to Android's
# network_security_config.xml, and forgetting it fails the same silent way.
$PB -c "Delete :NSAppTransportSecurity" "$PLIST" 2>/dev/null || true
$PB -c "Add :NSAppTransportSecurity dict" "$PLIST"
$PB -c "Add :NSAppTransportSecurity:NSAllowsLocalNetworking bool true" "$PLIST"

$PB -c "Delete :NSBonjourServices" "$PLIST" 2>/dev/null || true
$PB -c "Add :NSBonjourServices array" "$PLIST"
$PB -c "Add :NSBonjourServices:0 string _http._tcp" "$PLIST"

$PB -c "Delete :UIBackgroundModes" "$PLIST" 2>/dev/null || true
$PB -c "Add :UIBackgroundModes array" "$PLIST"
$PB -c "Add :UIBackgroundModes:0 string audio" "$PLIST"

echo "==> Info.plist:"
$PB -c "Print" "$PLIST"

# ---------------------------------------------------------------------------
# 3. Minimum iOS version
#
# record and wakelock_plus both require iOS 13. Flutter's generated Podfile
# leaves the platform line commented out, and the build fails much later with
# a pod dependency error that never mentions the real cause.
# ---------------------------------------------------------------------------
PODFILE="ios/Podfile"
if grep -q "^platform :ios" "$PODFILE"; then
  sed -i '' "s/^platform :ios.*/platform :ios, '$MIN_IOS'/" "$PODFILE"
elif grep -q "^# *platform :ios" "$PODFILE"; then
  sed -i '' "s/^# *platform :ios.*/platform :ios, '$MIN_IOS'/" "$PODFILE"
else
  printf "platform :ios, '%s'\n%s\n" "$MIN_IOS" "$(cat "$PODFILE")" > "$PODFILE"
fi
echo "==> Podfile platform: $(grep '^platform :ios' "$PODFILE")"

# ---------------------------------------------------------------------------
# 4. Dependencies
# ---------------------------------------------------------------------------
flutter pub get
( cd ios && pod install --repo-update )

echo "==> iOS project ready"
