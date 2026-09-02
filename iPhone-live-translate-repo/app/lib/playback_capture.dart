import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/services.dart';

/// Captures the audio other apps are playing, instead of the microphone.
///
/// This is the answer to translating something you're listening to on
/// headphones. Android taps the stream before it reaches the output device,
/// so Bluetooth earbuds make no difference, and you get a clean digital
/// signal rather than a microphone recording of a speaker.
///
/// Limits worth knowing before you debug silence:
///  - Android 10 or later. [isSupported] reports this.
///  - The user approves a system dialog every session. Android 15 no longer
///    allows the permission to be remembered across app restarts.
///  - Apps can opt out with android:allowAudioPlaybackCapture="false", and
///    DRM-protected playback is always excluded. Netflix and Spotify opt out;
///    social video apps generally don't.
///  - A persistent notification is shown while capturing. Not optional.
class PlaybackCapture {
  static const _methods = MethodChannel('live_translate/playback');
  static const _events = EventChannel('live_translate/playback_audio');

  static Future<bool> isSupported() async {
    try {
      return await _methods.invokeMethod<bool>('isSupported') ?? false;
    } on PlatformException {
      return false;
    } on MissingPluginException {
      // Running on a platform without the native side, e.g. iOS.
      return false;
    }
  }

  /// Shows the consent dialog and begins capture. Returns false if the user
  /// declined or the platform refused.
  static Future<bool> start() async {
    try {
      return await _methods.invokeMethod<bool>('start') ?? false;
    } on PlatformException {
      return false;
    }
  }

  static Future<void> stop() async {
    try {
      await _methods.invokeMethod<void>('stop');
    } on PlatformException {
      // Already stopped, or never started.
    }
  }

  /// Raw PCM16 mono at 16 kHz, in roughly 100 ms chunks — the same shape the
  /// microphone path produces, so the rest of the pipeline is unchanged.
  static Stream<Uint8List> get audio => _events
      .receiveBroadcastStream()
      .map((event) => event as Uint8List);
}

/// The floating subtitle bar shown over other apps in Media mode.
///
/// Lives on the same platform channel as [PlaybackCapture] because the two are
/// always used together: capturing what another app plays is only useful if
/// you can show the result without leaving that app.
class SubtitleOverlay {
  static const _methods = MethodChannel('live_translate/playback');

  /// Drawing over other apps is granted on a system settings page rather than
  /// by a dialog, so this can't be requested inline like a normal permission.
  static Future<bool> hasPermission() async {
    try {
      return await _methods.invokeMethod<bool>('overlayHasPermission') ?? false;
    } on PlatformException {
      return false;
    } on MissingPluginException {
      return false;
    }
  }

  /// Opens the system settings page. Returns immediately — the user grants it
  /// out of band, so check [hasPermission] again when they come back.
  static Future<void> requestPermission() async {
    try {
      await _methods.invokeMethod<bool>('overlayRequestPermission');
    } on PlatformException {
      // Nothing useful to do; hasPermission will still report false.
    }
  }

  static Future<bool> show() async {
    try {
      return await _methods.invokeMethod<bool>('overlayShow') ?? false;
    } on PlatformException {
      return false;
    }
  }

  /// [primary] is the translation, shown large. [secondary] is the original,
  /// shown small underneath, or null to hide that line.
  static Future<void> update(String primary, {String? secondary}) async {
    try {
      await _methods.invokeMethod<void>('overlayUpdate', {
        'primary': primary,
        'secondary': secondary,
      });
    } on PlatformException {
      // Overlay was dismissed underneath us.
    }
  }

  static Future<void> hide() async {
    try {
      await _methods.invokeMethod<void>('overlayHide');
    } on PlatformException {
      // Already gone.
    }
  }
}
