import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:record/record.dart';

import 'playback_capture.dart';
import 'settings.dart';
import 'package:wakelock_plus/wakelock_plus.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

/// One thing somebody said, plus its translation.
///
/// The server sends `partial` updates while a person is still talking and a
/// `final` once they pause. Both carry the same [id], so a partial is replaced
/// in place rather than appended.
class Utterance {
  final int id;
  final String transcript;
  final String translation;
  final String sourceLanguage;
  final String targetLanguage;
  final bool isFinal;
  final int latencyMs;

  /// True for messages the user typed rather than spoke.
  final bool isOutgoing;

  const Utterance({
    required this.id,
    required this.transcript,
    required this.translation,
    required this.sourceLanguage,
    required this.targetLanguage,
    required this.isFinal,
    required this.latencyMs,
    this.isOutgoing = false,
  });

  /// True when the server had nothing to translate — Whisper already
  /// produced the target language. Showing the same sentence twice would
  /// just look like a bug.
  bool get isPassthrough =>
      translation.isEmpty || translation.trim() == transcript.trim();

  factory Utterance.fromJson(Map<String, dynamic> json) => Utterance(
        id: json['id'] as int,
        transcript: json['transcript'] as String? ?? '',
        translation: json['translation'] as String? ?? '',
        sourceLanguage: json['source_language'] as String? ?? '',
        targetLanguage: json['target_language'] as String? ?? '',
        isFinal: json['type'] != 'partial',
        latencyMs: json['latency_ms'] as int? ?? 0,
        isOutgoing: json['type'] == 'composed',
      );
}

/// A generated response to an utterance, in English and in whatever language
/// was actually spoken.
class Reply {
  final int utteranceId;
  final String english;
  final String spoken;
  final String spokenLanguage;
  final int latencyMs;

  const Reply({
    required this.utteranceId,
    required this.english,
    required this.spoken,
    required this.spokenLanguage,
    required this.latencyMs,
  });

  factory Reply.fromJson(Map<String, dynamic> json) => Reply(
        utteranceId: json['id'] as int,
        english: json['reply_english'] as String? ?? '',
        spoken: json['reply_spoken'] as String? ?? '',
        spokenLanguage: json['spoken_language'] as String? ?? '',
        latencyMs: json['latency_ms'] as int? ?? 0,
      );

  /// True when the speaker was already speaking English, so both fields hold
  /// the same sentence and only one should be shown.
  bool get isEnglishOnly =>
      spoken.isEmpty || spoken.trim() == english.trim();
}

enum LinkState { idle, connecting, listening, failed }

class TranslationClient extends ChangeNotifier {
  final AudioRecorder _recorder = AudioRecorder();

  WebSocketChannel? _channel;
  StreamSubscription<Uint8List>? _audioSub;
  bool _usingPlayback = false;
  StreamSubscription? _socketSub;

  LinkState state = LinkState.idle;
  String? errorMessage;
  final List<Utterance> utterances = [];

  /// Replies keyed by the utterance they answer.
  final Map<int, Reply> replies = {};

  /// Null until the server reports whether it has a reply model loaded.
  bool? replyAvailable;

  /// Current input loudness, 0 to 1, for the level meter. This is measured
  /// before the server's gain is applied, so it reflects what the microphone
  /// is genuinely picking up.
  double level = 0.0;
  DateTime _lastLevelPush = DateTime.fromMillisecondsSinceEpoch(0);

  bool get isActive =>
      state == LinkState.listening || state == LinkState.connecting;

  /// Median-ish view of recent performance, for the status line.
  int? get lastLatencyMs {
    for (final u in utterances.reversed) {
      if (u.isFinal && u.latencyMs > 0) return u.latencyMs;
    }
    return null;
  }

  void clear() {
    utterances.clear();
    replies.clear();
    notifyListeners();
  }

  Future<void> start({
    required String serverUrl,
    required String source,
    required String target,
    double gain = 1.0,
    String audioSource = 'mic',
    String environment = 'normal',
    bool replyMode = false,
    String persona = 'partner',
    String responsiveness = 'balanced',
  }) async {
    if (isActive) return;

    _set(LinkState.connecting, error: null);

    if (!await _recorder.hasPermission()) {
      _fail('Microphone access is off. Turn it on in Settings, then Apps, '
          'then Live Translate, then Permissions.');
      return;
    }

    // Check the plain HTTP health endpoint first. It separates "the server
    // isn't reachable" from "it's reachable but the socket failed", which are
    // very different problems and otherwise look identical.
    final health = _healthUrl(serverUrl);
    if (health == null) {
      _fail('That server address is not valid. It should look like\n'
          'ws://192.168.1.50:8000/ws');
      return;
    }
    try {
      final res =
          await http.get(health).timeout(const Duration(seconds: 6));
      if (res.statusCode != 200) {
        _fail('The server answered with status ${res.statusCode}. '
            'Check that it finished loading its models.');
        return;
      }
      final body = jsonDecode(res.body) as Map<String, dynamic>;
      if (body['status'] != 'ok') {
        _fail('The server is still starting up. Wait for it to print '
            '"Models ready", then try again.');
        return;
      }
    } catch (_) {
      _fail("Can't reach $health\n\n"
          'Check that the server is running, that both devices are on the '
          'same network, and that Windows Firewall allows inbound TCP on '
          'port 8000.');
      return;
    }

    try {
      final channel = WebSocketChannel.connect(Uri.parse(serverUrl));
      await channel.ready.timeout(const Duration(seconds: 8));
      _channel = channel;

      channel.sink.add(jsonEncode({
        'type': 'config',
        'source': source,
        'target': target,
        'gain': gain,
        'environment': environment,
        'reply': replyMode,
        'persona': persona,
        'responsiveness': responsiveness,
      }));

      _socketSub = channel.stream.listen(
        _onMessage,
        onError: (_) => _fail('Lost the connection to the server.'),
        onDone: () {
          if (state == LinkState.listening) {
            _fail('The server closed the connection.');
          }
        },
      );
    } catch (_) {
      _fail('The health check passed but the WebSocket would not open.\n\n'
          'Make sure the address ends in /ws and starts with ws://');
      return;
    }

    // Two possible sources. Device audio taps what other apps are playing,
    // which is the only way to translate something you're hearing on
    // headphones — a microphone can't reach it.
    if (isPlaybackSource(audioSource)) {
      if (!await PlaybackCapture.isSupported()) {
        _fail('Capturing device audio needs Android 10 or later.\n\n'
            'Choose a microphone source in settings instead.');
        return;
      }
      if (!await PlaybackCapture.start()) {
        _fail('Screen capture permission was declined.\n\n'
            'Android asks every session and will not remember it. Tap Start '
            'listening again and allow it.');
        return;
      }
      _usingPlayback = true;
      _audioSub = PlaybackCapture.audio.listen(
        (chunk) {
          _measure(chunk);
          _channel?.sink.add(chunk);
        },
        onError: (_) => _fail('Device audio capture stopped.'),
      );
      await WakelockPlus.enable();
      _set(LinkState.listening);
      return;
    }

    // Native microphone capture — AudioRecord under the hood, not a webview.
    final stream = await _recorder.startStream(
      RecordConfig(
        encoder: AudioEncoder.pcm16bits,
        sampleRate: 16000,
        numChannels: 1,
        // These three are RecordConfig properties, not Android ones.
        // autoGain lets the device lift quiet or distant speech. The other
        // two are off deliberately: they're tuned for phone calls and duck
        // anything that isn't a close voice, which is wrong for a room mic.
        autoGain: true,
        echoCancel: false,
        // On outdoors and in noisy rooms, let the phone's own suppression run
        // first. It works in the hardware audio path, costs no latency, and
        // the manufacturer tuned it for this. Off elsewhere, where it mainly
        // dulls quiet speech.
        noiseSuppress: kSuppressEnvironments.contains(environment),
        androidConfig: AndroidRecordConfig(
          audioSource: _sourceFor(audioSource),
        ),
      ),
    );

    _audioSub = stream.listen(
      (chunk) {
        _measure(chunk);
        _channel?.sink.add(chunk);
      },
      onError: (_) => _fail('The microphone stopped unexpectedly.'),
    );

    // A screen that sleeps mid-conversation drops the socket.
    await WakelockPlus.enable();
    _set(LinkState.listening);
  }

  Future<void> stop() async {
    await _audioSub?.cancel();
    await _socketSub?.cancel();
    _audioSub = null;
    _socketSub = null;

    if (_usingPlayback) {
      await PlaybackCapture.stop();
      _usingPlayback = false;
    } else if (await _recorder.isRecording()) {
      await _recorder.stop();
    }
    await _channel?.sink.close();
    _channel = null;

    level = 0.0;
    await WakelockPlus.disable();
    if (state != LinkState.failed) _set(LinkState.idle);
  }

  /// Send an English sentence to be translated into whatever was last spoken.
  /// The result comes back as an outgoing entry in the transcript.
  void compose(String text) {
    final trimmed = text.trim();
    if (trimmed.isEmpty || _channel == null) return;
    _channel!.sink.add(jsonEncode({'type': 'compose', 'text': trimmed}));
  }

  AndroidAudioSource _sourceFor(String key) => switch (key) {
        'camcorder' => AndroidAudioSource.camcorder,
        'voiceRecognition' => AndroidAudioSource.voiceRecognition,
        'voiceCommunication' => AndroidAudioSource.voiceCommunication,
        'unprocessed' => AndroidAudioSource.unprocessed,
        _ => AndroidAudioSource.mic,
      };

  /// RMS of a PCM16 chunk, mapped onto 0..1 across a 60 dB window.
  ///
  /// Wrapped in a guard on purpose: this runs inside the audio stream's
  /// listener, and an exception thrown here would tear down the subscription
  /// and stop recording with no visible error. A level meter must never be
  /// able to break capture.
  void _measure(Uint8List chunk) {
    try {
      if (chunk.lengthInBytes < 2) return;
      // sublistView respects the chunk's byte offset. asInt16List would throw
      // if that offset happened to be odd.
      final view = ByteData.sublistView(chunk);
      final count = view.lengthInBytes ~/ 2;
      if (count == 0) return;

      // Every fourth sample is plenty for a meter and keeps this cheap.
      var sum = 0.0;
      var n = 0;
      for (var i = 0; i < count; i += 4) {
        final v = view.getInt16(i * 2, Endian.little) / 32768.0;
        sum += v * v;
        n++;
      }
      if (n == 0) return;

      final rms = math.sqrt(sum / n);
      final db = rms > 1e-7 ? 20 * math.log(rms) / math.ln10 : -90.0;
      final next = ((db + 60) / 60).clamp(0.0, 1.0);

      // Jump up instantly, fall back slowly, so the bar doesn't strobe.
      level = next > level ? next : level * 0.82 + next * 0.18;

      final now = DateTime.now();
      if (now.difference(_lastLevelPush).inMilliseconds >= 60) {
        _lastLevelPush = now;
        notifyListeners();
      }
    } catch (_) {
      // Never let metering interfere with capture.
    }
  }

  void _onMessage(dynamic raw) {
    final msg = jsonDecode(raw as String) as Map<String, dynamic>;

    if (msg['type'] == 'ready') {
      replyAvailable = msg['reply_available'] as bool?;
      notifyListeners();
      return;
    }

    if (msg['type'] == 'composed') {
      utterances.add(Utterance.fromJson(msg));
      notifyListeners();
      return;
    }

    if (msg['type'] == 'reply') {
      final reply = Reply.fromJson(msg);
      replies[reply.utteranceId] = reply;
      notifyListeners();
      return;
    }

    final line = Utterance.fromJson(msg);
    final at = utterances.indexWhere((u) => u.id == line.id);
    if (at >= 0) {
      utterances[at] = line;
    } else {
      utterances.add(line);
    }
    notifyListeners();
  }

  /// ws://host:8000/ws  ->  http://host:8000/health
  Uri? _healthUrl(String serverUrl) {
    try {
      final uri = Uri.parse(serverUrl);
      if (uri.host.isEmpty) return null;
      return uri.replace(
        scheme: uri.scheme == 'wss' ? 'https' : 'http',
        path: '/health',
      );
    } catch (_) {
      return null;
    }
  }

  void _set(LinkState next, {String? error}) {
    state = next;
    if (error != null || next != LinkState.failed) errorMessage = error;
    notifyListeners();
  }

  void _fail(String message) {
    _set(LinkState.failed, error: message);
    unawaited(stop());
    state = LinkState.failed;
    errorMessage = message;
    notifyListeners();
  }

  @override
  void dispose() {
    stop();
    _recorder.dispose();
    super.dispose();
  }
}
