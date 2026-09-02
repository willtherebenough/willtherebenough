import 'package:shared_preferences/shared_preferences.dart';

/// Whisper language codes the server knows how to translate between.
/// These must stay in sync with NLLB_CODES in the server's main.py.
const Map<String, String> kLanguages = {
  'auto': 'Detect automatically',
  'en': 'English',
  'es': 'Spanish',
  'fr': 'French',
  'de': 'German',
  'it': 'Italian',
  'pt': 'Portuguese',
  'nl': 'Dutch',
  'sv': 'Swedish',
  'pl': 'Polish',
  'cs': 'Czech',
  'ro': 'Romanian',
  'hu': 'Hungarian',
  'el': 'Greek',
  'ru': 'Russian',
  'uk': 'Ukrainian',
  'tr': 'Turkish',
  'ar': 'Arabic',
  'he': 'Hebrew',
  'fa': 'Persian',
  'hi': 'Hindi',
  'bn': 'Bengali',
  'ur': 'Urdu',
  'ta': 'Tamil',
  'th': 'Thai',
  'vi': 'Vietnamese',
  'id': 'Indonesian',
  'ms': 'Malay',
  'tl': 'Tagalog',
  'zh': 'Chinese',
  'ja': 'Japanese',
  'ko': 'Korean',
  'sw': 'Swahili',
  'yo': 'Yoruba',
  'zu': 'Zulu',
  'ha': 'Hausa',
  'am': 'Amharic',
  'so': 'Somali',
  'ig': 'Igbo',
};

String languageName(String code) => kLanguages[code] ?? code.toUpperCase();

/// Android microphone sources, and what each is actually good for.
///
/// These map to AudioSource constants in the Android framework. The default
/// most speech apps use is voiceRecognition, but it's tuned for someone
/// holding the phone near their mouth: on many devices it disables automatic
/// gain control, which is exactly wrong when the speaker is across a room.
const Map<String, String> kAudioSources = {
  // Not a microphone at all: taps what other apps are playing. Android 10+.
  'playback': 'Device audio  (other apps, not the mic)',
  'mic': 'Standard  (balanced, good default)',
  'camcorder': 'Room  (higher gain, distant speech)',
  'voiceRecognition': 'Close talk  (phone near mouth)',
  'voiceCommunication': 'Call  (strongest noise rejection)',
  'unprocessed': 'Raw  (no processing; needs high gain)',
};

/// Sensitivity and noise rejection are the same dial turned opposite ways.
/// There is no single setting that both catches a quiet voice across a room
/// and ignores a television, so this picks which way to err.
const Map<String, String> kEnvironments = {
  'quiet': 'Quiet room  (most sensitive)',
  'normal': 'Normal  (balanced)',
  'noisy': 'Noisy  (music, TV indoors)',
  'street': 'Street  (outdoors, traffic)',
};

/// Environments where the phone's own noise suppression should be switched on.
/// It runs in the audio hardware path at no latency cost and is tuned by the
/// manufacturer for exactly this — worth having before anything reaches the
/// server. Off in quiet settings, where it only dulls the signal.
const Set<String> kSuppressEnvironments = {'noisy', 'street'};

/// Device audio is already a clean digital signal, so the noise and
/// sensitivity machinery meant for a microphone in a room does not apply.
bool isPlaybackSource(String source) => source == 'playback';

/// What the reply model should be doing. The app sends the key; the server
/// holds the actual prompt.
const Map<String, String> kPersonas = {
  'partner': 'Conversation partner  (practice)',
  'assistant': 'Assistant  (answers questions)',
  'interpreter': 'Suggested reply  (what to say back)',
};

/// Trade-off between how quickly a translation appears and how carefully the
/// model decodes it. Most of the delay people notice is the pause after they
/// stop talking, not the model.
const Map<String, String> kResponsiveness = {
  'fast': 'Fast  (shortest pause)',
  'balanced': 'Balanced',
  'accurate': 'Accurate  (slower, beam search)',
};

class Settings {
  String serverUrl;
  String source;
  String target;

  /// Multiplier applied to the audio on the server, before voice detection.
  /// Raising this makes quiet speech register; too high and room noise does
  /// too. 1x is untouched.
  double gain;

  /// Key into [kAudioSources].
  String audioSource;

  /// Key into [kEnvironments]. Controls how aggressively the server rejects
  /// non-speech before it reaches the model.
  String environment;

  /// Generate a spoken reply to each finished utterance.
  bool replyMode;

  /// Key into [kPersonas].
  String persona;

  /// Key into [kResponsiveness].
  String responsiveness;

  Settings({
    required this.serverUrl,
    required this.source,
    required this.target,
    required this.gain,
    required this.audioSource,
    required this.environment,
    required this.replyMode,
    required this.persona,
    required this.responsiveness,
  });

  static const _kUrl = 'server_url';
  static const _kSource = 'source_lang';
  static const _kTarget = 'target_lang';
  static const _kGain = 'input_gain';
  static const _kAudioSource = 'audio_source';
  static const _kEnvironment = 'environment';
  static const _kReply = 'reply_mode';
  static const _kPersona = 'persona';
  static const _kResponsiveness = 'responsiveness';

  static Future<Settings> load() async {
    final prefs = await SharedPreferences.getInstance();
    return Settings(
      serverUrl: prefs.getString(_kUrl) ?? 'ws://192.168.1.108:8000/ws',
      source: prefs.getString(_kSource) ?? 'auto',
      target: prefs.getString(_kTarget) ?? 'en',
      gain: prefs.getDouble(_kGain) ?? 2.0,
      audioSource: prefs.getString(_kAudioSource) ?? 'mic',
      environment: prefs.getString(_kEnvironment) ?? 'normal',
      replyMode: prefs.getBool(_kReply) ?? false,
      persona: prefs.getString(_kPersona) ?? 'partner',
      responsiveness: prefs.getString(_kResponsiveness) ?? 'balanced',
    );
  }

  Future<void> save() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_kUrl, serverUrl);
    await prefs.setString(_kSource, source);
    await prefs.setString(_kTarget, target);
    await prefs.setDouble(_kGain, gain);
    await prefs.setString(_kAudioSource, audioSource);
    await prefs.setString(_kEnvironment, environment);
    await prefs.setBool(_kReply, replyMode);
    await prefs.setString(_kPersona, persona);
    await prefs.setString(_kResponsiveness, responsiveness);
  }
}
