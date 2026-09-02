import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../settings.dart';
import '../theme.dart';
import '../translation_client.dart';
import 'settings_sheet.dart';

class TranslateScreen extends StatefulWidget {
  const TranslateScreen({super.key});

  @override
  State<TranslateScreen> createState() => _TranslateScreenState();
}

class _TranslateScreenState extends State<TranslateScreen> {
  final _client = TranslationClient();
  final _scroll = ScrollController();
  final _compose = TextEditingController();
  final _composeFocus = FocusNode();

  Settings? _settings;
  int _lastCount = 0;

  @override
  void initState() {
    super.initState();
    _client.addListener(_onClientChange);
    Settings.load().then((s) => setState(() => _settings = s));
  }

  void _onClientChange() {
    if (!mounted) return;
    setState(() {});
    if (_client.utterances.length != _lastCount) {
      _lastCount = _client.utterances.length;
      _scrollToEnd();
    }
  }

  void _scrollToEnd() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) {
        _scroll.animateTo(
          _scroll.position.maxScrollExtent,
          duration: const Duration(milliseconds: 220),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  void dispose() {
    _client.removeListener(_onClientChange);
    _client.dispose();
    _scroll.dispose();
    _compose.dispose();
    _composeFocus.dispose();
    super.dispose();
  }

  Future<void> _toggle() async {
    final s = _settings;
    if (s == null) return;
    if (_client.isActive) {
      await _client.stop();
    } else {
      await _client.start(
        serverUrl: s.serverUrl,
        source: s.source,
        target: s.target,
        gain: s.gain,
        audioSource: s.audioSource,
        environment: s.environment,
        replyMode: s.replyMode,
        persona: s.persona,
        responsiveness: s.responsiveness,
      );
    }
  }

  void _send() {
    final text = _compose.text;
    if (text.trim().isEmpty) return;
    _client.compose(text);
    _compose.clear();
    _composeFocus.requestFocus();
  }

  Future<void> _openSettings() async {
    final s = _settings;
    if (s == null) return;
    final updated = await showSettingsSheet(context, s);
    if (updated != null) {
      await updated.save();
      setState(() => _settings = updated);
    }
  }

  void _copyAll() {
    final buffer = StringBuffer();
    for (final u in _client.utterances.where((u) => u.isFinal)) {
      buffer.writeln(u.translation.isEmpty ? u.transcript : u.translation);
      final r = _client.replies[u.id];
      if (r != null) buffer.writeln('> ${r.english}');
    }
    final text = buffer.toString().trim();
    if (text.isEmpty) return;
    Clipboard.setData(ClipboardData(text: text));
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Transcript copied'),
        duration: Duration(seconds: 2),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_settings == null) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator(color: kLive)),
      );
    }
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            _header(),
            if (_client.errorMessage != null) _errorPanel(),
            Expanded(
              child: _client.utterances.isEmpty ? _empty() : _transcript(),
            ),
            if (_client.state == LinkState.listening) _composeBar(),
            _controls(),
          ],
        ),
      ),
    );
  }

  // ------------------------------------------------------------------

  Widget _header() {
    final (label, colour) = switch (_client.state) {
      LinkState.listening => ('LISTENING', kLive),
      LinkState.connecting => ('CONNECTING', kAmber),
      LinkState.failed => ('DISCONNECTED', kAlert),
      LinkState.idle => ('IDLE', kMuted),
    };
    final latency = _client.lastLatencyMs;

    return Container(
      padding: const EdgeInsets.fromLTRB(20, 14, 8, 12),
      decoration: const BoxDecoration(
        border: Border(bottom: BorderSide(color: kRule)),
      ),
      child: Row(
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(shape: BoxShape.circle, color: colour),
          ),
          const SizedBox(width: 10),
          Text(label, style: kLabel.copyWith(color: colour)),
          if (latency != null && _client.state == LinkState.listening) ...[
            const SizedBox(width: 12),
            Text('${latency}ms', style: kLabel),
          ],
          if (_client.state == LinkState.listening) ...[
            const SizedBox(width: 12),
            Expanded(child: _levelMeter()),
          ] else
            const Spacer(),
          if (_client.utterances.isNotEmpty) ...[
            IconButton(
              onPressed: _copyAll,
              icon: const Icon(Icons.copy_all_outlined, size: 20),
              color: kMuted,
              tooltip: 'Copy transcript',
            ),
            IconButton(
              onPressed: _client.clear,
              icon: const Icon(Icons.delete_outline, size: 20),
              color: kMuted,
              tooltip: 'Clear',
            ),
          ],
          IconButton(
            onPressed: _client.isActive ? null : _openSettings,
            icon: const Icon(Icons.tune, size: 20),
            color: kMuted,
            disabledColor: kRule,
            tooltip: 'Settings',
          ),
        ],
      ),
    );
  }

  /// Twelve segments across a 60 dB window. Anything reaching the amber zone
  /// is comfortably loud enough for Whisper; a bar that barely moves means the
  /// microphone, not the model, is the problem.
  Widget _levelMeter() {
    const segments = 12;
    final lit = (_client.level * segments).round();
    return Row(
      mainAxisAlignment: MainAxisAlignment.end,
      children: List.generate(segments, (i) {
        final on = i < lit;
        final colour = i >= segments - 2
            ? kAlert
            : (i >= segments - 5 ? kAmber : kLive);
        return Container(
          width: 3,
          height: 6.0 + i * 1.1,
          margin: const EdgeInsets.symmetric(horizontal: 1.5),
          decoration: BoxDecoration(
            color: on ? colour : kRule,
            borderRadius: BorderRadius.circular(1.5),
          ),
        );
      }),
    );
  }

  Widget _errorPanel() {
    return Container(
      width: double.infinity,
      color: const Color(0xFF3A1F1C),
      padding: const EdgeInsets.fromLTRB(20, 14, 20, 14),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Text(
              _client.errorMessage!,
              style: const TextStyle(
                fontSize: 13,
                height: 1.5,
                color: Color(0xFFE8B4A8),
              ),
            ),
          ),
          const SizedBox(width: 8),
          TextButton(
            onPressed: _openSettings,
            style: TextButton.styleFrom(
              foregroundColor: const Color(0xFFE8B4A8),
              padding: const EdgeInsets.symmetric(horizontal: 12),
            ),
            child: const Text('Settings'),
          ),
        ],
      ),
    );
  }

  Widget _empty() {
    final s = _settings!;
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 44),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              languageName(s.source),
              style: const TextStyle(
                fontSize: 15,
                color: kMuted,
                fontWeight: FontWeight.w400,
              ),
            ),
            const SizedBox(height: 6),
            const Icon(Icons.arrow_downward, size: 18, color: kRule),
            const SizedBox(height: 6),
            Text(
              languageName(s.target),
              style: const TextStyle(
                fontSize: 26,
                color: kPaper,
                fontWeight: FontWeight.w300,
              ),
            ),
            const SizedBox(height: 20),
            const Text(
              'Start listening, then speak normally. Each sentence is '
              'translated when you pause.',
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 14, color: kMuted, height: 1.55),
            ),
          ],
        ),
      ),
    );
  }

  Widget _transcript() {
    return ListView.builder(
      controller: _scroll,
      padding: const EdgeInsets.fromLTRB(20, 22, 20, 10),
      itemCount: _client.utterances.length,
      itemBuilder: (context, i) {
        final line = _client.utterances[i];
        final settled = line.isFinal;
        final reply = _client.replies[line.id];
        final headline =
            line.translation.isEmpty ? line.transcript : line.translation;

        if (line.isOutgoing) return _outgoing(line);

        return Padding(
          padding: const EdgeInsets.only(bottom: 26),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Text(
                    line.isPassthrough
                        ? '${line.sourceLanguage.toUpperCase()}  '
                            '(no translation needed)'
                        : '${line.sourceLanguage.toUpperCase()}  ->  '
                            '${line.targetLanguage.toUpperCase()}',
                    style: kLabel.copyWith(
                      color: line.isPassthrough ? kAmber : kMuted,
                    ),
                  ),
                  if (!settled) ...[
                    const SizedBox(width: 8),
                    const SizedBox(
                      width: 8,
                      height: 8,
                      child: CircularProgressIndicator(
                        strokeWidth: 1.4,
                        color: kMuted,
                      ),
                    ),
                  ],
                ],
              ),
              const SizedBox(height: 7),
              SelectableText(
                headline,
                style: TextStyle(
                  fontSize: 21,
                  height: 1.4,
                  color: settled ? kPaper : kMuted,
                ),
              ),
              // Only show the original underneath when it actually differs.
              if (!line.isPassthrough) ...[
                const SizedBox(height: 7),
                Text(
                  line.transcript,
                  style: const TextStyle(
                    fontSize: 13.5,
                    height: 1.45,
                    color: kMuted,
                  ),
                ),
              ],
              if (reply != null) _replyBlock(reply),
            ],
          ),
        );
      },
    );
  }

  /// Something the user typed. Right-aligned and tinted so the transcript
  /// reads as a conversation rather than one undifferentiated stream.
  Widget _outgoing(Utterance line) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 26, left: 40),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          Text(
            'YOU  ->  ${line.targetLanguage.toUpperCase()}',
            style: kLabel.copyWith(color: kLive),
          ),
          const SizedBox(height: 7),
          SelectableText(
            line.translation.isEmpty ? line.transcript : line.translation,
            textAlign: TextAlign.right,
            style: const TextStyle(fontSize: 20, height: 1.4, color: kPaper),
          ),
          if (!line.isPassthrough) ...[
            const SizedBox(height: 6),
            Text(
              line.transcript,
              textAlign: TextAlign.right,
              style: const TextStyle(
                fontSize: 13.5,
                height: 1.45,
                color: kMuted,
              ),
            ),
          ],
        ],
      ),
    );
  }

  /// Sits under the utterance it answers, indented behind a rule, so a glance
  /// tells you which lines were heard and which were generated.
  Widget _replyBlock(Reply reply) {
    return Container(
      margin: const EdgeInsets.only(top: 14, left: 2),
      padding: const EdgeInsets.only(left: 14),
      decoration: const BoxDecoration(
        border: Border(left: BorderSide(color: kLive, width: 2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('REPLY', style: kLabel.copyWith(color: kLive)),
          const SizedBox(height: 6),
          if (!reply.isEnglishOnly) ...[
            SelectableText(
              reply.spoken,
              style: const TextStyle(
                fontSize: 18,
                height: 1.4,
                color: kPaper,
              ),
            ),
            const SizedBox(height: 6),
          ],
          SelectableText(
            reply.english,
            style: TextStyle(
              fontSize: reply.isEnglishOnly ? 18 : 13.5,
              height: 1.45,
              color: reply.isEnglishOnly ? kPaper : kMuted,
            ),
          ),
        ],
      ),
    );
  }

  /// Type an English reply; the server translates it into whatever language
  /// was last spoken. Only shown while connected, since it needs the socket.
  Widget _composeBar() {
    return Container(
      padding: const EdgeInsets.fromLTRB(20, 10, 12, 10),
      decoration: const BoxDecoration(
        border: Border(top: BorderSide(color: kRule)),
      ),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _compose,
              focusNode: _composeFocus,
              style: const TextStyle(color: kPaper, fontSize: 15),
              minLines: 1,
              maxLines: 3,
              textInputAction: TextInputAction.send,
              onSubmitted: (_) => _send(),
              decoration: const InputDecoration(
                hintText: 'Reply in English',
                hintStyle: TextStyle(color: kMuted, fontSize: 15),
                border: InputBorder.none,
                isDense: true,
              ),
            ),
          ),
          IconButton(
            onPressed: _send,
            icon: const Icon(Icons.arrow_upward, size: 20),
            color: kInk,
            style: IconButton.styleFrom(backgroundColor: kLive),
          ),
        ],
      ),
    );
  }

  Widget _controls() {
    final s = _settings!;
    final active = _client.isActive;
    return Container(
      padding: const EdgeInsets.fromLTRB(20, 14, 20, 22),
      decoration: const BoxDecoration(
        border: Border(top: BorderSide(color: kRule)),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${languageName(s.source)}  to  ${languageName(s.target)}',
                  style: const TextStyle(fontSize: 13, color: kMuted),
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 3),
                Text(
                  Uri.tryParse(s.serverUrl)?.host ?? s.serverUrl,
                  style: kLabel.copyWith(fontSize: 10),
                ),
              ],
            ),
          ),
          const SizedBox(width: 12),
          SizedBox(
            height: 52,
            child: FilledButton(
              onPressed: _client.state == LinkState.connecting ? null : _toggle,
              style: FilledButton.styleFrom(
                backgroundColor: active ? kRule : kLive,
                foregroundColor: active ? kPaper : kInk,
                disabledBackgroundColor: kRule,
                padding: const EdgeInsets.symmetric(horizontal: 26),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(26),
                ),
              ),
              child: Text(
                switch (_client.state) {
                  LinkState.connecting => 'Connecting',
                  LinkState.listening => 'Stop',
                  _ => 'Start listening',
                },
                style: const TextStyle(
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
