import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../settings.dart';
import '../theme.dart';

/// Returns the edited settings, or null if the sheet was dismissed.
Future<Settings?> showSettingsSheet(BuildContext context, Settings current) {
  return showModalBottomSheet<Settings>(
    context: context,
    backgroundColor: kInk,
    isScrollControlled: true,
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
    ),
    builder: (context) => _SettingsSheet(current: current),
  );
}

class _SettingsSheet extends StatefulWidget {
  final Settings current;
  const _SettingsSheet({required this.current});

  @override
  State<_SettingsSheet> createState() => _SettingsSheetState();
}

class _SettingsSheetState extends State<_SettingsSheet> {
  late final TextEditingController _url =
      TextEditingController(text: widget.current.serverUrl);
  late String _source = widget.current.source;
  late String _target = widget.current.target;
  late double _gain = widget.current.gain;
  late String _audioSource = widget.current.audioSource;
  late String _environment = widget.current.environment;
  late bool _replyMode = widget.current.replyMode;
  late String _persona = widget.current.persona;
  late String _responsiveness = widget.current.responsiveness;

  @override
  void dispose() {
    _url.dispose();
    super.dispose();
  }

  void _save() {
    Navigator.pop(
      context,
      Settings(
        serverUrl: _url.text.trim(),
        source: _source,
        target: _target,
        gain: _gain,
        audioSource: _audioSource,
        environment: _environment,
        replyMode: _replyMode,
        persona: _persona,
        responsiveness: _responsiveness,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final targets = Map.of(kLanguages)..remove('auto');

    return Padding(
      padding: EdgeInsets.fromLTRB(
        20,
        20,
        20,
        MediaQuery.of(context).viewInsets.bottom + 30,
      ),
      child: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Container(
                width: 36,
                height: 4,
                decoration: BoxDecoration(
                  color: kRule,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            const SizedBox(height: 24),
            Text('SERVER', style: kLabel),
            const SizedBox(height: 8),
            TextField(
              controller: _url,
              style: const TextStyle(color: kPaper, fontSize: 15),
              keyboardType: TextInputType.url,
              autocorrect: false,
              inputFormatters: [
                FilteringTextInputFormatter.singleLineFormatter,
              ],
              decoration: const InputDecoration(
                hintText: 'ws://192.168.1.50:8000/ws',
                hintStyle: TextStyle(color: kRule),
                enabledBorder:
                    UnderlineInputBorder(borderSide: BorderSide(color: kRule)),
                focusedBorder:
                    UnderlineInputBorder(borderSide: BorderSide(color: kLive)),
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'The address printed by the server when it starts. On the same '
              'Wi-Fi as this phone, or a Tailscale address from anywhere.',
              style: TextStyle(fontSize: 12, color: kMuted, height: 1.5),
            ),
            const SizedBox(height: 28),
            _picker(
              label: 'SPOKEN LANGUAGE',
              value: _source,
              options: kLanguages,
              onChanged: (v) => setState(() => _source = v),
              note: 'Detection needs about two seconds of speech to be '
                  'reliable. Set it explicitly if you know it.',
            ),
            const SizedBox(height: 24),
            _picker(
              label: 'TRANSLATE INTO',
              value: _target,
              options: targets,
              onChanged: (v) => setState(() => _target = v),
            ),
            const SizedBox(height: 28),
            _picker(
              label: 'MICROPHONE',
              value: _audioSource,
              options: kAudioSources,
              onChanged: (v) => setState(() => _audioSource = v),
              note: 'Device audio translates what other apps are playing — '
                  'use it for video, and it works while you are wearing '
                  'headphones. Standard suits most in-person situations; Room '
                  'when the speaker is further away.',
            ),
            const SizedBox(height: 28),
            _picker(
              label: 'RESPONSIVENESS',
              value: _responsiveness,
              options: kResponsiveness,
              onChanged: (v) => setState(() => _responsiveness = v),
              note: 'Most of the delay is the pause after you stop speaking, '
                  'while the server confirms you have finished. Fast shortens '
                  'that but may cut in mid-sentence.',
            ),
            const SizedBox(height: 28),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              value: _replyMode,
              activeColor: kLive,
              inactiveTrackColor: kRule,
              title: Text('GENERATE REPLIES', style: kLabel),
              subtitle: const Text(
                'Compose a response to each finished sentence, in English and '
                'in the language spoken. Adds a second or two per turn.',
                style: TextStyle(fontSize: 12, color: kMuted, height: 1.5),
              ),
              onChanged: (v) => setState(() => _replyMode = v),
            ),
            if (_replyMode) ...[
              const SizedBox(height: 20),
              _picker(
                label: 'REPLY STYLE',
                value: _persona,
                options: kPersonas,
                onChanged: (v) => setState(() => _persona = v),
              ),
            ],
            const SizedBox(height: 28),
            _picker(
              label: 'ENVIRONMENT',
              value: _environment,
              options: kEnvironments,
              onChanged: (v) => setState(() => _environment = v),
              note: 'Street turns on noise removal and the phone\'s own '
                  'suppression, and is the one to use outdoors. Quiet room is '
                  'the most sensitive, for a silent space.',
            ),
            const SizedBox(height: 28),
            Row(
              children: [
                Text('SENSITIVITY', style: kLabel),
                const Spacer(),
                Text('${_gain.toStringAsFixed(1)}x',
                    style: kLabel.copyWith(color: kLive)),
              ],
            ),
            Slider(
              value: _gain,
              min: 1.0,
              max: 8.0,
              divisions: 14,
              activeColor: kLive,
              inactiveColor: kRule,
              onChanged: (v) => setState(() => _gain = v),
            ),
            const Text(
              'Boosts the signal before speech detection. Raise it if quiet '
              'speech is being missed; lower it if background noise keeps '
              'triggering. Watch the level meter while listening — peaks '
              'reaching the amber marks are ideal.',
              style: TextStyle(fontSize: 12, color: kMuted, height: 1.5),
            ),
            const SizedBox(height: 32),
            SizedBox(
              width: double.infinity,
              height: 50,
              child: FilledButton(
                onPressed: _save,
                style: FilledButton.styleFrom(
                  backgroundColor: kLive,
                  foregroundColor: kInk,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(25),
                  ),
                ),
                child: const Text(
                  'Save',
                  style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _picker({
    required String label,
    required String value,
    required Map<String, String> options,
    required ValueChanged<String> onChanged,
    String? note,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: kLabel),
        const SizedBox(height: 4),
        DropdownButton<String>(
          value: value,
          isExpanded: true,
          dropdownColor: const Color(0xFF1C1916),
          underline: Container(height: 1, color: kRule),
          style: const TextStyle(color: kPaper, fontSize: 15),
          items: options.entries
              .map((e) => DropdownMenuItem(value: e.key, child: Text(e.value)))
              .toList(),
          onChanged: (v) {
            if (v != null) onChanged(v);
          },
        ),
        if (note != null) ...[
          const SizedBox(height: 8),
          Text(
            note,
            style: const TextStyle(fontSize: 12, color: kMuted, height: 1.5),
          ),
        ],
      ],
    );
  }
}
