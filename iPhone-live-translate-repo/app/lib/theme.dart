import 'package:flutter/material.dart';

// A dark, high-contrast palette. This app gets used in noisy rooms, at
// arm's length, often while looking at someone rather than the screen — so
// the translated line is the only thing given full contrast, and everything
// else recedes.
const kInk = Color(0xFF12100E);
const kPaper = Color(0xFFF2EFE9);
const kMuted = Color(0xFF8A8378);
const kRule = Color(0xFF2A2622);
const kLive = Color(0xFF4DD6A8);
const kAmber = Color(0xFFD9A441);
const kAlert = Color(0xFFD97757);

// Monospace for status text: it stops the header jittering as the latency
// figure changes width.
const kLabel = TextStyle(
  fontFamily: 'monospace',
  fontSize: 11,
  letterSpacing: 1.5,
  color: kMuted,
  height: 1.2,
);

ThemeData buildTheme() {
  return ThemeData(
    brightness: Brightness.dark,
    scaffoldBackgroundColor: kInk,
    colorScheme: const ColorScheme.dark(
      surface: kInk,
      primary: kLive,
      onPrimary: kInk,
      error: kAlert,
    ),
    snackBarTheme: const SnackBarThemeData(
      backgroundColor: Color(0xFF1C1916),
      contentTextStyle: TextStyle(color: kPaper),
      behavior: SnackBarBehavior.floating,
    ),
    useMaterial3: true,
  );
}
