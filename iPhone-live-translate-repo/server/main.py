"""
Live speech translation server — pure PyTorch.

The phone opens a WebSocket, sends one JSON config frame, then streams raw
16 kHz mono PCM16 audio as binary frames. The server segments the stream with
Silero VAD, transcribes each utterance with Whisper, translates with NLLB-200,
and pushes JSON results back down the same socket.

Every model here runs on PyTorch, so the CUDA libraries bundled with your torch
install are the only ones needed. Nothing else has to resolve its own DLLs.

Run:  python run_server.py
"""

import asyncio
import importlib.util
import json
import logging
import os
import re
import socket
import sys
import time
import wave
from collections import deque
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

# Windows consoles default to a legacy code page, so a transcript containing an
# accent or a non-Latin script would raise UnicodeEncodeError from inside the
# logger. Set this before anything writes output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Hugging Face caches with symlinks, which need Developer Mode or an elevated
# shell on Windows. It falls back to copying and warns on every launch; the
# copy is fine, so silence it.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


def load_vad():
    """
    Load Silero VAD without importing the silero_vad package.

    That package's __init__ imports torchaudio, a compiled extension that has
    to match your torch build exactly. When it doesn't, Windows raises
    'WinError 127: The specified procedure could not be found' — which says
    nothing useful about the real problem.

    All we need is the TorchScript model file, which ships inside the package.
    find_spec locates it without executing __init__, so torchaudio never enters
    the picture. The model is tiny and runs per 512-sample frame; keeping it on
    CPU avoids a GPU round trip for every 32 ms of audio.
    """
    spec = importlib.util.find_spec("silero_vad")  # does not run __init__
    if spec is None or not spec.origin:
        raise RuntimeError(
            "silero-vad is not installed. Run: pip install silero-vad"
        )
    jit_path = os.path.join(os.path.dirname(spec.origin), "data", "silero_vad.jit")
    if not os.path.exists(jit_path):
        raise RuntimeError(f"Silero VAD model missing at {jit_path}")
    model = torch.jit.load(jit_path, map_location="cpu")
    model.eval()
    return model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s"
)
log = logging.getLogger("live-translate")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SAMPLE_RATE = 16_000
VAD_FRAME = 512  # Silero v5 requires exactly 512 samples at 16 kHz

# large-v3-turbo has the same encoder as large-v3 but four decoder layers
# instead of thirty-two. Transcription quality is nearly identical and decoding
# is several times faster, which matters a lot when the decoder runs on every
# utterance. Set ASR_MODEL=openai/whisper-large-v3 if you want the full model.
ASR_MODEL = os.environ.get("ASR_MODEL", "openai/whisper-large-v3-turbo")
MT_MODEL = os.environ.get("MT_MODEL", "facebook/nllb-200-distilled-600M")
# Path to a LoRA adapter produced by finetune_whisper.py, if you have one.
ASR_ADAPTER = os.environ.get("ASR_ADAPTER", "")

# Shared secret. Empty means no authentication, which is fine on a LAN and a
# bad idea anywhere else — this server will happily transcribe audio for
# whoever reaches it and spend your GPU doing it.
#
# Reaching the server from outside the house should still go through a private
# network (see NETWORK.md) rather than a forwarded port. This token is the
# second line, not the first.
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "")

# How confident Silero has to be that a frame contains speech. 0.5 is Silero's
# default and is tuned for a close microphone; it drops quiet or distant speech.
# 0.35 picks up considerably more at the cost of occasionally triggering on
# background noise, which the hallucination filter then discards anyway.
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.45"))

# How far above the room's noise floor a frame has to be before it counts as
# speech. Silero alone will happily classify music and TV dialogue as speech,
# which keeps an utterance open indefinitely. This is what separates "someone
# is talking" from "the room is not silent".
SNR_DB = float(os.environ.get("SNR_DB", "6.0"))

# An utterance whose average speech probability is below this is discarded
# without ever reaching Whisper. Music tends to sit in the middle of Silero's
# range; real speech sits high.
MIN_MEAN_PROB = float(os.environ.get("MIN_MEAN_PROB", "0.6"))

# Whisper's own confidence. Transcriptions of music or noise score far worse
# than transcriptions of speech, so this catches what VAD lets through.
MIN_AVG_LOGPROB = float(os.environ.get("MIN_AVG_LOGPROB", "-1.0"))

# Presets the app can choose between. Sensitivity and noise rejection are the
# same dial turned opposite ways, so there is no single setting that both
# catches a quiet voice across a room and ignores a television.
# --- Conversational replies -------------------------------------------------
#
# Optional third stage. The transcript is translated to English, an English
# reply is generated, and that reply is translated back into whatever language
# was spoken. Small models are far stronger in English than multilingually, so
# routing through English gets better replies than prompting in French would.
#
# Set ENABLE_LLM=0 to run as a plain translator and free about 4 GB.
ENABLE_LLM = os.environ.get("ENABLE_LLM", "1") == "1"
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
REPLY_MAX_TOKENS = int(os.environ.get("REPLY_MAX_TOKENS", "90"))
HISTORY_TURNS = int(os.environ.get("HISTORY_TURNS", "6"))

# Replies are read on a phone screen mid-conversation, so brevity matters more
# than completeness. Both personas are told to keep it to a couple of sentences.
PERSONAS = {
    "partner": (
        "You are a warm, natural conversation partner helping someone practise "
        "a language. Reply to what they said the way a friend would: react, "
        "then ask something back to keep the conversation going. One or two "
        "short sentences. Never mention that you are an AI, never correct "
        "their grammar unless asked, and never explain yourself."
    ),
    "assistant": (
        "You are a concise, practical assistant. Answer what was said or asked "
        "directly and usefully in one or two short sentences. No preamble, no "
        "restating the question, no offers of further help."
    ),
    "interpreter": (
        "You are helping two people who do not share a language. Reply only "
        "with a short, natural thing the listener could say back. One or two "
        "sentences, plain spoken language, no commentary."
    ),
}

# The gap between someone finishing a sentence and the translation appearing
# is mostly two things: how long we wait to be sure they've stopped, and how
# hard Whisper works on the final pass. Beam search was costing several times
# greedy decoding for a barely measurable accuracy gain on short utterances.
# --- Adaptation -------------------------------------------------------------
#
# Two mechanisms, in increasing order of effort.
#
# 1. A glossary. Whisper accepts a text prompt that biases what it expects to
#    hear, which is the cheapest way to fix names, jargon and regional words it
#    keeps getting wrong. Costs nothing and works immediately.
#
# 2. A dataset. Every finished utterance can be written to disk as audio plus
#    transcript. Correct the mistakes and that becomes training data for
#    finetune_whisper.py. This is the slow path and needs a few hundred
#    corrected utterances before it beats the glossary.
COLLECT_DATA = os.environ.get("COLLECT_DATA", "0") == "1"
DATA_DIR = os.environ.get("DATA_DIR", "dataset")
GLOSSARY_PATH = os.environ.get("GLOSSARY", "glossary.json")

# Feeding the previous sentence back in as context measurably helps continuous
# rapid speech, where utterance boundaries fall mid-thought. Off by default
# elsewhere because it can also propagate an error forward.
USE_ROLLING_CONTEXT = os.environ.get("ROLLING_CONTEXT", "1") == "1"

RESPONSIVENESS = {
    "fast": {"min_silence_ms": 400, "final_beams": 1},
    "balanced": {"min_silence_ms": 600, "final_beams": 1},
    "accurate": {"min_silence_ms": 900, "final_beams": 5},
}

ENVIRONMENTS = {
    "quiet": {
        "threshold": 0.32, "snr_db": 3.0, "min_mean_prob": 0.50,
        "denoise": False, "suppress": False,
    },
    "normal": {
        "threshold": 0.45, "snr_db": 6.0, "min_mean_prob": 0.60,
        "denoise": False, "suppress": False,
    },
    "noisy": {
        "threshold": 0.60, "snr_db": 10.0, "min_mean_prob": 0.72,
        "denoise": True, "suppress": True,
    },
    # For a busy street: everything on, and the gates opened back up, because
    # outdoors the noise floor is high enough that strict gating starts
    # discarding real speech along with the traffic.
    "street": {
        "threshold": 0.50, "snr_db": 7.0, "min_mean_prob": 0.62,
        "denoise": True, "suppress": True,
    },
}

# Whisper was trained on loudness-normalised audio and degrades sharply on
# quiet input. Each utterance is scaled to roughly this RMS before inference.
TARGET_RMS = float(os.environ.get("TARGET_RMS", "0.06"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Running this on CPU is technically possible and practically useless — tens of
# seconds per utterance. Fail loudly instead, unless explicitly overridden.
ALLOW_CPU = os.environ.get("ALLOW_CPU", "0") == "1"


def preflight():
    """Report exactly which interpreter and torch build are in play."""
    build = torch.__version__
    log.info("Interpreter : %s", sys.executable)
    log.info("torch       : %s  (CUDA build: %s)", build, torch.version.cuda)
    log.info("cuda.is_available(): %s", torch.cuda.is_available())

    if torch.cuda.is_available():
        log.info("GPU         : %s", torch.cuda.get_device_name(0))
        return

    cpu_wheel = build.endswith("+cpu") or torch.version.cuda is None
    print()
    print("=" * 72)
    if cpu_wheel:
        print("  This environment has the CPU-only build of PyTorch.")
        print()
        print("  silero-vad requires torch, so installing requirements.txt")
        print("  first pulls the default PyPI wheel — CPU-only on Windows.")
        print("  Your other project works because it is a different venv.")
        print()
        print("  Fix, in this venv:")
        print("      pip uninstall -y torch torchvision torchaudio")
        print("      pip install -r requirements-torch.txt")
        print()
        print("  Edit requirements-torch.txt first so the version and index")
        print("  match your working project. To find them, run this there:")
        print("      python check_gpu.py")
        print("  It prints the exact pip line and requirements entry to use.")
        print()
        print("  Note: cu124 no longer carries current torch releases.")
        print("  cu130 is the default now; cu126 is for older drivers.")
    else:
        print("  PyTorch has CUDA support but no GPU is visible.")
        print()
        print("  Check that nvidia-smi runs, that CUDA_VISIBLE_DEVICES is not")
        print("  set to an empty value, and that no other process has taken")
        print("  the card exclusively.")
    print()
    print("  Confirm you are on the venv you expect:")
    print("      python -c \"import sys; print(sys.executable)\"")
    print()
    print("  To run on CPU anyway (slow, for testing only): set ALLOW_CPU=1")
    print("=" * 72)
    print()
    sys.stdout.flush()

    if not ALLOW_CPU:
        raise SystemExit(1)
    log.warning("ALLOW_CPU=1 — continuing on CPU. Expect many seconds per utterance.")


if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Whisper uses ISO 639-1; NLLB uses its own script-tagged codes.
NLLB_CODES = {
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "nl": "nld_Latn", "sv": "swe_Latn",
    "pl": "pol_Latn", "cs": "ces_Latn", "ro": "ron_Latn", "hu": "hun_Latn",
    "el": "ell_Grek", "ru": "rus_Cyrl", "uk": "ukr_Cyrl", "tr": "tur_Latn",
    "ar": "arb_Arab", "he": "heb_Hebr", "fa": "pes_Arab", "hi": "hin_Deva",
    "bn": "ben_Beng", "ur": "urd_Arab", "ta": "tam_Taml", "th": "tha_Thai",
    "vi": "vie_Latn", "id": "ind_Latn", "ms": "zsm_Latn", "tl": "tgl_Latn",
    "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang",
    "sw": "swh_Latn", "yo": "yor_Latn", "zu": "zul_Latn", "ha": "hau_Latn",
    "am": "amh_Ethi", "so": "som_Latn", "ig": "ibo_Latn",
}

# Whisper emits these on silence or noise. Drop them rather than show them.
HALLUCINATIONS = {
    "thank you.", "thanks for watching!", "thank you for watching.",
    "subscribe to my channel", "please subscribe", "you", "bye.",
    "thanks for watching.", ".", "so", "okay.", "oh.", "hmm.",
}


def _ms(n: int) -> int:
    """Milliseconds to samples."""
    return int(SAMPLE_RATE * n / 1000)


class Glossary:
    """
    Terms the models keep getting wrong, and what they should be.

    Reloaded from disk whenever the file changes, so you can edit it while the
    server is running and hear the difference on the next sentence.
    """

    def __init__(self, path: str):
        self.path = path
        self._mtime = 0.0
        self.terms: list[str] = []
        self.replacements: dict[str, dict[str, str]] = {}
        self.reload()

    def reload(self):
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            return
        if mtime == self._mtime:
            return
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.terms = [str(x) for x in data.get("terms", [])]
            self.replacements = {
                str(k): {str(a): str(b) for a, b in v.items()}
                for k, v in data.get("replacements", {}).items()
            }
            self._mtime = mtime
            log.info(
                "Glossary loaded: %d terms, %d languages with replacements",
                len(self.terms), len(self.replacements),
            )
        except Exception as exc:
            log.warning("Could not read %s: %s", self.path, exc)

    def prompt(self, context: str = "") -> str:
        """
        Text handed to Whisper as an expectation of what it's about to hear.
        Whisper caps this around 224 tokens, so keep it short — a long list
        dilutes the bias rather than strengthening it.
        """
        self.reload()
        parts = []
        if self.terms:
            parts.append(", ".join(self.terms[:60]))
        if context:
            parts.append(context)
        return " ".join(parts).strip()[:800]

    def apply(self, text: str, lang: str) -> str:
        """Fix translations the model renders wrongly no matter what."""
        table = self.replacements.get(lang)
        if not table or not text:
            return text
        for wrong, right in table.items():
            text = re.sub(
                rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE
            )
        return text


def save_example(audio: np.ndarray, meta: dict):
    """
    Write one utterance to the dataset as 16-bit WAV plus a JSON sidecar.

    Deliberately uses the stdlib wave module rather than adding a dependency,
    and writes the sidecar last so a crash mid-write leaves an obviously
    incomplete pair rather than a corrupt one.
    """
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        wav_path = os.path.join(DATA_DIR, f"{stamp}.wav")

        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16)
        with wave.open(wav_path, "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(SAMPLE_RATE)
            fh.writeframes(pcm.tobytes())

        meta = dict(meta, audio=os.path.basename(wav_path), corrected=False)
        with open(
            os.path.join(DATA_DIR, f"{stamp}.json"), "w", encoding="utf-8"
        ) as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
    except Exception:
        log.exception("could not save training example")


def _stft(x: np.ndarray, n_fft: int, hop: int):
    win = np.hanning(n_fft + 1)[:n_fft].astype(np.float32)  # periodic
    frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    return np.fft.rfft(x[idx] * win, axis=1), win


def _istft(spec: np.ndarray, win: np.ndarray, hop: int, length: int):
    frames = np.fft.irfft(spec, n=win.shape[0], axis=1) * win
    n_fft = win.shape[0]
    out = np.zeros(length, dtype=np.float64)
    norm = np.zeros(length, dtype=np.float64)
    sq = (win ** 2).astype(np.float64)
    for i in range(frames.shape[0]):
        s = i * hop
        e = min(s + n_fft, length)
        if e <= s:
            break
        out[s:e] += frames[i, : e - s]
        norm[s:e] += sq[: e - s]
    # Only divide where the windows genuinely overlap. Near the ends the sum
    # of squared windows tends to zero, and dividing by it amplifies the edge
    # samples by orders of magnitude — which is exactly the bug that made an
    # earlier version of this function louder than its input.
    good = norm > 1e-3
    out[good] /= norm[good]
    out[~good] = 0.0
    return out.astype(np.float32)


def denoise(
    audio: np.ndarray,
    over: float = 1.8,
    floor: float = 0.1,
    highpass_hz: float = 90.0,
):
    """
    Spectral subtraction: estimate the noise spectrum and take it away.

    The noise estimate is a low percentile of magnitude in each frequency bin
    across the utterance. That works because traffic, wind and room hum are
    roughly constant while speech is not — the quiet end of each bin's
    distribution is the noise, whatever the noise happens to be. No
    calibration step, and it adapts to wherever you are standing.

    `floor` keeps a fraction of the original magnitude rather than subtracting
    to zero. Subtracting all the way produces "musical noise": isolated
    warbling tones that Whisper cheerfully transcribes as words.

    numpy only, deliberately. A learned denoiser would do better, but nothing
    here needs a Windows wheel that has to match your torch build.
    """
    n_fft, hop = 512, 128
    if len(audio) < n_fft * 4:
        return audio

    # Pad so every real sample sits under a full set of overlapping windows.
    pad = n_fft
    padded = np.pad(audio.astype(np.float32), (pad, pad + n_fft))

    spec, win = _stft(padded, n_fft, hop)
    mag = np.abs(spec)
    phase = np.angle(spec)

    noise = np.percentile(mag, 15, axis=0, keepdims=True)
    clean = np.maximum(mag - over * noise, floor * mag)

    # Traffic rumble, wind buffeting and handling noise all live below the
    # bottom of the human voice. Discarding those bins outright removes more
    # street noise than the subtraction above does, and costs no speech: the
    # lowest male fundamentals sit around 85 Hz and the intelligibility is
    # carried far higher regardless.
    cutoff = int(highpass_hz * n_fft / SAMPLE_RATE)
    if cutoff > 0:
        clean[:, :cutoff] = 0.0

    out = _istft(clean * np.exp(1j * phase), win, hop, len(padded))
    return out[pad : pad + len(audio)]


def _dbfs(audio: np.ndarray) -> float:
    """Peak level in dBFS. -inf is silence, 0 is full scale."""
    if audio.size == 0:
        return -99.0
    peak = float(np.max(np.abs(audio)))
    return 20.0 * float(np.log10(peak)) if peak > 1e-9 else -99.0


def _normalize(audio: np.ndarray) -> np.ndarray:
    """
    Bring an utterance up to a consistent loudness before transcription.

    This is the single biggest win for a distant or quiet microphone. Whisper's
    training data was loudness-normalised, so feeding it audio 30 dB down costs
    real accuracy even when the words are perfectly audible to a human.

    Only ever boosts, never attenuates, and backs the gain off if it would
    clip.
    """
    if audio.size == 0:
        return audio
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    peak = float(np.max(np.abs(audio)))
    if rms < 1e-6 or peak < 1e-6:
        return audio
    gain = min(TARGET_RMS / rms, 0.97 / peak)
    gain = float(np.clip(gain, 1.0, 40.0))
    return (audio * gain).astype(np.float32)


def _is_degenerate(text: str) -> bool:
    """
    Catches Whisper's repetition loops — "and then and then and then ..." —
    which VAD gating doesn't prevent.
    """
    words = text.lower().split()
    if len(words) < 8:
        return False
    for n in (1, 2, 3):
        grams = [tuple(words[i : i + n]) for i in range(0, len(words) - n + 1, n)]
        if grams and len(set(grams)) <= max(1, len(grams) // 4):
            return True
    return False


# --------------------------------------------------------------------------
# Voice activity segmentation
# --------------------------------------------------------------------------


class VadSegmenter:
    """
    Turns a continuous PCM stream into utterances.

    Emits ("partial", audio) while someone is still speaking so the screen
    updates as they talk, and ("final", audio) once they pause.

    max_speech_ms stays under 30 s deliberately: that's the fixed window
    Whisper's encoder accepts, and anything longer would need chunking.
    """

    def __init__(
        self,
        vad,
        threshold: float = VAD_THRESHOLD,
        snr_db: float = SNR_DB,
        min_mean_prob: float = MIN_MEAN_PROB,
        min_speech_ms: int = 250,
        min_silence_ms: int = 700,
        max_speech_ms: int = 20_000,
        pad_ms: int = 300,
        partial_every_ms: int = 1200,
    ):
        self.vad = vad
        self.threshold = threshold
        self.snr_ratio = 10.0 ** (snr_db / 20.0)
        self.min_mean_prob = min_mean_prob
        # Roughly five seconds of frame loudness. The 20th percentile of this
        # is a decent running estimate of the room's noise floor, and unlike a
        # simple average it isn't dragged upward by the speech itself.
        self.recent_rms: deque[float] = deque(maxlen=int(5 * SAMPLE_RATE / VAD_FRAME))
        self.seg_probs: list[float] = []
        self.min_speech = _ms(min_speech_ms)
        self.min_silence = _ms(min_silence_ms)
        self.max_speech = _ms(max_speech_ms)
        self.pad = _ms(pad_ms)
        self.partial_every = _ms(partial_every_ms)

        self.residual = np.zeros(0, dtype=np.float32)
        self.prebuf = np.zeros(0, dtype=np.float32)
        self.speaking = False
        self.segment: list[np.ndarray] = []
        self.speech_run = 0
        self.silence_run = 0
        self.since_partial = 0

    def _end_utterance(self) -> np.ndarray:
        audio = (
            np.concatenate(self.segment)
            if self.segment
            else np.zeros(0, dtype=np.float32)
        )
        self.segment = []
        self.speaking = False
        self.speech_run = 0
        self.silence_run = 0
        self.since_partial = 0
        self.prebuf = np.zeros(0, dtype=np.float32)
        self.seg_probs = []
        self.vad.reset_states()
        return audio

    def push(self, pcm: np.ndarray) -> list[tuple[str, np.ndarray]]:
        self.residual = np.concatenate([self.residual, pcm])
        events: list[tuple[str, np.ndarray]] = []

        while len(self.residual) >= VAD_FRAME:
            frame = self.residual[:VAD_FRAME]
            self.residual = self.residual[VAD_FRAME:]

            with torch.no_grad():
                prob = self.vad(torch.from_numpy(frame), SAMPLE_RATE).item()

            # Track the noise floor and require speech to stand out from it.
            rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
            self.recent_rms.append(rms)
            if len(self.recent_rms) >= 40:
                floor = float(np.percentile(self.recent_rms, 20))
                above_floor = floor < 1e-6 or rms > floor * self.snr_ratio
            else:
                above_floor = True  # not enough history yet

            is_speech = prob >= self.threshold and above_floor
            if is_speech and self.speaking:
                self.seg_probs.append(prob)

            if not self.speaking:
                # Hold a rolling pad so we don't clip the first syllable.
                self.prebuf = np.concatenate([self.prebuf, frame])[-self.pad :]
                if is_speech:
                    self.speech_run += VAD_FRAME
                    if self.speech_run >= self.min_speech:
                        self.speaking = True
                        self.segment = [self.prebuf.copy()]
                        self.prebuf = np.zeros(0, dtype=np.float32)
                        self.silence_run = 0
                        self.since_partial = 0
                else:
                    self.speech_run = 0
                continue

            self.segment.append(frame)
            self.since_partial += VAD_FRAME
            self.silence_run = 0 if is_speech else self.silence_run + VAD_FRAME
            length = sum(len(x) for x in self.segment)

            if self.silence_run >= self.min_silence or length >= self.max_speech:
                mean_prob = (
                    sum(self.seg_probs) / len(self.seg_probs)
                    if self.seg_probs
                    else 0.0
                )
                audio = self._end_utterance()
                if mean_prob >= self.min_mean_prob:
                    events.append(("final", audio))
                else:
                    log.debug(
                        "discarded %.1fs, mean speech probability %.2f",
                        len(audio) / SAMPLE_RATE, mean_prob,
                    )
            elif self.since_partial >= self.partial_every:
                self.since_partial = 0
                events.append(("partial", np.concatenate(self.segment)))

        return events

    def flush(self):
        if self.speaking and self.segment:
            self._end_utterance()


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


class Engine:
    def __init__(self):
        preflight()

        log.info("Loading %s", ASR_MODEL)
        self.asr_proc = WhisperProcessor.from_pretrained(ASR_MODEL)
        try:
            self.asr = WhisperForConditionalGeneration.from_pretrained(
                ASR_MODEL,
                torch_dtype=DTYPE,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                attn_implementation="sdpa",
            )
        except (ValueError, ImportError):
            # Older transformers without SDPA support for Whisper
            self.asr = WhisperForConditionalGeneration.from_pretrained(
                ASR_MODEL, torch_dtype=DTYPE, low_cpu_mem_usage=True
            )
        if ASR_ADAPTER:
            try:
                from peft import PeftModel

                self.asr = PeftModel.from_pretrained(self.asr, ASR_ADAPTER)
                self.asr = self.asr.merge_and_unload()
                log.info("Merged fine-tuned adapter from %s", ASR_ADAPTER)
            except Exception as exc:
                log.warning(
                    "Could not load adapter %s (%s). Using the base model.",
                    ASR_ADAPTER, exc,
                )

        self.asr = self.asr.to(DEVICE).eval()

        log.info("Loading %s", MT_MODEL)
        self.mt_tok = AutoTokenizer.from_pretrained(MT_MODEL)
        self.mt = (
            AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL, torch_dtype=DTYPE)
            .to(DEVICE)
            .eval()
        )

        self.glossary = Glossary(GLOSSARY_PATH)
        self.llm = None
        self.llm_tok = None
        if ENABLE_LLM:
            self._load_llm()

        self._warned_detect = False
        self._build_language_map()
        self._warmup()
        if DEVICE == "cuda":
            used = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            log.info("VRAM: %.1f GB used of %.1f GB", used, total)
            free = total - used
            if free > 4.5 and "1.5B" in LLM_MODEL:
                log.info(
                    "About %.1f GB spare. Qwen/Qwen2.5-3B-Instruct gives "
                    "noticeably better replies and still fits.", free
                )
            if free > 3.0 and MT_MODEL.endswith("600M"):
                log.info(
                    "facebook/nllb-200-distilled-1.3B would improve "
                    "translation quality if you have room for it."
                )
        log.info("Models ready")

    def _load_llm(self):
        """
        Load the reply model. Failure here is not fatal — the server keeps
        working as a translator, which is more useful than refusing to start
        because a 1.5B model wouldn't fit alongside the other two.
        """
        try:
            log.info("Loading %s", LLM_MODEL)
            self.llm_tok = AutoTokenizer.from_pretrained(LLM_MODEL)
            self.llm = (
                AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL,
                    torch_dtype=DTYPE,
                    low_cpu_mem_usage=True,
                )
                .to(DEVICE)
                .eval()
            )
        except Exception as exc:
            self.llm = None
            self.llm_tok = None
            log.warning(
                "Could not load the reply model (%s). Continuing as a "
                "translator only. Set a smaller LLM_MODEL, or ENABLE_LLM=0 "
                "to stop trying.",
                exc,
            )

    @torch.inference_mode()
    def respond(
        self, history: list[dict], text_en: str, persona: str
    ) -> str:
        """Generate a short English reply to an English utterance."""
        if not self.llm or not text_en:
            return ""

        system = PERSONAS.get(persona, PERSONAS["partner"])
        messages = [{"role": "system", "content": system}]
        messages.extend(history[-HISTORY_TURNS * 2 :])
        messages.append({"role": "user", "content": text_en})

        prompt = self.llm_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.llm_tok(prompt, return_tensors="pt").to(DEVICE)
        out = self.llm.generate(
            **enc,
            max_new_tokens=REPLY_MAX_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=self.llm_tok.eos_token_id,
        )
        # Only the newly generated tokens; the prompt is echoed back otherwise.
        reply = self.llm_tok.decode(
            out[0][enc.input_ids.shape[-1] :], skip_special_tokens=True
        )
        return re.sub(r"\s+", " ", reply).strip()

    def _build_language_map(self):
        """
        Map Whisper's language token ids back to ISO codes.

        Reading the language off the decoded string doesn't work: recent
        transformers strips special tokens from generate() output even when
        skip_special_tokens=False, so the `<|fr|>` marker never appears. Token
        ids survive, so match on those instead.
        """
        self._id_to_lang: dict[int, str] = {}
        lang_to_id = getattr(self.asr.generation_config, "lang_to_id", None)
        if lang_to_id:
            for token, tid in lang_to_id.items():
                self._id_to_lang[int(tid)] = token.strip("<|>")
        else:
            # Older generation configs don't carry lang_to_id; ask the
            # tokenizer directly for the languages we can translate.
            for code in NLLB_CODES:
                tid = self.asr_proc.tokenizer.convert_tokens_to_ids(f"<|{code}|>")
                if isinstance(tid, int) and tid > 0:
                    self._id_to_lang[tid] = code
        log.info("Language tokens mapped: %d", len(self._id_to_lang))

    def _warmup(self):
        """
        First generate() call pays for CUDA kernel autotuning. Do it here so
        the first thing someone says isn't twice as slow as everything after.
        """
        log.info("Warming up")
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        self.transcribe(silence, "en", final=False, beams=1)
        self.translate("hello", "en", "es")
        if self.llm:
            self.respond([], "hello", "partner")

    @torch.inference_mode()
    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None,
        final: bool,
        beams: int = 1,
        context: str = "",
        clean: bool = False,
    ):
        # Denoise before normalising, or the gain calculation is dominated by
        # the noise we are about to remove.
        if clean:
            audio = denoise(audio)
        audio = _normalize(audio)

        # Bias Whisper toward the vocabulary we expect. This is the cheapest
        # accuracy lever there is: no training, effective immediately, and it
        # fixes exactly the failures a general model makes on names, jargon
        # and regional usage.
        prompt_ids = None
        hint = self.glossary.prompt(context if USE_ROLLING_CONTEXT else "")
        if hint:
            try:
                prompt_ids = self.asr_proc.get_prompt_ids(
                    hint, return_tensors="pt"
                ).to(DEVICE)
            except Exception as exc:
                log.debug("prompt conditioning unavailable: %s", exc)

        inputs = self.asr_proc(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            return_attention_mask=True,
        )
        features = inputs.input_features.to(DEVICE, dtype=DTYPE)
        mask = inputs.get("attention_mask")
        if mask is not None:
            mask = mask.to(DEVICE)

        out = self.asr.generate(
            features,
            attention_mask=mask,
            language=language,  # None means auto-detect
            task="transcribe",
            num_beams=beams if final else 1,
            prompt_ids=prompt_ids,
            return_timestamps=False,
            return_dict_in_generate=True,
            output_scores=True,
            # max_new_tokens deliberately omitted: Whisper's own generation
            # config caps output at 448, and setting both makes transformers
            # warn on every single call.
        )
        tokens = out.sequences

        detected = language or self._detect_language(tokens, features)

        text = self.asr_proc.batch_decode(tokens, skip_special_tokens=True)[0]
        text = re.sub(r"\s+", " ", text).strip()

        if text.lower() in HALLUCINATIONS or _is_degenerate(text):
            return "", detected

        # Whisper's own confidence. Music and background noise produce
        # fluent-looking text with much worse token probabilities than real
        # speech, so this catches what the voice detector let through.
        confidence = self._avg_logprob(out)
        if confidence is not None and confidence < MIN_AVG_LOGPROB:
            log.debug("low confidence %.2f, discarded: %s", confidence, text[:60])
            return "", detected

        return text, detected

    def _avg_logprob(self, out) -> float | None:
        """Mean log probability of the chosen tokens, or None if unavailable."""
        try:
            scores = self.asr.compute_transition_scores(
                out.sequences,
                out.scores,
                getattr(out, "beam_indices", None),
                normalize_logits=True,
            )
            row = scores[0]
            row = row[torch.isfinite(row)]
            if row.numel() == 0:
                return None
            return float(row.mean())
        except Exception as exc:  # transformers version differences
            log.debug("confidence unavailable: %s", exc)
            return None

    def _detect_language(self, tokens, features) -> str:
        # First choice: the language token in the generated sequence.
        for tid in tokens[0].tolist():
            code = self._id_to_lang.get(int(tid))
            if code:
                return code

        # Some transformers versions strip it. Ask the model directly.
        try:
            lang_ids = self.asr.detect_language(input_features=features)
            code = self._id_to_lang.get(int(lang_ids[0].item()))
            if code:
                return code
        except (AttributeError, RuntimeError, IndexError) as exc:
            log.debug("detect_language unavailable: %s", exc)

        if not self._warned_detect:
            self._warned_detect = True
            log.warning(
                "Automatic language detection is not working with this "
                "transformers version. Set the spoken language explicitly in "
                "the app's settings — translation will not work otherwise."
            )
        return "en"

    @torch.inference_mode()
    def translate(self, text: str, src: str, tgt: str) -> str:
        if not text:
            return ""
        if src == tgt:
            # Nothing to do — Whisper already produced the target language.
            log.debug("no translation needed (%s == %s)", src, tgt)
            return text

        src_code = NLLB_CODES.get(src)
        tgt_code = NLLB_CODES.get(tgt)
        if not src_code:
            log.warning(
                "Whisper detected '%s', which has no NLLB code. Add it to "
                "NLLB_CODES, or set the spoken language explicitly in the app.",
                src,
            )
            return ""
        if not tgt_code:
            log.warning("Target '%s' has no NLLB code.", tgt)
            return ""

        self.mt_tok.src_lang = src_code
        enc = self.mt_tok(
            text, return_tensors="pt", truncation=True, max_length=384
        ).to(DEVICE)
        out = self.mt.generate(
            **enc,
            forced_bos_token_id=self.mt_tok.convert_tokens_to_ids(tgt_code),
            max_length=256,  # max_length, not max_new_tokens, to avoid a warning
            num_beams=1,
        )
        result = self.mt_tok.batch_decode(out, skip_special_tokens=True)[0].strip()
        return self.glossary.apply(result, tgt)


engine: Engine | None = None
# One worker: a single GPU should run one job at a time, and this serialises
# the queue for us without any extra locking.
pool = ThreadPoolExecutor(max_workers=1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load models once at startup. Replaces the deprecated on_event hook."""
    global engine
    engine = Engine()
    yield
    pool.shutdown(wait=False)


app = FastAPI(title="Live Translate", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok" if engine else "loading",
        "auth_required": bool(AUTH_TOKEN),
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
        "asr_model": ASR_MODEL,
        "mt_model": MT_MODEL,
        "llm_model": LLM_MODEL if (engine and engine.llm) else None,
        "personas": sorted(PERSONAS),
        "languages": sorted(NLLB_CODES),
    }


# --------------------------------------------------------------------------
# WebSocket
# --------------------------------------------------------------------------


@app.websocket("/ws")
async def stream(ws: WebSocket):
    await ws.accept()
    authenticated = not AUTH_TOKEN
    loop = asyncio.get_running_loop()

    source: str | None = None  # None means auto-detect
    target = "en"
    gain = 1.0  # set by the app's sensitivity slider
    reply_mode = False
    persona = "partner"
    final_beams = 1
    use_denoise = False
    last_spoken = "en"  # language of the most recent utterance
    last_text = ""      # previous utterance, fed back as decoding context
    composed_id = 0     # composed messages get negative ids to avoid clashes
    history: list[dict] = []  # English turns, for LLM context
    vad_model = load_vad()
    seg = VadSegmenter(vad_model)
    busy = False
    utterance_id = 0

    async def _reply(uid: int, text: str, detected: str):
        """English in, English out, then back into the spoken language."""
        r0 = time.time()

        # The LLM thinks in English, so give it English regardless of what
        # the user chose as their display target.
        if detected == "en":
            text_en = text
        else:
            text_en = await loop.run_in_executor(
                pool, engine.translate, text, detected, "en"
            )
        if not text_en:
            return

        reply_en = await loop.run_in_executor(
            pool, engine.respond, history, text_en, persona
        )
        if not reply_en:
            return

        history.append({"role": "user", "content": text_en})
        history.append({"role": "assistant", "content": reply_en})
        del history[: max(0, len(history) - HISTORY_TURNS * 2)]

        reply_spoken = (
            reply_en
            if detected == "en"
            else await loop.run_in_executor(
                pool, engine.translate, reply_en, "en", detected
            )
        )

        log.info("   reply [en] %s", reply_en)
        if detected != "en":
            log.info("   reply [%s] %s", detected, reply_spoken)

        await ws.send_text(
            json.dumps(
                {
                    "type": "reply",
                    "id": uid,
                    "reply_english": reply_en,
                    "reply_spoken": reply_spoken,
                    "spoken_language": detected,
                    "latency_ms": int((time.time() - r0) * 1000),
                }
            )
        )

    async def _compose(uid: int, typed: str, into: str):
        """
        Translate something the user typed in English into whatever language
        was last spoken, so they can reply in the conversation.
        """
        try:
            spoken = (
                typed
                if into == "en"
                else await loop.run_in_executor(
                    pool, engine.translate, typed, "en", into
                )
            )
            log.info("   typed [en] %s", typed)
            if into != "en":
                log.info("   typed [%s] %s", into, spoken)
            await ws.send_text(
                json.dumps(
                    {
                        "type": "composed",
                        "id": uid,
                        "source_language": "en",
                        "target_language": into,
                        "transcript": typed,
                        "translation": spoken,
                    }
                )
            )
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("compose failed")

    async def run_job(kind: str, audio: np.ndarray, uid: int):
        nonlocal busy, last_spoken, last_text
        busy = True
        try:
            t0 = time.time()
            text, detected = await loop.run_in_executor(
                pool, engine.transcribe, audio, source,
                kind == "final", final_beams, last_text, use_denoise,
            )
            if not text:
                return
            translation = await loop.run_in_executor(
                pool, engine.translate, text, detected, target
            )
            if kind == "final":
                log.info(
                    "level: peak %.1f dBFS over %.1fs",
                    _dbfs(audio), len(audio) / SAMPLE_RATE,
                )
                log.info("[%s] %s", detected, text)
                log.info("   -> [%s] %s", target, translation or "(none)")
            await ws.send_text(
                json.dumps(
                    {
                        "type": kind,
                        "id": uid,
                        "source_language": detected,
                        "target_language": target,
                        "transcript": text,
                        "translation": translation,
                        "seconds": round(len(audio) / SAMPLE_RATE, 2),
                        "latency_ms": int((time.time() - t0) * 1000),
                    }
                )
            )

            # Replies only ever follow a finished utterance. Generating one
            # for every partial would mean answering half-sentences.
            if kind == "final":
                last_spoken = detected
                last_text = text
                if COLLECT_DATA:
                    save_example(
                        audio,
                        {
                            "transcript": text,
                            "translation": translation,
                            "language": detected,
                            "target": target,
                            "seconds": round(len(audio) / SAMPLE_RATE, 2),
                            "peak_dbfs": round(_dbfs(audio), 1),
                            "environment": env_name,
                        },
                    )
                if reply_mode and engine.llm:
                    await _reply(uid, text, detected)
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("inference failed")
        finally:
            busy = False

    try:
        while True:
            msg = await ws.receive()

            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                cfg = json.loads(msg["text"])
                if cfg.get("type") == "compose":
                    if not authenticated:
                        continue
                    typed = (cfg.get("text") or "").strip()
                    if typed:
                        composed_id += 1
                        asyncio.create_task(
                            _compose(
                                -composed_id,
                                typed,
                                cfg.get("language") or last_spoken,
                            )
                        )
                    continue

                if cfg.get("type") == "config":
                    if AUTH_TOKEN and cfg.get("token") != AUTH_TOKEN:
                        log.warning(
                            "Rejected a session from %s: bad token",
                            ws.client.host if ws.client else "unknown",
                        )
                        await ws.send_text(
                            json.dumps(
                                {
                                    "type": "error",
                                    "reason": "auth",
                                    "message": "Wrong or missing access token.",
                                }
                            )
                        )
                        await ws.close(code=4401)
                        return
                    authenticated = True

                    src = cfg.get("source")
                    source = None if src in (None, "auto", "") else src
                    target = cfg.get("target", "en")
                    gain = float(cfg.get("gain", 1.0))
                    reply_mode = bool(cfg.get("reply", False))
                    persona = cfg.get("persona", "partner")
                    history = []
                    if reply_mode and not engine.llm:
                        log.warning(
                            "Replies requested but no reply model is loaded."
                        )

                    env_name = cfg.get("environment", "normal")
                    env = ENVIRONMENTS.get(env_name, ENVIRONMENTS["normal"])
                    speed_name = cfg.get("responsiveness", "balanced")
                    speed = RESPONSIVENESS.get(
                        speed_name, RESPONSIVENESS["balanced"]
                    )
                    final_beams = speed["final_beams"]
                    use_denoise = (
                        os.environ.get("DENOISE", "").lower() == "on"
                        or env.get("denoise", False)
                    )
                    seg = VadSegmenter(
                        vad_model,
                        threshold=env["threshold"],
                        snr_db=env["snr_db"],
                        min_mean_prob=env["min_mean_prob"],
                        min_silence_ms=speed["min_silence_ms"],
                    )
                    log.info(
                        "session: %s -> %s  (gain %.1fx, environment '%s', "
                        "responsiveness '%s', wait %dms, beams %d, "
                        "denoise %s)",
                        source or "auto-detect", target, gain, env_name,
                        speed_name, speed["min_silence_ms"], final_beams,
                        "on" if use_denoise else "off",
                    )
                    if source and source == target:
                        log.warning(
                            "Spoken language and target are both '%s'. Nothing "
                            "will be translated. Change one in the app's "
                            "settings.",
                            target,
                        )
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "ready",
                                "target": target,
                                "reply_available": engine.llm is not None,
                            }
                        )
                    )
                continue

            data = msg.get("bytes")
            if not data:
                continue
            if not authenticated:
                continue  # audio before a valid config is silently dropped

            pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if gain != 1.0:
                pcm = np.clip(pcm * gain, -1.0, 1.0)
            for kind, audio in seg.push(pcm):
                if kind == "partial" and busy:
                    continue  # drop stale partials rather than build a backlog
                if kind == "final":
                    utterance_id += 1
                    uid = utterance_id
                else:
                    # Partials belong to the utterance still in progress, so
                    # they carry the id the next final will claim.
                    uid = utterance_id + 1
                asyncio.create_task(run_job(kind, audio, uid))

    except WebSocketDisconnect:
        pass
    finally:
        seg.flush()
        log.info("session closed")


# --------------------------------------------------------------------------
# Launcher
# --------------------------------------------------------------------------

PORT = 8000


def local_ip() -> str:
    """Best guess at the LAN address the phone should connect to."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # no packets sent, this just picks the route
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def serve():
    import uvicorn

    ip = local_ip()
    print()
    print("  Loading models. First run downloads about 3.5 GB.")
    print("  When it prints 'Models ready', point the app at:")
    print(f"      ws://{ip}:{PORT}/ws")
    print(f"  Check it in a browser first:  http://{ip}:{PORT}/health")
    print()
    sys.stdout.flush()

    uvicorn.run(
        app,
        host="0.0.0.0",  # 127.0.0.1 would refuse the phone
        port=PORT,
        log_level="info",
    )


if __name__ == "__main__":
    serve()
