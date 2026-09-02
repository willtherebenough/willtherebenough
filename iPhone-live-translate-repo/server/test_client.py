"""
Smoke test: streams your PC's microphone to the server and prints results.

Run this before you touch the phone. It uses the same protocol the Android app
does, so if this works the server is correct and any remaining problem is on the
phone or the network — which is a much smaller haystack.

    pip install sounddevice websockets
    python test_client.py --target es

Ctrl+C to stop.
"""

import argparse
import asyncio
import json
import queue
import sys

# Windows consoles default to a legacy code page (cp1252), and printing a
# Spanish accent or a Japanese character raises UnicodeEncodeError mid-session.
# Reconfigure before anything gets written.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import sounddevice as sd  # noqa: E402
import websockets  # noqa: E402

SAMPLE_RATE = 16_000
BLOCK = 1600  # 100 ms


async def main(url: str, source: str, target: str):
    audio_q: queue.Queue = queue.Queue()

    def on_audio(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        audio_q.put(bytes(indata))

    print(f"Connecting to {url} ...")
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps(
            {"type": "config", "source": source, "target": target}
        ))

        stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK,
            dtype="int16",
            channels=1,
            callback=on_audio,
        )

        async def send_audio():
            loop = asyncio.get_running_loop()
            with stream:
                while True:
                    chunk = await loop.run_in_executor(None, audio_q.get)
                    await ws.send(chunk)

        async def receive():
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("type") == "ready":
                    print("Server ready. Start talking.\n")
                    continue
                tag = "  ..." if msg["type"] == "partial" else ">>>"
                print(f"{tag} [{msg['source_language']}] {msg['transcript']}")
                if msg.get("translation"):
                    print(f"    {msg['translation']}"
                          f"    ({msg['latency_ms']} ms)")

        await asyncio.gather(send_audio(), receive())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    p.add_argument("--source", default="auto")
    p.add_argument("--target", default="es")
    args = p.parse_args()
    try:
        asyncio.run(main(args.url, args.source, args.target))
    except KeyboardInterrupt:
        print("\nStopped.")
