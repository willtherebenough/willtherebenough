"""
Fine-tune Whisper on utterances this server collected and you corrected.

This is the slow path. Before reaching for it, exhaust the cheap options:
edit glossary.json, and set the spoken language explicitly in the app. Those
cost nothing and fix most of what a general model gets wrong on names and
regional vocabulary. Fine-tuning only starts to win once you have a few
hundred corrected utterances of one speaker or one dialect.

Workflow
--------
1. Run the server with COLLECT_DATA=1. Each finished utterance writes a .wav
   and a .json into the dataset folder.
2. Correct the transcripts. Open each .json, fix the "transcript" field where
   it's wrong, and set "corrected": true. Only corrected files are used.
3. Run this script. It trains a LoRA adapter, which is a small set of extra
   weights rather than a whole new model.
4. Point the server at it:  ASR_ADAPTER=whisper-lora  python main.py

Requirements
------------
    pip install -r requirements-train.txt
"""

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

SAMPLE_RATE = 16_000


class CorrectedUtterances(Dataset):
    """Only reads examples whose transcript a human has checked."""

    def __init__(self, folder: str, processor: WhisperProcessor, language: str):
        self.processor = processor
        self.language = language
        self.items = []

        for name in sorted(os.listdir(folder)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(folder, name), encoding="utf-8") as fh:
                meta = json.load(fh)
            if not meta.get("corrected"):
                continue
            if language != "auto" and meta.get("language") != language:
                continue
            wav = os.path.join(folder, meta.get("audio", ""))
            if meta.get("transcript") and os.path.exists(wav):
                self.items.append((wav, meta["transcript"]))

        if not self.items:
            raise SystemExit(
                f"No corrected examples in {folder}.\n"
                'Set "corrected": true in the .json files you have fixed.'
            )
        print(f"{len(self.items)} corrected examples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import wave

        path, text = self.items[i]
        with wave.open(path, "rb") as fh:
            raw = fh.readframes(fh.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        features = self.processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features[0]

        self.processor.tokenizer.set_prefix_tokens(
            language=self.language if self.language != "auto" else "english",
            task="transcribe",
        )
        labels = self.processor.tokenizer(text).input_ids
        return {"input_features": features, "labels": labels}


@dataclass
class Collator:
    processor: WhisperProcessor

    def __call__(self, batch):
        features = torch.stack([b["input_features"] for b in batch])
        labels = self.processor.tokenizer.pad(
            [{"input_ids": b["labels"]} for b in batch], return_tensors="pt"
        )
        ids = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        # The decoder adds the BOS token itself; leaving ours in doubles it.
        if (ids[:, 0] == self.processor.tokenizer.bos_token_id).all():
            ids = ids[:, 1:]
        return {"input_features": features, "labels": ids}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--out", default="whisper-lora")
    ap.add_argument("--language", default="es",
                    help="Train on one language. 'auto' uses everything.")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    processor = WhisperProcessor.from_pretrained(
        args.model,
        language=args.language if args.language != "auto" else None,
        task="transcribe",
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float32
    )
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # LoRA trains a small number of extra parameters instead of all of them,
    # which is what makes this fit on a consumer card and makes it hard to
    # wreck the model's general ability with a few hundred examples.
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    dataset = CorrectedUtterances(args.data, processor, args.language)

    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=args.out,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=max(1, 8 // args.batch),
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            remove_unused_columns=False,
            label_names=["labels"],
            report_to=[],
        ),
        train_dataset=dataset,
        data_collator=Collator(processor),
    )

    trainer.train()
    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)

    print(f"\nAdapter written to {args.out}")
    print("Use it with:  set ASR_ADAPTER=" + args.out + "  before starting main.py")


if __name__ == "__main__":
    main()
