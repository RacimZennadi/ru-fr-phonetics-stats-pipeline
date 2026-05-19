from __future__ import annotations

import collections
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
import yaml
from transformers import Wav2Vec2Model, Wav2Vec2Processor


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_word_table(csv_path: Path, min_word_duration_s: float) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        sep=";",
        header=None,
        names=["word", "start", "end"],
    )
    df = df[df["word"].notna()]
    df["word"] = df["word"].astype(str).str.strip().str.lower()
    df = df[df["word"] != ""]
    df = df[(df["end"] - df["start"]) >= min_word_duration_s]
    return df


def main() -> None:
    params = load_params()
    corpus_root = Path(params["corpus_root"])
    model_name = params.get("model_name", "facebook/wav2vec2-base")
    num_speakers = params.get("num_speakers")
    target_sr = int(params.get("target_sr", 16000))
    min_word_duration_s = float(params.get("min_word_duration_s", 0.05))

    if not corpus_root.exists():
        raise FileNotFoundError(
            f"Corpus root not found: {corpus_root}. "
            "Place the Russian-French interference corpus at the path configured in params.yaml."
        )

    print("Loading Wav2Vec2 processor and model...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()

    representations: list[np.ndarray] = []
    labels: list[dict[str, str]] = []

    speakers = sorted(path.name for path in corpus_root.iterdir() if path.is_dir())
    if num_speakers is not None:
        speakers = speakers[: int(num_speakers)]
    print(f"Found {len(speakers)} speaker directories to process in {corpus_root}")

    start_time = time.time()
    processed_files = 0

    for speaker in speakers:
        speaker_dir = corpus_root / speaker
        csv_files = sorted(speaker_dir.glob("*_words.csv"))
        if csv_files:
            print(f"\nProcessing speaker: {speaker} ({len(csv_files)} files)")

        for csv_path in csv_files:
            wav_path = csv_path.with_name(csv_path.name.replace("_words.csv", ".wav"))
            if not wav_path.exists():
                continue

            print(f"  -> Extracting words from: {csv_path.name}")
            try:
                waveform, sr = torchaudio.load(str(wav_path))
                if waveform.ndim > 1:
                    waveform = waveform.mean(dim=0)
                if sr != target_sr:
                    waveform = torchaudio.functional.resample(waveform, sr, target_sr)

                df = load_word_table(csv_path, min_word_duration_s)

                for _, row in df.iterrows():
                    word = row["word"]
                    start_sample = int(row["start"] * target_sr)
                    end_sample = int(row["end"] * target_sr)
                    segment = waveform[start_sample:end_sample]
                    if segment.numel() < 400:
                        continue

                    inputs = processor(
                        segment.numpy(),
                        sampling_rate=target_sr,
                        return_tensors="pt",
                        padding=True,
                    )
                    with torch.no_grad():
                        output = model(**inputs)
                    embedding = (
                        output.last_hidden_state.squeeze(0).mean(dim=0).cpu().numpy().astype(np.float64)
                    )
                    representations.append(embedding)
                    labels.append({"speaker": speaker, "word": word})

                processed_files += 1
                elapsed = time.time() - start_time
                avg_time_per_file = elapsed / processed_files
                print(
                    f"     [Processed {processed_files} files total | "
                    f"Elapsed: {elapsed:.1f}s | Avg: {avg_time_per_file:.2f}s/file]"
                )
            except Exception as exc:
                print(f"     [Error processing {csv_path.name}: {exc}]")

    data_64 = np.array(representations, dtype=np.float64)
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / "rep_float64.npy", data_64)
    with open(output_dir / "labels.json", "w", encoding="utf-8") as handle:
        json.dump(labels, handle)

    speakers_set = {item["speaker"] for item in labels}
    words_set = {item["word"] for item in labels}
    word_counts = collections.Counter(item["word"] for item in labels)

    print("\n--- EXTRACTION COMPLETE ---")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print(f"Total vectors: {len(labels)}")
    print(f"Speakers ({len(speakers_set)}): {sorted(speakers_set)}")
    print(f"Unique words ({len(words_set)}): {sorted(words_set)[:20]}...")
    print(f"Top 10 words: {word_counts.most_common(10)}")


if __name__ == "__main__":
    main()
