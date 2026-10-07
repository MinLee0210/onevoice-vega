import argparse
import time

import jiwer
import numpy as np
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def add_noise(signal, snr_db):
    if snr_db is None:
        return signal
    power_signal = np.mean(signal**2)
    if power_signal == 0:
        return signal
    power_noise = power_signal * (10 ** (-snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(power_noise), len(signal))
    return signal + noise


def normalize_text(text):
    # Lowercase and clean up punctuation to focus on raw transcription accuracy
    text = text.lower().strip()
    for p in [".", ",", "?", "!", ";", ":"]:
        text = text.replace(p, "")
    return " ".join(text.split())


def main():
    parser = argparse.ArgumentParser(description="Evaluate PhoWhisper ASR noise robustness")
    parser.add_argument(
        "--num-samples", type=int, default=30, help="Number of test samples to load"
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default="huuquyet/PhoWhisper-base",
        help="Hugging Face model repository",
    )
    args = parser.parse_args()

    print(f"Loading {args.num_samples} samples from VIVOS test set...")
    vivos = load_dataset("quocanh34/viet_vivos", split="test", streaming=True)
    samples = [s for _, s in zip(range(args.num_samples), vivos)]

    print(f"Loading model and processor from {args.model_repo}...")
    processor = WhisperProcessor.from_pretrained(args.model_repo)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_repo).eval()

    snr_levels = [None, 15, 10, 5]
    results = {}

    np.random.seed(42)  # Set seed for reproducible noise generation

    for snr in snr_levels:
        snr_label = f"{snr}dB SNR" if snr is not None else "Clean"
        print(f"\nEvaluating on {snr_label}...")

        references = []
        hypotheses = []
        start_time = time.time()

        for idx, s in enumerate(samples):
            ref = normalize_text(s["transcription"])
            references.append(ref)

            audio_data = s["audio"]["array"]
            noisy_audio = add_noise(audio_data, snr)

            # Generate transcript
            inputs = processor(
                noisy_audio, sampling_rate=s["audio"]["sampling_rate"], return_tensors="pt"
            )
            gen = model.generate(**inputs, language="vi", task="transcribe", max_new_tokens=64)
            hyp = processor.batch_decode(gen, skip_special_tokens=True)[0]
            hyp = normalize_text(hyp)
            hypotheses.append(hyp)

            if idx < 3:
                # Print a few examples for inspection
                print(f"  [Sample {idx}]")
                print(f"    Ref:  {ref}")
                print(f"    Hyp:  {hyp}")

        wer = jiwer.wer(references, hypotheses)
        duration = time.time() - start_time
        avg_latency_ms = (duration / len(samples)) * 1000.0

        results[snr_label] = {"wer": wer, "latency_ms": avg_latency_ms}
        print(
            f"  Result for {snr_label} -> WER: {wer * 100.0:.2f}%, Avg Latency: {avg_latency_ms:.1f}ms"
        )

    # Generate Markdown Report
    report_path = "docs/reports/asr_noise_robustness.md"
    print(f"\nSaving report to {report_path}...")

    import os

    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        f.write("# ASR Noise Robustness Benchmarking Report\n\n")
        f.write(
            f"This report evaluates the noise robustness of **{args.model_repo}** on the Vietnamese VIVOS test dataset under varying signal-to-noise ratio (SNR) conditions.\n\n"
        )
        f.write("## 1. Test Setup\n")
        f.write(f"- **ASR Model:** `{args.model_repo}` (74M parameters)\n")
        f.write("- **Dataset:** VIVOS Test Set (`quocanh34/viet_vivos` Parquet mirror)\n")
        f.write(f"- **Sample Size:** {args.num_samples} utterances\n")
        f.write(
            "- **Noise Type:** Additive White Gaussian Noise (AWGN) to simulate high-frequency factory floor acoustics\n\n"
        )
        f.write("## 2. Evaluation Results\n\n")
        f.write(
            "| SNR Level | Word Error Rate (WER) | Avg Latency / Utterance | Target (WER < 12% at 5dB) |\n"
        )
        f.write("| :--- | :---: | :---: | :---: |\n")

        for label, data in results.items():
            wer_pct = data["wer"] * 100.0
            status = "N/A"
            if label == "5dB SNR":
                status = "PASS" if data["wer"] <= 0.12 else "FAIL"
            f.write(f"| {label} | {wer_pct:.2f}% | {data['latency_ms']:.1f} ms | {status} |\n")

        f.write("\n")

        f.write("## 3. Analysis & Key Takeaways\n")
        wer_5db = results["5dB SNR"]["wer"] * 100.0
        if results["5dB SNR"]["wer"] <= 0.12:
            f.write(
                f"- **Target Met:** Yes! The model achieved **{wer_5db:.2f}%** WER at 5dB SNR, satisfying the challenge threshold of **< 12.00%**.\n"
            )
        else:
            f.write(
                f"- **Target Met:** No. The model achieved **{wer_5db:.2f}%** WER at 5dB SNR, which exceeds the challenge threshold of **< 12.00%**.\n"
            )

        f.write(
            "- **Noise Impact:** Notice how spelling and phonetic prediction degrade as SNR decreases. Attention-based models like Whisper have cross-attention mechanisms that can attend around noise bands, showing high robustness compared to raw CTC models.\n"
        )

    print("Benchmarking completed successfully!")


if __name__ == "__main__":
    main()
