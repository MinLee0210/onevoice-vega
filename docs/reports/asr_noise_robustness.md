# ASR Noise Robustness Benchmarking Report

This report evaluates the noise robustness of **huuquyet/PhoWhisper-base** on the Vietnamese VIVOS test dataset under varying signal-to-noise ratio (SNR) conditions.

## 1. Test Setup
- **ASR Model:** `huuquyet/PhoWhisper-base` (74M parameters)
- **Dataset:** VIVOS Test Set (`quocanh34/viet_vivos` Parquet mirror)
- **Sample Size:** 30 utterances
- **Noise Type:** Additive White Gaussian Noise (AWGN) to simulate high-frequency factory floor acoustics

## 2. Evaluation Results

| SNR Level | Word Error Rate (WER) | Avg Latency / Utterance | Target (WER < 12% at 5dB) |
| :--- | :---: | :---: | :---: |
| Clean | 4.93% | 278.8 ms | N/A |
| 15dB SNR | 7.04% | 333.0 ms | N/A |
| 10dB SNR | 10.21% | 275.7 ms | N/A |
| 5dB SNR | 12.32% | 279.8 ms | FAIL |

## 3. Analysis & Key Takeaways
- **Target Met:** No. The model achieved **12.32%** WER at 5dB SNR, which exceeds the challenge threshold of **< 12.00%**.
- **Noise Impact:** Notice how spelling and phonetic prediction degrade as SNR decreases. Attention-based models like Whisper have cross-attention mechanisms that can attend around noise bands, showing high robustness compared to raw CTC models.
