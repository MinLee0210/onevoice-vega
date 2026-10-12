# vi-ctc

- Source checkpoint: dragonSwing/wav2vec2-base-vietnamese
- License: Apache-2.0 (verify the card)
- Exported: 2026-10-12
- Architecture: Wav2Vec2ForCTC — single forward pass, CTC greedy decode
- ONNX: `model.onnx` (input_values [1,N] f32 -> logits [1,T,V])

Files: `model.onnx`, `vocab.json`, `preprocessor_config.json`, `config.json`
