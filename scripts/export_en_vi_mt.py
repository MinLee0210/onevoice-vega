import json
import shutil
from pathlib import Path
from datetime import date
from optimum.exporters.onnx import main_export
import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as spm_pb2
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import Unigram
from tokenizers.normalizers import Precompiled, Replace, Sequence

MODEL_ID = "Helsinki-NLP/opus-mt-en-vi"
EXPORT_DIR = Path("models/mt/en-vi/_export")
FINAL_DIR = Path("models/mt/en-vi")

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

print("Exporting model to ONNX...")
main_export(
    model_name_or_path=MODEL_ID,
    output=str(EXPORT_DIR),
    task="text2text-generation-with-past",
)

print("Converting tokenizer...")
with open(EXPORT_DIR / "vocab.json") as f:
    vocab_json = json.load(f)

vocab_size = max(vocab_json.values()) + 1
id_to_piece = [None] * vocab_size
for piece, idx in vocab_json.items():
    id_to_piece[idx] = piece
assert all(p is not None for p in id_to_piece), "gap in vocab id space"

sp = spm.SentencePieceProcessor(model_file=str(EXPORT_DIR / "source.spm"))
piece_scores = {sp.id_to_piece(i): sp.get_score(i) for i in range(sp.get_piece_size())}
min_score = min(piece_scores.values())

vocab_list = [(piece, piece_scores.get(piece, min_score - 1.0)) for piece in id_to_piece]
unk_id = vocab_json["<unk>"]

fast_tokenizer = Tokenizer(Unigram(vocab_list, unk_id=unk_id, byte_fallback=False))

proto = spm_pb2.ModelProto()
proto.ParseFromString((EXPORT_DIR / "source.spm").read_bytes())
charsmap = proto.normalizer_spec.precompiled_charsmap
fast_tokenizer.normalizer = Sequence([Precompiled(charsmap), Replace(" {2,}", " ")])
fast_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.Metaspace(replacement="\u2581", prepend_scheme="always")]
)
fast_tokenizer.decoder = decoders.Metaspace(replacement="\u2581", prepend_scheme="always")

(EXPORT_DIR / "tokenizer.json").write_text("")
fast_tokenizer.save(str(EXPORT_DIR / "tokenizer.json"))
print(f"vocab_size={vocab_size} unk_id={unk_id}")

print("Packaging artifacts...")
keep = ["encoder_model.onnx", "decoder_model_merged.onnx", "tokenizer.json"]
for name in keep:
    shutil.copyfile(EXPORT_DIR / name, FINAL_DIR / name)

manifest = f"""# mt/en-vi

- Source checkpoint: {MODEL_ID}
- License: Apache 2.0 (Opus-MT / Helsinki-NLP — safe to commercialize,
  unlike NLLB's CC-BY-NC; see CLAUDE.md license trap note)
- Exported: {date.today().isoformat()}
- Precision: fp32 (baseline export, not yet quantized — INT8 via AI Hub
  Workbench is the documented upgrade path in CLAUDE.md's model stack
  table; revisit before flashing to device, current size is ~500MB,
  well over the ~300MB/pair memory budget target)
- Decode: greedy (beam=1), matches CLAUDE.md's latency-driven decode
  strategy

Files:
- `encoder_model.onnx`
- `decoder_model_merged.onnx` (covers both the no-cache first step and
  the with-cache subsequent steps via `use_cache_branch`)
- `tokenizer.json` (Rust `tokenizers`-crate-compatible; derived from slow MarianTokenizer)
"""
(FINAL_DIR / "MANIFEST.md").write_text(manifest)

shutil.rmtree(EXPORT_DIR)
print("Done! Exported and packaged models/mt/en-vi successfully.")
