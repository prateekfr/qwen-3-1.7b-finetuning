"""
Build English dictionary from Qwen tokenizer vocabulary and training prompt datasets.
Produces a clean, newline-delimited words file for rapid O(1) set membership testing.
"""
import json
import re
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent

def main():
    words = set()

    # 1. Extract from all JSONL files in data/
    for path in (ROOT / "data").glob("**/*.jsonl"):
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                for k in ("prompt", "response", "original_response"):
                    if k in row and row[k]:
                        for w in re.findall(r"[a-zA-Z]+", row[k].lower()):
                            if len(w) >= 2:
                                words.add(w)
        except Exception:
            continue

    # 2. Extract from Qwen Tokenizer vocabulary
    weights_dir = ROOT / "weights" / "qwen3-1.7b"
    if weights_dir.exists():
        tok = AutoTokenizer.from_pretrained(str(weights_dir), trust_remote_code=True)
        for token, tid in tok.get_vocab().items():
            cleaned = tok.decode([tid]).strip().lower()
            if cleaned.isalpha() and 2 <= len(cleaned) <= 15:
                words.add(cleaned)

    # 3. Add common single letter words 'a', 'i'
    words.add("a")
    words.add("i")

    out_file = ROOT / "data" / "dictionary" / "english_words.txt"
    sorted_words = sorted(words)
    out_file.write_text("\n".join(sorted_words), encoding="utf-8")
    print(f"✅ Saved {len(sorted_words)} words to {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
