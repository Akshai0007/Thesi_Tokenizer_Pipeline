"""Compatibility checks for an exported tokenizer directory.

Run before using the tokenizer with any training stack:
    python -m mltok verify --tokenizer runs/<name>/best
"""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

from . import registry

CASES = [
    "കേരളം ഇന്ത്യയുടെ തെക്കുപടിഞ്ഞാറൻ തീരത്തുള്ള ഒരു സംസ്ഥാനമാണ്.",
    "ചോദ്യം: നിങ്ങൾ എവിടെയാണ് താമസിക്കുന്നത്?\nഉത്തരം: ഞാൻ കൊച്ചിയിലാണ്.",
    "Hello world, this is English text 12345!",
    "മിക്സഡ് text with English and 2024 numbers 3.5%",
    "  leading and  double  spaces ",
    "emoji 😀 and other scripts: हिन्दी தமிழ் 日本語",
    "old chillu: \u0d28\u0d4d\u200d \u0d30\u0d4d\u200d \u0d15\u0d4d\u200d",
    "zwnj: സദ്\u200cവാരം",
    "\ttabs\tand\nnewlines\n\n",
    "",
]


def verify_dir(path: str, normalizer="ml", log=print) -> bool:
    registry.load_plugins()
    norm = registry.make("normalizer", normalizer)
    d = Path(path)
    ok = True

    def check(name: str, cond: bool, detail: str = ""):
        nonlocal ok
        ok &= bool(cond)
        log(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    if not (d / "tokenizer.json").exists():
        hint = ("this directory only has spm.model: run  python -m mltok export-hf --spm %s/spm.model --normalizer <name>"
                % d) if (d / "spm.model").exists() else "no tokenizer.json found"
        log(f"  [FAIL] {hint}")
        return False
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    log(f"tokenizer.json: vocab={tok.get_vocab_size()}")
    for t in ("<pad>", "<s>", "</s>", "<unk>"):
        check(f"special token {t} present", tok.token_to_id(t) is not None)
    mult = tok.get_vocab_size() % 64 == 0          # efficiency recommendation, not a correctness requirement
    log(f"  [{'PASS' if mult else 'WARN'}] vocab size multiple of 64 (GPU efficiency) {'' if mult else '- consider padding/choosing a multiple of 64'}")

    bad = []
    for c in CASES:
        got = tok.decode(tok.encode(c, add_special_tokens=False).ids, skip_special_tokens=False)
        if got != norm.normalize(c):
            bad.append((c[:30], got[:30]))
    check("lossless round-trip (vs normalized text)", not bad, str(bad[:2]))

    ids = tok.encode("😀", add_special_tokens=False).ids
    check("emoji handled by byte fallback (no <unk>)", tok.token_to_id("<unk>") not in ids)

    try:
        from transformers import AutoTokenizer

        hf = AutoTokenizer.from_pretrained(str(d))
        enc = hf("ഇത് ഒരു പരീക്ഷണം ആണ്")["input_ids"]
        check("transformers AutoTokenizer loads", True)
        check("transformers == tokenizers ids", enc == tok.encode("ഇത് ഒരു പരീക്ഷണം ആണ്", add_special_tokens=False).ids)
        msgs = [{"role": "user", "content": "ഹലോ"}]
        txt = hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        check("chat template renders", "<|im_start|>assistant" in txt)
    except ImportError:
        log("  [SKIP] transformers not installed - skipped AutoTokenizer/chat-template checks")
    except Exception as e:  # noqa: BLE001
        check("transformers AutoTokenizer loads", False, f"{type(e).__name__}: {e}")
    log("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return ok
