# RESULTS_LOG — what the real runs showed

Keep this file honest: record what was measured, what it does *not* show, and which bugs the data exposed.

## Run 1 (Kaggle 2×T4, commit a806e54, `kaggle_small.yaml`)
**Setup:** FineWeb-2 `mal_Mlym` + Wikipedia `ml` + Sangraha `verified/mal`; 700 MB training cap; proxy LM = GPT *tiny*
(11.1M non-embedding params), 120 MB text ≈ 8.6M tokens ≈ 524 steps, equal-bytes budget, bits-per-byte (BPB) scoring, 3 seeds for the final comparison.

**Final comparison (32 000 vocab, mean ± std over 3 seeds)**

| tokenizer | BPB | fertility | invalid-start tokens | akshara breaks | lossless on stress set |
|---|---|---|---|---|---|
| **SentencePiece Unigram** | **0.7463 ± 0.0015** | **1.946** | 15.5 % | 41.8 % | yes |
| HF BPE (stock) | 0.7648 ± 0.0010 | 2.002 | 8.5 % | 24.7 % | **no** (0.82; no byte fallback) |
| our constrained BPE (`mlbpe`) | 0.7686 ± 0.0033 | 2.010 | **0.4 %** | **7.4 %** | yes |

**What this shows**
* SentencePiece Unigram is best: −0.0223 BPB (≈ 3 %) vs `mlbpe` (≈ 6.7× the largest seed spread) and −0.0185 vs HF BPE.
* `mlbpe` vs HF BPE is a **tie** (gap 0.0038 vs spread 0.0033) — but `mlbpe` has ~20× fewer invalid-start tokens and is lossless, while HF BPE is not.
* The script-integrity diagnostics (`mark_start_rate`, `akshara_break_rate`) did **not** predict BPB: the tokenizer with the *worst* values won, the one with the *best* values came last. Treat them as diagnostics, not objectives.
* Vocabulary (single seed, SentencePiece): 16K → 32K → 48K gives BPB 0.7672 → 0.7428 → 0.7343 and fertility ≈ 2.17 → 1.94 → 1.83. Still improving at 48K, with diminishing returns (confound: embedding parameters grow 17.4M → 23.5M → 29.7M).
* Normalization (`none`/`nfc`/`ml`) and ZWNJ keep/drop changed fertility by < 0.001 on this data: the custom normalizer is insurance here, not a gain.
* Fertility vs published tokenizers on the same eval text (muril 2.36, xlm-r 2.74, sarvam-1 3.34) is **not like-for-like**: ours is Malayalam-only, 32K, trained on in-distribution data.

**What it does NOT show**
* Anything about larger models/datasets: the proxy is tiny and sees ~8.6M tokens. The ranking must be re-checked on the DGX with a bigger proxy before the final decision.
* That the data mix is right — see bug 1.

**Bugs the real run exposed (all fixed in the repo after run 1)**
1. **Data mix:** the 700 MB cap was filled by the first source alone — training data was 100 % FineWeb-2 (Wikipedia and Sangraha only reached *eval*; Wikipedia fertility 2.37 vs 1.88 for web). Fix: sources are interleaved by weight (`split.balance`, `split.weights`), per-source shares are logged (`train_share`), and the freeze checklist refuses a mix with one source > 90 %. Regression test: `test_capped_training_set_mixes_all_sources_not_just_the_first`.
2. **SentencePiece winners could not be used or verified:** `best/` held only `spm.model`, `verify` could not check it, and a notebook patch skipped the check, so the gate "passed" without verifying anything. Fix: SentencePiece Unigram is exported to Hugging Face `tokenizer.json` automatically, with a parity check against SentencePiece (identical ids or an exact Viterbi score tie; anything else rejects the export); `mltok export-hf` converts old runs; `verify` can no longer be skipped. Measured on the real winner: 2957/2957 identical (one earlier set showed 1 exact tie in 1329).
3. **Freeze gate logic:** it compared the winner with itself when the winner was a baseline algorithm, and with a lossy variant that can never be chosen. It now compares against the best *other eligible* variant and prints inconvenient facts (custom BPE lost; a better-scoring ineligible variant exists).
4. **Possible handicap (unconfirmed):** `max_word_types: 400000` let our BPE see only the top 400K of 2.98M word types while HF BPE / SentencePiece saw all text. The cap is raised to 1.5M and a warning prints when coverage < 97 %. Whether this explains any of the `mlbpe` gap is unknown.

## Known limitations (not fixed)
* Text containing special tokens does not round-trip exactly (`<|im_start|>user` decodes as `<|im_start|> user`): the word-start marker is added per text segment. Plain-text pretraining is unaffected (documents are separated by ids); it matters for chat/SFT formatting.
* SentencePiece and the converted tokenizer may break an exact Viterbi tie differently (rare; equal probability, both lossless).

## Next experiments (in order)
1. Re-run selection with balanced data (`kaggle_v2.yaml`), 3 seeds, all algorithm families, vocab 32K/48K/64K.
2. Confirm the top-2 on the DGX with the `small` proxy (≈ 85M non-embedding params, 2.5 GB text).
3. If Unigram still wins: test a mark-constrained Unigram (prune pieces that start with a combining mark) — the one idea that combines what won with what the diagnostics suggested.
