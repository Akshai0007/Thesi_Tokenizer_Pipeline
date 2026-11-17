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

## Run 2 (Kaggle 2×T4, one-shot notebook, `kaggle_v2.yaml`, 2 h 10 min, balanced data)
**Setup:** same proxy (tiny, 3 seeds); training data now interleaved: FineWeb-2 280 MB (40 %) / Sangraha 210 MB (30 %) / Wikipedia 210 MB (30 %) — exactly the configured weights (bug 1 fixed). Two GPUs ran 8 proxy jobs each in parallel. Checkpoints went to the Hugging Face repo and were restored in a later cell (59 files).

**Results at 32 000 vocab (BPB, mean ± std of 3 seeds; lower is better)**

| variant | BPB | fertility | invalid-start tokens | note |
|---|---|---|---|---|
| **SentencePiece Unigram** | **0.7456 ± 0.0032** | 1.949 | 15.1 % | best |
| `mlbpe` (constraint + seeds), word pre-tok | 0.7620 ± 0.0023 | 2.006 | **0.3 %** | best BPE |
| `mlbpe_plain` (same trainer, no constraint/seeds) | 0.7647 ± 0.0023 | 2.021 | 10.6 % | control |
| stock HF BPE | 0.7660 ± 0.0020 | 2.030 | 11.2 % | not lossless (stress 0.82) |
| SentencePiece BPE | 0.7751 ± 0.0025 | 1.969 | 12.7 % | |
| `mlbpe`, whitespace pre-tok | 0.7797 ± 0.0068 | 1.899 | 0.3 % | lowest BPE fertility, worst BPE BPB |

**Vocabulary sweep (SentencePiece Unigram, 3 seeds)**

| vocab | BPB | fertility | total params (non-embedding fixed at 11.06M) | step vs previous |
|---|---|---|---|---|
| 32 000 | 0.7456 ± 0.0032 | 1.949 | 23.5M | |
| 48 000 | 0.7380 ± 0.0005 | 1.840 | 29.7M | −0.0076 (z ≈ 3.3, clear) |
| 64 000 | 0.7349 ± 0.0021 | 1.769 | 35.8M | −0.0031 (z ≈ 2.0, **borderline**) |

(z = gap / standard error of the difference of two 3-seed means.)

**What this shows**
* **Unigram beats every BPE variant, in both runs, on two data mixes.** Best BPE (`mlbpe`) is +0.0164 BPB behind (z ≈ 5.9); SentencePiece BPE +0.0295. Four independent BPE implementations are all behind, so it is the algorithm class, not one implementation.
* **The constraint is free but not a win.** `mlbpe` vs `mlbpe_plain` (identical trainer): 0.7620 vs 0.7647 (z ≈ 1.2) — a tie that favours the constraint slightly while removing ~97 % of invalid-start tokens. "Valid tokens at no measurable cost" holds; "valid tokens improve the model" is not shown.
* **Fertility is not a cross-algorithm predictor.** `mlbpe` with whitespace pre-tokenization has much lower fertility (1.899 vs 2.006) and a clearly *worse* BPB (0.7797 vs 0.7620). Within one algorithm, a larger vocabulary lowers both together.
* **The `max_word_types` handicap is not the explanation:** HF BPE (sees all text) scored the same as `mlbpe_plain` (95.4 % coverage): 0.7660 vs 0.7647.
* **The vocabulary choice is a cost trade-off, not a result.** 64K gives −0.4 % BPB over 48K and 9 % fewer tokens per word than 32K, but adds 6M embedding parameters here. Part of the gain may simply be extra parameters (the proxy's non-embedding size is fixed), so the right vocabulary depends on the size of the model you will train.

**Not shown:** anything at larger scale (proxy is 11M non-embedding params, ~8.5M tokens); the interaction with a bilingual vocabulary.

**Operational findings (fixed afterwards)**
* The first mid-training checkpoint of **every** job was pushed (283 MB) because each job is a new process whose "last push" timer started at zero, and pushes ran inside the training loop. Training stalled during uploads, and a Hub outage (HTTP 500/503) stalled one worker for ~100 s. Fix: pushes run in a background thread, and a checkpoint is only pushed after a full interval (regression tests added). Old checkpoint objects remain in the repo history; `store.squash_every` / `HfApi.super_squash_history` reclaims the space (irreversible).
* The dry-run gate "both GPUs used" failed only because the dry run's results were restored from the store (no job ran). The gate now judges only jobs that ran in the current session.
* The freeze gate compared the winner/runner-up gap to one seed standard deviation, which is not a significance test (it called the borderline 64K-vs-48K gap clear). It now uses the standard error of the difference (≥ 2 required; borderline cases are labelled).

## Known limitations (not fixed)
* Text containing special tokens does not round-trip exactly (`<|im_start|>user` decodes as `<|im_start|> user`): the word-start marker is added per text segment. Plain-text pretraining is unaffected (documents are separated by ids); it matters for chat/SFT formatting.
* SentencePiece and the converted tokenizer may break an exact Viterbi tie differently (rare; equal probability, both lossless).

## Next experiments (in order)
1. Decide the target model size (it fixes the sensible vocabulary range), then confirm the top candidates on the DGX with the `small` proxy (≈ 85M non-embedding params, 2.5 GB text): Unigram 48K/64K vs best BPE.
2. Test a mark-constrained Unigram (prune pieces that start with a combining mark): the one idea that combines what won with what the diagnostics suggested. Expect at best a tie on BPB (the constraint was neutral for BPE).
3. If the model will become bilingual: repeat on a Malayalam+English mix before committing to a vocabulary.
