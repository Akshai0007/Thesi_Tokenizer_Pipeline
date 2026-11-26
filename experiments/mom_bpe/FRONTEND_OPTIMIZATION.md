# Malayalam 48K frontend and Picky ablations

The analysis uses the verified 19,405-document development training pool and its separate 186-document validation split. The validation denominator is 999,950 canonical UTF-8 bytes. Fits at different training sizes retain separate identities and share only the validation set. M48 is the legacy MLBPE implementation refitted on each matched corpus. The immutable published 48K baseline, historical 22,120-document benchmark and future scientific holdout retain their separate identities. [Verified input and previous fits](SCALE_PERFORMANCE.md)

The [aggregate evidence](frontend-evidence-v2.json) records manifest identities, candidate configurations and hashes, per-arm measurements, paired document statistics, and each laptop preflight decision. Raw text, model files and individual token traces stay in the ignored run tree.

## What explains the original gap

Every M48, T5 and T8 trace reproduced the same canonical bytes across all 186 documents. M48 totaled 77,451 tokens and T8 totaled 107,627. T8 emitted 36,658 whitespace-only tokens and M48 emitted 3,534, a difference of 33,124. M48 also emitted 34,699 tokens spanning whitespace and other text, compared with two for T8. All remaining token classes combined contributed −2,948 to the T8-minus-M48 difference, leaving the observed 30,176-token gap. This decomposition measures emitted tokens. A controlled refit is needed to estimate how much of the difference can be recovered at the same vocabulary size.

M48 uses the native word pretokenizer and replaces ASCII spaces with a metaspace marker. It also prepends a synthetic marker that the decoder removes. T8 partitions script runs, whitespace runs, and residual graphemes, with cuts inside a reference unit coalesced before replay. The latter usually prevents a learned merge between a space and the following word. The word frontend can retain mixed-script letter runs, and its punctuation and digit grouping differs from the script frontend. Under the token payload classification, T8 emitted 13,115 punctuation/digit/other tokens versus M48's 10,670. T8 emitted 59 byte fallback IDs and M48 74. Byte fallback is too small to explain the compression gap.

The model inventories also differ. M48 starts with 1,898 initial symbols and retains 45,839 merges under combining-mark-start restrictions and selected akshara seeds. T5 and T8 each start with 2,399 symbols, including 512 protected akshara atoms. The event learner applies chronological merge and split-back events with a different eligibility policy. T5 stores 46,397 merges and 804 removals. T8 stores 46,503 merges and 918 removals. The learner and backend share the replay program, and emitted IDs resolved to the active vocabulary. Comparable occupied vocabularies can still distribute capacity differently across whitespace-bearing forms, script fragments and intermediate pieces. Length and token-frequency histograms are reported as aggregates.

The shared-site analysis reconstructs token payload bytes, including native metaspace and byte fallback, and verifies both the encoding's surface and the concatenated payload against every canonical document. It then scores cuts on the same 164,827 Malayalam akshara-interior sites and 165,308 grapheme-interior sites. Native decoder-prefix tokens with zero width contribute to token counts only. Byte continuation boundaries inside a UTF-8 scalar are recorded separately. A scalar-aligned cut is byte-backed when either neighbor is a fallback ID. On the common sites M48 had 1,045 lexical akshara cuts, T5 had 886 and T8 had 407. Earlier reports counted 559 M48 cuts under partial native offset coverage. These measurements use different coverage. The local largest-disagreement report retains representative canonical spans and each model's pieces for the eight documents with the largest T8-minus-M48 differences.

## Controlled frontend design

| Arm | Base partition | Single literal ASCII space attachable | Learner |
| --- | --- | --- | --- |
| T8 | Script runs | No | Guarded hybrid, Picky 0.9 |
| F-word | Word | No | Same guarded hybrid learner |
| F-space | Script runs | Yes | Same guarded hybrid learner |
| F-word-space | Word | Yes | Same guarded hybrid learner |
| M48 | Word with metaspace | Yes, via marker | Legacy seeded BPE |

The alternative plugin keeps the actual space in the merge domain before the next existing piece. It preserves multiple-space runs, tabs, line breaks, literal marker characters and ordinary decoder behavior. It neither synthesizes a document-start space nor normalizes it differently. The crossed event arms separate word partition and literal-space attachment effects. They approximate the attachment permitted by M48 while keeping exact reconstruction. Native M48 still differs in its prefix convention, atom initialization, restrictions and learner. Changing the partition also changes the types and frequencies estimated by a shared alphabet policy, so claims concern the frontend choice as a whole.

The Picky controls use the existing unguarded hybrid without Picky (T4), unguarded Picky (T5), guarded hybrid without Picky (T8-guard-only), and guarded Picky (T8). Only comparisons within the same manifest establish a matched contrast. Word-bounded BPE, whitespace-aware subword tokenization, and Picky vocabulary refinement have prior published implementations. The experiment tests their interaction on this dataset. [Sennrich et al.](https://aclanthology.org/P16-1162/) · [SentencePiece](https://aclanthology.org/D18-2012/) · [Picky BPE](https://aclanthology.org/2024.emnlp-main.925/)

## Matched 48K results on the full corpus

Every row used the same 99,999,974 canonical training bytes, occupied 48,000 vocabulary slots and reproduced 186 of 186 validation documents under both canonical and raw round-trip checks. `Used IDs` counts distinct IDs emitted on validation. Both cut columns count lexical cuts on common sites.

| Arm | Tokens | UTF-8 bytes/token | Fertility | Used IDs | Fallback IDs emitted | Akshara cuts | Grapheme cuts | Encode MB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M48 | 77,451 | 12.911 | 2.102 | 17,605 | 74 | 1,045 | 1,140 | 1.552 |
| T4 | 107,930 | 9.265 | 2.930 | 16,595 | 59 | 814 | 861 | 0.250 |
| T5 | 107,662 | 9.288 | 2.923 | 16,586 | 59 | 886 | 938 | 0.243 |
| T8-guard-only | 108,045 | 9.255 | 2.933 | 16,501 | 59 | 391 | 439 | 0.257 |
| T8 | 107,627 | 9.291 | 2.922 | 16,519 | 59 | 407 | 459 | 0.238 |
| F-word | 107,316 | 9.318 | 2.913 | 16,551 | 59 | 407 | 459 | 0.253 |
| F-space | 77,446 | 12.912 | 2.102 | 17,726 | 59 | 463 | 514 | 0.250 |

F-space removed 30,181 tokens relative to T8, a 28.0% reduction, with fewer tokens in all 186 documents and a median reduction of 101. Its five-token aggregate difference from M48 is a practical compression tie. F-space used fewer tokens in 69 documents, more in 86 and the same in 31, with a median difference of zero. It made 582 fewer lexical akshara cuts and 626 fewer lexical grapheme cuts than M48, while adding 56 and 55 relative to T8. F-space reduced standalone whitespace tokens from 36,658 to 2,958 and raised whitespace-bearing content tokens from two to 33,702 relative to T8. F-word alone saved 311 tokens and left the measured cut counts unchanged.

All full-corpus arms had zero byte-backed akshara or grapheme cuts at the scalar-aligned reference sites. M48 had 77,214 scalar-aligned transitions, 49 transitions inside UTF-8 scalars and two zero-width decoder-prefix tokens. F-space had 77,221 scalar-aligned transitions, 39 transitions inside scalars and zero zero-width tokens. T8 had 107,402 scalar-aligned transitions, 39 transitions inside scalars and zero zero-width tokens. These transition counts differ with the tokenizations. The underlying reference-site counts are shared. F-space took 316.6 seconds to fit and peaked at 2,549 MiB on this host. Its measured encoder throughput was close to T8's, and the native M48 encoder was about six times as fast in this session.

## Picky ablation on the full corpus

With the same 99,999,974 training bytes, enabling Picky at 0.9 saved 268 tokens without the guard (107,930 to 107,662) and 418 with it (108,045 to 107,627). The corresponding event histories contain 804 and 918 removals. The controls without Picky have zero removals. Lexical akshara cuts increased from 814 to 886 without guarding and from 391 to 407 with guarding. Grapheme cuts increased from 861 to 938 and from 439 to 459, respectively. All four arms emitted 59 fallback IDs. Picky modestly improved compression here and increased measured fragmentation. The much larger compression effect comes from allowing space-bearing merges. These controls gave little reason to tune additional thresholds on the same development set.

## Complete frontend comparison at 64 MB

On the identical 63,993,376-byte training slice, F-space used **30,172 fewer tokens than T8** and **three more than M48**. It reduced token counts in all 186 documents relative to T8, with a median paired reduction of 99.5 tokens. Relative to M48, 65 documents used fewer tokens, 95 used more and 26 tied. The median change was +1 token. The aggregate near-tie does not imply identical segmentation.

F-space emitted 2,965 whitespace-only tokens and 33,696 whitespace-bearing tokens that also contained other text. T8 emitted 36,658 and two, respectively. Both filled 48,000 slots, emitted 62 fallback IDs and reproduced every validation text exactly. F-space had 473 lexical akshara cuts on common sites, compared with 420 for T8 and 1,032 for M48. It recovers most of the compression gap while giving up 53 cuts relative to T8 and avoiding 559 relative to M48. Its same-session encoding throughput was similar to T8's. M48's native encoder was faster in the same session.

Changing only the base partition saved 322 tokens with space attachment disabled and 302 with attachment enabled. The punctuation/digit/other class accounted for reductions of 325 and 321 tokens in those comparisons. Allowing the space to attach saved 30,172 tokens with the script partition and 30,152 with the word partition. The two contrasts isolate the dominant effect of space attachment under this learner. F-word-space used 77,135 tokens, 299 fewer than M48, and made the same 473 lexical akshara cuts as F-space on this slice. It used fewer tokens than M48 in 85 documents, more in 70 and the same in 31.

| 64 MB comparison | Tokens | Lexical akshara cuts | Lexical grapheme cuts | Same-session encode MB/s |
| --- | ---: | ---: | ---: | ---: |
| M48 | 77,434 | 1,032 | 1,129 | 2.462 |
| T8 | 107,609 | 420 | 471 | 0.420 |
| F-word | 107,287 | 421 | 472 | 0.459 |
| F-space | 77,437 | 473 | 524 | 0.437 |
| F-word-space | 77,135 | 473 | 524 | 0.448 |

In the earlier focused session, F-space encoded at 0.215 MB/s and T8 at 0.212 MB/s. The timing variation across sessions makes these figures operational estimates. The underlying counts and byte reconstruction matched across sessions.

Use one CPU process at a time and a new output directory per manifest and arm. Reusing the same command and directory resumes compatible checkpoints.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:HF_HUB_OFFLINE = '1'
$manifest = 'experiments/mom_bpe/runs/scale48k-v1/a-100mb/manifest.json'
$work = 'experiments/mom_bpe/runs/frontend-optimization-v1/full100'
$prior = 'experiments/mom_bpe/runs/scale48k-next-v1/full100/fits'
foreach ($arm in @('F-space', 'F-word', 'F-word-space')) {
    python -m experiments.mom_bpe.frontend_fit --manifest $manifest --arm $arm --out "$work/fits/$arm"
}
foreach ($arm in @('T4', 'T8-guard-only')) {
    python -m experiments.mom_bpe.scale_fit --manifest $manifest --arm $arm --out "$work/fits/$arm"
}
python -m experiments.mom_bpe.frontend_analysis --manifest $manifest --out "$work/focused-recheck" `
    --arm "M48=$prior/M48" --arm "T8=$prior/T8" --arm "F-space=$work/fits/F-space"
```

## Proposed 38.8M proxy schedule

Keep M48, T5 and T8, and add F-space as the fourth arm with all tokenizers trained on the same full corpus. F-word-space remains a follow-up for a host that can fit it on the full corpus. The existing 8,192-target run design exercises integration. A first meaningful development comparison targets several hundred optimizer updates and reports learning curves and paired document loss.

Use the existing tied 38,799,936-parameter GPT, with width 448, seven layers, seven heads and a 512-token context. Fix the initialization policy, special IDs, optimizer, validation texts and canonical BPB scorer across arms. Use two independent exposure protocols.

- **Fixed source bytes** — take an ordered whole-document prefix up to 50,000,000 canonical bytes. Hash the exact document stream once. The measured development bytes per token suggest roughly 3.9–5.4M predictions across current candidates. Encode the training selection independently with every tokenizer and report the actual targets and updates.
- **Fixed training tokens** — plan 5,000,000 nonpadding predictions from the same ordered pool in one pass. Verify that every arm reaches this count before scheduling. Keep the exact byte and target ledgers and lower the shared budget if the least exposed arm cannot fill it. Track partial-document bytes only when offsets support them.

Begin with seed 17 and score checkpoints at 1M, 3M and 5M targets or at verified whole-document prefixes near 16, 32 and 50 MB. A batch target of 16,384 real predictions gives about 306 optimizer updates at 5M targets. The proposed common AdamW schedule uses a peak learning rate of `3e-4`, betas `(0.9, 0.95)`, weight decay `0.1`, global norm clipping at `1.0`, linear warmup through 5% of exposure, and cosine decay to `3e-5`. Set the exposure clock to canonical bytes or nonpadding targets for its respective protocol. Normalize accumulated loss by the actual nonpadding target count. The rough dense-model estimate `6 × parameters × predicted tokens` is 1.16 × 10¹⁵ FLOPs per 5M-target fit before padding, recomputation and attention overhead. Measure effective throughput and memory with an authorized small run before booking hardware. A confirmation uses seeds 17, 23 and 41 with the same trajectory, and may scale to 10–20M predictions with a reviewed corpus. Repeated epochs count as additional exposure and carry a separate unique-byte total. Report paired document BPB differences across seeds. This small proxy remains exploratory. [Compute scaling reference](https://arxiv.org/abs/2203.15556)

The current per-target sliding-window reference repeats prefixes and is too expensive at millions of targets. A production driver should batch document-local target blocks, right-pad each block, mask padding loss and reset context at documented block boundaries. Include BOS as context only and score EOS once for a complete document. Score every target exactly once. Use the same block policy at validation and check it against the reference scorer on equivalent local contexts. Compute BPB from total nats divided by ln(2) and total verified canonical UTF-8 bytes. At roughly 0.21 MB/s, pre-encoding a 50 MB selection with an event encoder takes about four minutes on this laptop, subject to the composition of the slice. Cache the encoded blocks by tokenizer, corpus and context policy so each seed can reuse them. Use gradient accumulation to reach the target batch size from smaller measured microbatches. Recover optimizer, schedule, RNG and exposure position from a verified checkpoint before continuing. The continuous-stream path is excluded. The new block driver and GPU run need separate verification and authorization. [Existing reference scorer](../../mltok/lib/canonical_bpb.py) · [Existing integration preflight](SCALE_PERFORMANCE.md#downstream-readiness)

## Resource limits and artifact preservation

Each new fit used one laptop CPU process. Preflight required the forecast peak with a 20% margin, an additional 1.25 GiB of available physical memory, 1 GiB of available commit, and 10 GiB of free storage. Smaller completed fits and the measured 64-to-100 MB growth of F-space informed later forecasts. The full F-word-space checks fell short of the commit reserve, so its result belongs to the 63,993,376-byte cohort. Peak working set and measured wall time are in the evidence file. Historical fit times reflect their original sessions. The encoding measurements compare models within one session per cohort.

The local snapshot `11d5909a6a3e96aa01f7443fa1d2fa118a715520091f00e82a7792b0b8500584` contains 40 verified members across eight new fits, including each final event checkpoint. An independent local restore verified every SHA-256, reloaded each model, checked the checkpoint against the event program, and exercised decoding. The archive and source files are on the same workstation. Off-machine storage for these Wikimedia-derived models awaits a permissions review. Existing private snapshot receipts and all original artifacts are retained. [Source and storage scope](SCALE_48K.md) · [Prior backup inventory](ARTIFACT_INVENTORY_V1.md)

## Offline verification

The new tests cover literal leading-space partitioning, exact Unicode reconstruction after reloading, native byte-fallback edge accounting, incompatible arm labels and byte-identical event artifacts after CPU checkpoint resumption on a small fixture.

```powershell
python -m pytest -q -p no:cacheprovider -rs --tb=short --show-capture=no --basetemp experiments/mom_bpe/runs/fo --ignore=tests/test_gpu.py
```

The offline CPU suite had 316 passes, one failure, zero skips and two protobuf deprecation warnings. The final focused frontend suite passed all nine cases, including a regression for reopening an analysis with byte-length and frequency histograms and verifying its checksum. `test_finished_work_is_pushed_then_restored_without_retraining` failed again when rerun in the milestone workspace and passed once in an isolated checkout of the base commit. A controlled probe on that base commit reproduced `WinError 5` when `os.replace` attempted to replace a checkpoint with an open reader. The previous checkpoint remained byte-identical and the write succeeded after closing the reader. This pre-existing Windows sharing risk affects the legacy proxy and local sync path. Resolve it before running that combination for a long experiment. The frontend fits use immutable event generations and the new recovery test passed.
