# Dataset recovery and bounded CPU engineering

## Historical recovery

The original 22,120-document DGX evaluation set remains unavailable. The archived `c1321250cd78` fingerprint identifies shard filenames and sizes. There were no recoverable evaluation shard candidates or original ordered document hashes in the inspected local pipeline paths, manifests or previous DGX inventories. The live private experiment archive had 301 files at the same pinned revision recorded in the baseline audit. The private raw backup had the same three source streams totaling 7,518,303,142 compressed bytes. It has raw sources and the earlier audit records the associated configuration, but neither the identity of the exact streams consumed by DGX nor the original evaluation ordering is established. The verified backup repository also supplied no original shard candidate. The live checks read metadata and downloaded zero private corpus bytes. Historical metrics remain associated with their archived evaluation identity. [Earlier recovery evidence](../../audit/48k-baseline/FOUNDATION.md)

## Replacement source draft

`ml-wikipedia-replacement-source-draft-v1` stages seven pinned, main-namespace Malayalam Wikipedia revisions. The 130,539 bytes of original wikitext matched the published revision digests. Five pages (76,709 bytes) are assigned to draft training and two (53,830 bytes) to draft evaluation. Their page families are disjoint. The small set passed exhaustive exact and five-gram near-duplicate screening after NFC and whitespace normalization. The ledger records revision timestamps, stable page and revision IDs, attribution URLs, content SHA-256, raw and canonical UTF-8 byte counts, split assignments and screening status. The source and split files remain in the ignored local run directory. [Replacement draft ledger](replacement-draft-v1.json)

This is an independently versioned staging corpus. It keeps unrendered wikitext and covers only the encyclopedic domain. Wikimedia's text terms provide the starting license basis; exact attribution, exclusions for third-party content and redistribution conditions require review for each selected page. Full-document prose extraction, duplicate screening of the resulting prose, other domain sources, split and license review by two maintainers, and the independent orthography panel remain pending. Seal the final held-out set before selecting tokenizer hyperparameters. [Wikimedia terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) · [Evaluation protocol](../../audit/48k-baseline/evaluation-protocol-v1.json)

## Engineering fits

Every measurement in this section is **ENGINEERING_DIAGNOSTIC_ONLY**. Each arm used the same 32 repository-authored rows, comprising four repeated texts and 3,456 canonical UTF-8 bytes. Preprocessing, control IDs and the fallback policy were shared. All three candidates used a nominal budget of 288 slots. The reference fit, interrupted fit and serialized reload produced identical model bytes and token IDs within each arm. The simulated interruption occurred after one durable merge. A fresh fit invocation restored that generation from disk. No morphology was used.

| Arm | Occupied / 288 | Saved generations | Reference fit seconds | Resumed fit seconds | Peak RSS during reference fit |
| --- | ---: | ---: | ---: | ---: | ---: |
| A codepoint | 288 | 13 | 0.143 | 0.160 | 298,156,032 bytes |
| B grapheme | 282 | 7 | 0.060 | 0.085 | 298,500,096 bytes |
| C akshara | 282 | 7 | 0.102 | 0.088 | 297,160,704 bytes |

Peak RSS includes the process and imported libraries. The small repeated corpus and tiny fit times do not predict production resource needs. Reference and resumed measurements are separate attempts. All arms round-tripped the 54 ordinary cases exactly and after normalization. On the 11 dedicated fidelity cases they passed 9 raw and 11 normalized checks. The intended canonicalization changes account for the two raw differences.

| Arm | Eligible ordinary transitions | Akshara / EGC breaks | Excluded transitions | Fallback IDs from absent / pruned units |
| --- | ---: | ---: | ---: | ---: |
| A | 354 | 228 / 228 | 4,035 | 1,631 / 1,792 |
| B | 3 | 0 / 0 | 6,836 | 6,092 / 523 |
| C | 3 | 0 / 0 | 6,836 | 6,092 / 523 |

Independent unit attribution matched every observed fallback count. On the Unicode stress panel, B and C had zero eligible transitions after exclusions. Their zero break counts cannot establish orthographic quality. The engineering artifacts do not support comparison with the historical production tokenizer. [Immutable engineering summary](engineering-summary-v1.json)

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:HF_HUB_OFFLINE='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
foreach ($arm in 'A','B','C') {
    python experiments/mom_bpe/engineering.py --out experiments/mom_bpe/runs/engineering-v1 --arm $arm
}
```

After the data freeze, the scientific path uses separate bounded inputs and preserves the target at 48,000 occupied entries. An underfilled arm retains its fit receipt and stops before evaluation. The current source draft lacks the domain coverage, licensed prose and review evidence needed to authorize that comparison. Its capacity to fill 48K has not been established. [Scientific pilot instructions](PILOT.md)

## Encoder and canonical BPB

The decoder now rejects non-integer, out-of-range and unallocated IDs. The reserved control IDs decode to their registered spellings. Allocated fallback IDs still decode their bytes, with the Unicode replacement character for malformed UTF-8 sequences. The previously reviewed encoder fingerprints remain loadable when their grammar and runtime match. Fixture tests checked full ID sequences after resume and reload.

A model-backed CPU test matched the canonical reference scorer against independent unpadded, serial scoring. Each document starts with BOS as context and scores every text ID and one EOS. The bounded context window uses local positions after truncation. Padded batch positions receive an attention mask and contribute no target loss. The denominator sums only frozen canonical text bytes, including the UTF-8 bytes represented by unchanged fallback sequences. The test covered NFC byte-length changes and a document longer than the context window. This validates the reference adapter against a small synthetic causal model. Thesis-model integration and measured downstream BPB remain pending.

The preparation suite passed 87 tests. The later offline CPU regression passed 185 tests with zero failures or skips and two deprecation warnings from the existing SentencePiece contract test. The added replacement-draft regression passed separately. Actual CPU engineering fits used the authored corpus only.

## Morphology reviewer instructions

Preserve the 108-prompt packet under its current manifest. Give each reviewer a separate blind copy and keep the evidence sheet closed until both initial judgments are recorded. Validate the supplied context or add a gloss, confirm the lemma family, and record raw and canonical codepoint offsets separately. Use `ambiguous` when multiple readings remain defensible and `abstain` when the analysis cannot be projected. Keep those rows in the coverage report and record the reason. Open the analyzer evidence during adjudication, preserving its readings as suggestions. A training label needs independent reviewer agreement and a surface-aligned rule accepted by the projection engine. Keep training families distinct from held-out morphology. Save completed sheets under a new identity. The current authorized label count is zero. [Blind review packet](morphology-review-v2/manifest.json)
