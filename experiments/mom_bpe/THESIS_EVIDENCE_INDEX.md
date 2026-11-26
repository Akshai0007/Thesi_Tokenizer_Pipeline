# Malayalam tokenizer thesis evidence index

The archived 48K benchmark, authored engineering diagnostics, exploratory tokenizer comparisons and small CPU proxy runs are separate evidence sets. The independently approved final benchmark is still incomplete. The [artifact inventory report](ARTIFACT_INVENTORY_V1.md) and [file-level inventory](thesis-artifact-inventory-v1.json) document provenance, visibility, hashes and recovery coverage.

## 1. Historical 48K tokenizer

The original model is pinned to revision `cff0d31c84b598436087f21cc8183021993831e1` and tokenizer SHA-256 `2f4b7e8b0706007f7ab3f0b8b2cc1b169094072c334585dabc4081d455e69ff5`. The historical benchmark used an evaluation corpus recorded as 22,120 documents. Its original ordered text remains unavailable. [Baseline artifact manifest](../../audit/48k-baseline/baseline-artifacts.json) · [Historical benchmark and recovery findings](../../audit/48k-baseline/FOUNDATION.md)

## 2. Baseline audit and limitations

The 48K model was evaluated on authored diagnostic fixtures. These results document exact and normalized fidelity, Unicode edge cases, token and domain metrics, offset coverage, and the limits of comparisons to historical results. The aggregate publication omits individual text and token examples. [Baseline audit](../../audit/48k-baseline/REPORT.md) · [Public diagnostic aggregates](../../audit/48k-baseline/public-diagnostics-v1.json) · [Completed measurement manifest](../../audit/48k-baseline/measurement-v3/COMPLETE.json)

## 3. MOM-BPE implementation and independent review

The reviewed prototype supports codepoint, extended-grapheme and Malayalam akshara initial units with deterministic event or rank replay through the project encoder. The review covers interfaces, normalization, byte fallback, serialization and recovery. [MOM-BPE specification](README.md) · [Independent review](REVIEW.md) · [Reviewed ablation configuration](arms.yaml) · [MOM-BPE entry point](../../mltok/plugins/algorithm/mom_bpe.py)

## 4. Orthographic and morphological mechanisms

The surface projector uses reviewed confidence weights and abstains when the proposed boundary is ambiguous or cannot be projected to the written form. The reviewer materials include independent-review templates and desk evidence. The packet has no authorized training annotations. [Morphology review packet manifest](morphology-review-v2/manifest.json) · [Review procedure](RECOVERY_AND_ENGINEERING.md#morphology-reviewer-instructions) · [Analyzer and alignment findings](review-results-v1.json) · [Morphology dependency lock](../../audit/48k-baseline/morphology-dependencies.json)

## 5. Hybrid and Picky BPE development

The first natural-text exploratory comparison used 47 training and 12 validation articles. Matched new fits filled 7,744 slots. T5 used 110 fewer validation IDs than T4 and reduced training entries never emitted independently from 605 to 55. The pinned reference check covers removal events, active vocabulary and replay. [Report](PERFORMANCE.md) · [Experiment configuration](performance-arms.yaml) · [Measurements and reference parity](performance-results-v1.json)

## 6. T8 Boundary-Guarded Hybrid

T8 reduced observed lexical akshara breaks from 722 to 563 with 32 additional validation IDs at the same 7,744-slot budget. Both arms emitted 20 fallback byte IDs. The guard-off control reproduced the T5 event program. The larger capacity probes stopped near 10.3K occupied entries. [T8 report and DGX preparation commands](BOUNDARY_OPTIMIZATION.md) · [T8 configuration](optimization-arms.yaml) · [Matched results and capacity evidence](optimization-results-v1.json)

## 7. Vocabulary utility and fallback

Coverage, occupancy, lexical utilization, fallback IDs and break counts are reported with their original denominators. The early review attributes the heavy fallback trend to unseen initial units on the small fixture. The later byte-offset audit identifies byte-backed akshara cuts omitted from the older lexical-only view. [Prototype fallback attribution](review-results-v1.json) · [Baseline vocabulary aggregates](../../audit/48k-baseline/public-diagnostics-v1.json) · [T8 transition coverage and capacity](BOUNDARY_OPTIMIZATION.md#what-caused-the-previous-breaks)

## 8. Small CPU model-backed BPB

T5 scored 1.260310 BPB at fixed source bytes and 1.293476 at fixed training tokens. T8 scored 1.274335 and 1.314152. These two-excerpt, single-seed measurements exercise canonical UTF-8 denominators, document-local context, BOS/EOS and padding. They support retaining T5 as the downstream lead for a larger comparison. [CPU proxy scope and results](BOUNDARY_OPTIMIZATION.md#executed-cpu-proxy) · [Model, exposure and checkpoint receipts](optimization-results-v1.json) · [Canonical BPB scorer](../../mltok/lib/canonical_bpb.py)

## 9. Corpus provenance

The corpus survey preserves pinned source receipts for Sangraha, FineWeb2, Malayalam Wikipedia sentences, IndicCorpV2 and HPLT. The matched development manifests record attributed Wikipedia and Wikisource revisions, document and split hashes, normalization, deduplication and separation. The replacement draft remains a separate identity. [Source-quality and permissions matrix](CORPUS_STUDY.md) · [Survey fingerprints](corpus-evidence-v1.json) · [Wikipedia development manifest](performance-results-v1.json) · [Expanded development manifest](optimization-results-v1.json) · [Replacement draft](replacement-draft-v1.json)

## 10. Checkpoint and reproducibility verification

The CPU engineering record includes interrupted-versus-uninterrupted model and token-ID equality. The T8 record includes resumed-fit and model-backed proxy receipts. The artifact inventory records verified snapshot coverage, while the new private restore contains a member map back to each original path. [Engineering results](engineering-summary-v1.json) · [Engineering and recovery report](RECOVERY_AND_ENGINEERING.md) · [Private backup and restore status](ARTIFACT_INVENTORY_V1.md#private-backup-and-restore) · [Snapshot implementation](../../mltok/snapshots.py)

## 11. Remaining scientific evaluation

Complete rights review and independent approval for each selected natural-text record. Seal the 2,400-document, six-domain held-out set and the separate orthography panel under the existing protocol. Morphology weight claims need adjudicated alignments and disjoint families. Use a sufficiently occupied 48K development corpus and matched downstream budgets before assessing thesis-level BPB or historical-baseline superiority. The 7,744-slot comparisons and small CPU proxy are preliminary. [Evaluation freeze protocol](../../audit/48k-baseline/evaluation-protocol-v1.json) · [Read-only freeze verifier](verify_frozen.py) · [CPU pilot procedure](PILOT.md)
