# Thesis artifact inventory and publication

Inventory date 23 November 2026. The initial local branch, fetched GitHub branch and remote research ref all resolved to `research/mom-bpe-review` at `62201a396795dcc1476c8eef0b94ad2d7d0abe0c`. The initial tracked working tree and index were clean. The workspace contained 18 untracked audit files and 1,016 ignored files before this work created its staging, test and restore directories.

The [versioned JSON inventory](thesis-artifact-inventory-v1.json) reports each relevant file's path, byte size, working-file SHA-256, stage, tracked and ignored status, content class, credential and identifying-information review, redistribution status, reproducibility importance and storage destination. Its `fields` entry defines the columns of `records`. For existing Git files it also records the initial remote Git blob's SHA-256 separately from the local byte hash, so checkout line endings are unambiguous. The [thesis evidence index](THESIS_EVIDENCE_INDEX.md) links the primary scientific evidence.

The inventory hashes 1,126 initial files. The 95 already-published records comprise 92 paths in the original GitHub tree and three pinned model originals at their upstream revision. Another 95 local paths have exact verified private copies. This path count can include identical content at multiple local paths and differs from membership in the two snapshots. The remaining 788 records are local-only and 148 require manual review before export. Common credential patterns matched zero inventory records. Identifying-information indicators and unresolved checks are recorded per file.

## GitHub evidence

The initial remote already contained the historical baseline manifest and scope, reports, experiment configurations, evaluator and freeze protocol, dependency lock, engineering results, morphology packet, the source study, and the Hybrid/Picky and T8 comparisons. The inventory marks those paths individually. [Baseline report](../../audit/48k-baseline/REPORT.md) · [Historical recovery and limits](../../audit/48k-baseline/FOUNDATION.md) · [Engineering results](engineering-summary-v1.json) · [Corpus evidence](corpus-evidence-v1.json) · [Hybrid/Picky evidence](performance-results-v1.json) · [T8 and CPU proxy evidence](optimization-results-v1.json)

Four metadata files were selected for release on the research branch.

| File | Evidence added |
| --- | --- |
| `audit/48k-baseline/public-diagnostics-v1.json` | Aggregate baseline boundary, Unicode, morphology, vocabulary and domain counts. Includes baseline, fixture, code and completion-manifest pins. Omits example text, token sequences, raw documents and learned vocabulary strings. |
| `experiments/mom_bpe/thesis-artifact-inventory-v1.json` | Per-file hashes, purposes, status, storage decision and backup coverage across the research stages. |
| `experiments/mom_bpe/ARTIFACT_INVENTORY_V1.md` | Human-readable inventory, publication decisions and recovery gaps. |
| `experiments/mom_bpe/THESIS_EVIDENCE_INDEX.md` | Index from historical baseline and implementation through corpus, BPB, checkpoints and outstanding freeze requirements. |

No ignore rules were changed. The baseline ignores protect downloaded originals, caches, complete vocabulary dumps, tokenized examples and private recovery files. The experiment `/runs/` rule protects source captures, prepared corpora, model artifacts, optimizer state and temporary runs. Useful aggregates were released through the explicit public allowlist. [Baseline ignore rules](../../audit/48k-baseline/.gitignore) · [Experiment ignore rules](.gitignore)

## Preserved untracked audit outputs

Paths below are relative to `audit/48k-baseline/`. The inventory contains their full SHA-256 values. “Earlier snapshot” means a copy with the same size and SHA-256, including identical duplicates under a different local path. All 18 originals remain at their initial paths and outside Git.

| Path | Bytes | SHA-256 prefix | Evidence and exact-copy coverage |
| --- | ---: | --- | --- |
| `boundary-analysis.json` | 9,994 | `2add96370f14` | Earlier boundary diagnostic, local |
| `configuration.json` | 1,533 | `2369de31d0a1` | Earlier audit identifiers, local |
| `domain-analysis.json` | 5,356 | `a96283af7f7e` | Earlier measurement strata, local |
| `hf-compatibility.json` | 639 | `70dc6303dd78` | Encoder and export checks, earlier snapshot |
| `measurement-v3/boundary-analysis.json` | 25,489 | `5311f6053668` | Offset coverage and examples, earlier snapshot |
| `measurement-v3/boundary-root-causes.json` | 6,211 | `c40d7c9eea4b` | Exact and indirect boundary cases, earlier snapshot |
| `measurement-v3/domain-analysis.json` | 5,351 | `2e10e460401a` | Current diagnostic strata, earlier snapshot |
| `measurement-v3/morphology-analysis.json` | 21,250 | `80652470ca7a` | Detailed author and analyzer diagnostics, earlier snapshot |
| `measurement-v3/tokenizer-configuration.json` | 3,924 | `13433b1657b1` | Model configuration with learned examples, local |
| `measurement-v3/unicode-analysis.json` | 31,061 | `ece0dc9317eb` | Detailed normalization and Unicode cases, earlier snapshot |
| `measurement-v3/vocabulary-analysis.json` | 10,921 | `69769e08d0f0` | Vocabulary utility and examples, earlier snapshot |
| `measurements.json` | 1,726 | `8460cde3c44f` | Earlier measurements, local |
| `morphology-analysis.json` | 16,750 | `3b1a8ca09ad8` | Earlier surface and analyzer cases, local |
| `progress.json` | 1,715 | `c7c7109af327` | Earlier audit progress and recovery history, local |
| `retrieval.json` | 2,015 | `aab755488cd0` | Original artifact retrieval and benchmark recovery metadata, earlier snapshot |
| `tokenizer-configuration.json` | 3,924 | `13433b1657b1` | Byte-identical configuration duplicate, local |
| `unicode-analysis.json` | 31,061 | `ece0dc9317eb` | Byte-identical Unicode diagnostic duplicate, earlier snapshot |
| `vocabulary-analysis.json` | 10,921 | `69769e08d0f0` | Byte-identical vocabulary diagnostic duplicate, earlier snapshot |

## Private backup and restore

The existing Hub account and destination privacy were checked before any write. The new private allowlist contains 22 trained model artifacts, 19 event-only tokenizer checkpoints, all ten saved CPU proxy checkpoint generations, 42 reports or metadata files, and two bundle control records. Total size is 22,727,831 bytes. Proxy checkpoint ZIP integrity, filename SHA-256, run identity and expected state keys were checked in weights-only mode. Event checkpoints were admitted only if they contained matching chronological events and empty dataset sequence and flag fields. Source attribution and terms accompany the development artifacts. The backup excludes all corpus payloads, tokenized documents, caches and raw source captures. [Dataset source and permissions matrix](CORPUS_STUDY.md) · [Wikimedia terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) · [ODC-By database license](https://opendatacommons.org/licenses/by/1-0/)

The private bundle uses short member names so the independent restore works on Windows. Its member index maps every item to its original relative file path, byte size and SHA-256. The full private receipts and the independent restore directories are retained under the ignored `experiments/mom_bpe/runs/thesis-evidence-backup-v1/` directory.

Both independent restores completed. The new 95-member bundle matched every SHA-256 at blob revision `9db79dbd64572bab10c44c7ec8660d8dc103b443`. A separate read-only restoration retrieved all 13 files and 144,183 bytes from the earlier audit snapshot at its existing blob revision `78737cb615e81ba24878f6e7ae7ac02ca6d36d52`. Ten of the 18 preserved local paths have byte-identical copies in that earlier snapshot. The remaining eight retain earlier diagnostic or configuration details for local use and review. The [inventory](thesis-artifact-inventory-v1.json) records both immutable snapshot identifiers, receipt hashes and which original paths have an exact private copy.

## Local evidence and review gaps

- Keep downloaded baseline files at the original pinned model revision. The three local baseline files matched their exact manifest. The user-owned upstream original and the separately verified audit backup preserve distinct evidence.
- Keep original/extracted corpus JSONL, rendered pages, SQLite indices and token dumps in the workspace. The allowlist excludes source text from Sangraha, FineWeb2, SMC, IndicCorpV2, HPLT and Wikimedia. Rights-holder permission, attribution, privacy and overlap need source-level review before exporting any such data. Upstream render drift can prevent byte-identical reacquisition, so the local captures remain important.
- Keep installed dependencies, intermediate checkpoints, temporary caches and older diagnostic variants locally. The selected event-only checkpoints preserve the final verified event history per chosen fit. Other generations remain outside the new private snapshot. Private model and optimizer-state recovery still needs the exact data identity if work is resumed.
- Treat detected contact or credential patterns as review indicators and keep the matched values local. Pattern scans cannot establish the absence of personal information, especially in natural source text and derived artifacts. The published inventory contains decisions and fingerprints only.
- Preserve the source-qualified development measurements as preliminary. The historical 22,120-document benchmark remains unavailable. The final independently approved six-domain natural-text benchmark and orthography panel remain incomplete. The 7,744-slot runs and two-excerpt CPU proxy support exploratory comparisons under their recorded conditions.

## Integrity checks

The verification parsed 49 source JSON files and ten YAML configurations. It matched the three baseline files, all 12 completed audit members, five bounded source samples, two development manifests and four split digests, ten matched tokenizer artifacts and eight optimization source fingerprints. The offline CPU suite passed 70 tests, with five training-triggering tests deselected and zero failures, skips or warnings. No tokenizer or model training ran. The checked evidence retains the authored, exploratory and historical labels.
