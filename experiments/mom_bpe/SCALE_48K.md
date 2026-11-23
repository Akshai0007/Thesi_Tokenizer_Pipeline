# Matched 48K Malayalam tokenizer capacity and evaluation

The 48K experiments use the existing event-BPE plugins with the same conservative Malayalam normalizer, script-run pretokenizer, special-token IDs, 256 byte IDs, document ordering and canonical UTF-8 budgets. The published mlbpe-48K model remains pinned and serves as a separate historical comparator. The scale plan admits only train and development validation. The historical 22,120-document evaluation corpus remains unavailable. The final natural-text and orthography panels still require the approvals in the fixed protocol. [Measured evidence and fingerprints](scale-48k-evidence-v1.json) · [Baseline](../../audit/48k-baseline/REPORT.md) · [Evaluation protocol](../../audit/48k-baseline/evaluation-protocol-v1.json)

## Source eligibility

| Source | Verified identity | Use and provenance |
| --- | --- | --- |
| Malayalam Wikipedia | Wikimedia `20231101.ml/train`, Hub revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, Parquet LFS SHA-256 `f2a0f5e7921f5cd56137cfce8f6da05bb885875f83e315b64d6971bb7810041b` | Local single-domain capacity diagnostic. Each accepted record keeps its page ID, link, page-history link, row-group coordinate, exact text SHA-256 and documented dump snapshot. The dataset exposes the snapshot date and page ID. Individual revision IDs are unavailable in this Parquet schema. |
| AI4Bharat Sangraha Verified Malayalam | `8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8`, `verified/mal` | Priority for scale B/C once traceable records and permission evidence are available. The sampled raw schema has `doc_id`, `text` and `type`. The CC BY 4.0 collection declaration and source types leave underlying web, PDF and transcript permissions unresolved for those records. |
| FineWeb2 Malayalam | `af9c13333eb981300149d5ca60a8e9d659b276b9`, converted Parquet `a092823f5cba550080cb8e56451fc9ac5def625c`, `mal_Mlym/train` | Priority for diverse, independently reviewed web records. Its ODC-By 1.0 declaration covers the database. Use original URLs and WARC provenance to establish the scope for page text. The bounded study did not establish individual permissions. |

The Wikimedia dataset card lists CC BY-SA 3.0 and GFDL. Wikimedia's terms describe attribution through article history and exceptions for imported or third-party material. The capacity corpus and its learned artifacts are retained locally pending review for redistribution. The Parquet range hashes are a receipt for the retrieved bytes. Its full-file LFS digest is publisher metadata because only selected ranges were fetched. The previously measured Wikisource material can diversify a reviewed development mix after its work-level permission and group provenance are checked. [Wikimedia dataset card](https://huggingface.co/datasets/wikimedia/wikipedia) · [Wikimedia licensing terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content) · [Sangraha card](https://huggingface.co/datasets/ai4bharat/sangraha) · [FineWeb2 card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) · [Prior source inventory and bounded evidence](CORPUS_STUDY.md)

The local capture reads selected Parquet columns in batches over a capped public connection. Each row group is committed separately with its range and file hashes. Corpus preparation checks independent fastText Malayalam confidence, Malayalam script proportion, Unicode validity and boilerplate markers. Automatic rejections are reported by cause, and the filter's treatment of short and code-mixed text limits its diagnostic scope. Accepted text retains valid joiners and orthographic variation. Exact text, canonical text and source ancestry are grouped in SQLite. Near duplicates use NFC and whitespace collapse for screening and exact character five-gram Jaccard verification at 0.85. The 8 MB pilot used a rare-first shared prefix order. The larger preparation uses a 512-bucket upper bound on set intersection to discard impossible candidates, and verifies all surviving candidates exactly. Hash collisions can only increase the upper bound. One document per connected group is assigned to train or development validation. The evaluation pool uses a stable hash split and a 1,000,000 canonical-byte limit. Train data uses only whole documents and reports any shortfall. Each split fingerprints its ordered canonical byte stream with an 8-byte length prefix per document to preserve document boundaries. Exposure counts exclude these prefixes.

The pinned Wikimedia capture retained 34 of 86 Parquet row groups and used 73,438,129 bytes of range transfers under the 75 MB cap. The first four groups yielded 3,999 articles and 21,085,072 raw text bytes. Screening accepted 3,405 articles and 16,349,717 canonical bytes. Exact Jaccard linked four near-duplicate pairs. The diagnostic selected 1,872 training articles with 7,999,910 canonical bytes and 275 separate development articles with 999,752 bytes. The train stream SHA-256 is `af04b3a3dea8b81fa84e45dd8fde0d9b10fc5f846f2c96415f12211c8eb86150` and development is `ef62f2fe42a57e46c55d9afc75b44e70a37b2369551b3230c197051f947eb4dd`. All four arms filled 48,000 slots on those same training articles.

The same captured prefix was prepared again with the bucket bound. Both pilot split files matched the earlier prefix-index preparation byte for byte. A restart of the completed T8 experiment verified and reused its original fit receipt.

| Arm | Dev tokens | UTF-8 bytes per token | Fallback IDs | Measured lexical akshara cuts | Byte-backed akshara cuts | Peak process RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T0 historical comparator | 82,478 | 12.1214 | 48 | 773, partial offset coverage | unavailable | — |
| T1 codepoint | 109,039 | 9.1688 | 126 | 2,879 | 0 | 652 MiB |
| T3 strict akshara | 112,444 | 8.8911 | 3,987 | 0 | 1,015 | 521 MiB |
| T5 hybrid + Picky | 108,833 | 9.1861 | 126 | 1,030 | 0 | 545 MiB |
| T8 guarded hybrid + Picky | 108,802 | 9.1887 | 126 | 706 | 0 | 554 MiB |

All five decoded the canonical text of all 275 development documents. Exact raw-text round-trip succeeded on 274, with the remaining document changed by the common canonical normalization. On this split T8 had 324 fewer cuts under the table's lexical metric and 31 fewer token IDs than T5. Including lexical cuts next to joiners gives 730 for T8 and 1,055 for T5. The historical model used fewer tokens than every newly fitted arm. Its training overlap with these documents is unknown, and its different offset coverage limits direct lexical-cut comparisons. These results establish 48K capacity for this training slice and describe tokenization behavior on encyclopedia text. A scientific benchmark and downstream BPB remain pending.

## Scale preparation and fit

| Plan | Canonical train target | Corpus and execution |
| --- | ---: | --- |
| A | 100,000,000 bytes | Local Wikimedia capacity check and bounded growth profiling. Results are exploratory even when all requested slots fill. |
| B | 1,500,000,000 bytes | Review-gated mixed-source development. Requires the completed indexed capture and independent permission ledger. |
| C | 8,000,000,000 bytes | Review-gated final training target. Needs sufficiently diverse eligible text, storage authorization, a new resource profile, and a separate sealed evaluation. |

Scale A used a verified prefix of 27 captured row groups. It selected 19,405 training articles with 99,999,974 canonical bytes and 186 development articles with 999,950 canonical bytes. The 26-byte training shortfall follows the whole-document budget. The train stream SHA-256 is `9fc7e6bc214465b6686c586585b155d076fb18a876d0927d9377534dfba303ac` and validation is `efdeb7e194c1941b8d8124d7060c6f5bfdfdf92a1ff4f833dbc2f4d56702f514`.

Merge-domain types grew from 123,525 at about 10 MB to 409,443 at about 50 MB and 670,732 at the full scale-A slice. Of these, 218,563 occurred at least twice. The local profile took 78.96 seconds with a 335 MiB peak. T1 began with 6,523 pair types meeting the trainer's minimum frequency of two, T3 with 99,898, and the shared T5/T8 hybrid initialization with 64,537. T1 can acquire new pair types after each merge. Its initial count cannot bound eventual vocabulary size. The column below characterizes the initial pair index size. The completed 48K fits used the smaller pilot. A full 100 MB fit has its own RAM, checkpoint and runtime measurements pending.

| Initial segmentation | Eligible pair types | Pair-to-merge-domain postings |
| --- | ---: | ---: |
| Codepoint | 6,523 | 7,278,555 |
| Strict akshara | 99,898 | 3,217,309 |
| Hybrid | 64,537 | 3,741,493 |

Use a fresh output directory for each new recipe or input identity. For A, the first command reads public source data under a fixed 75 MB transfer cap. A/B/C never fetch data during fitting. The 8 MB pilot uses the first four captured groups and a distinct run identity.

```powershell
$root = 'experiments/mom_bpe/runs/scale48k-v1'
$lid = 'experiments/mom_bpe/runs/corpus-study-v1/lid/lid.176.ftz'
$env:CUDA_VISIBLE_DEVICES = '-1'
python -m experiments.mom_bpe.scale_wikipedia --out "$root/wiki" --groups 53 --transfer-cap 75000000
python -m experiments.mom_bpe.scale_data capacity --capture "$root/wiki" --lid $lid --group-limit 4 --target 8000000 --out "$root/reproduced-pilot"
python -m experiments.mom_bpe.scale_data capacity --capture "$root/wiki" --lid $lid --group-limit 27 --target 100000000 --out "$root/a-100mb"
python -m experiments.mom_bpe.scale_capacity --manifest "$root/a-100mb/manifest.json" --out "$root/a-100mb/capacity.json"
foreach ($arm in 'T1','T3','T5','T8') {
    python -m experiments.mom_bpe.scale_fit --manifest "$root/a-100mb/manifest.json" --out "$root/a-100mb/fits/$arm" --arm $arm --fit-only
}
```

Use the pinned fastText wheel and model from the prior corpus study. The Windows range reads here used the operating system's CA trust in Python. The corresponding launcher is `python -c "from pip._vendor import truststore; truststore.inject_into_ssl(); import runpy; runpy.run_module('experiments.mom_bpe.scale_wikipedia', run_name='__main__')" --out "$root/wiki" --groups 53 --transfer-cap 75000000`.

For larger plans, first provide revision-pinned local capture chunks, the pinned fastText resource manifest, and the independent review ledger described in the existing corpus workflow. Use the same produced training manifest for all arms. The reviewed materializer gives domain allotments and source priorities to whole documents and reserves proposal holdout groups before training. The fitting plan exports train and development validation only. A 1.5 GB or 8 GB raw cap can yield fewer canonical bytes after normalization or domain quotas. The manifest reports both shortfalls. [Review gates and source preparation commands](CORPUS_STUDY.md)

```bash
export CUDA_VISIBLE_DEVICES=-1
python -m experiments.mom_bpe.corpus_build --captures "$CAPTURES" --lid "$LID_RESOURCES" --out "$STUDY" --index-only
python -m experiments.mom_bpe.scale_data reviewed --study "$STUDY" --review "$REVIEW" --scale B --out "$SCRATCH/scale-b-v1"
python -m experiments.mom_bpe.scale_data reviewed --study "$STUDY" --review "$REVIEW" --scale C --out "$SCRATCH/scale-c-v1"
# Set the approved scale directory to scale-b-v1 or scale-c-v1 before running.
for arm in T1 T3 T5 T8; do
  python -m experiments.mom_bpe.scale_fit --manifest "$SCRATCH/$SCALE_DIR/manifest.json" --out "$SCRATCH/$SCALE_DIR/fits/$arm" --arm "$arm" --fit-only --allow-large
done
```

A restart uses the exact same command and paths. The existing trainer saves event generations every 1,024 events and at completion. Resumption replays the events and rebuilds in-memory counts. Training and artifact fingerprints are checked before reusing a final fit. Inspect `fit.json` for occupied slots, merge and removal counts, exhaustion reason, elapsed time, peak process RSS and per-generation checkpoint sizes. Once a fit reaches 48,000 occupied slots, omit `--fit-only` to measure fertility, bytes per token, exact and canonical round-trip, lexical boundary and byte-backed cuts, covered and excluded transitions, fallback and vocabulary utilization by source and domain, and encoding throughput. `--arm T0` compares the pinned historical tokenizer on the same development documents and records its native special IDs and offset coverage. Its prior BPB remains historical.

Fit throughput depends on unique merge domains, examples, pair postings and their update frequencies. Those structures remain in memory even when acquisition and corpus selection stream. The diagnostic process reports peak RSS for each arm, and currently there is no validated linear RAM or runtime estimate beyond A. For a first university DGX CPU reservation use one arm at a time, one trainer process, 4 allocated CPU threads, 32 GiB RAM and 100 GB of private scratch as provisional headroom. Before B, profile a smaller stage and increase RAM from its measured type and pair growth. Budget at least 5–10 times the canonical corpus size for capture, indexing, JSONL, SQLite transaction logs and checkpoints, then measure the actual ratio. C needs its own preflight and a stop point if projected memory or scratch approaches the reservation. These are planning estimates. No workload has run on DGX.

## Mechanism contrasts and downstream plan

The planned contrasts use the same data and require 48,000 occupied slots. T4 minus T1 measures hybrid initialization. T5 minus T4 measures Picky refinement without guarding. T8 minus T5 measures guarding with Picky enabled. `T8-guard-only` keeps hybrid initialization and disables Picky, so its contrast with T4 measures guarding without Picky. Compare the two guard contrasts for their interaction. `T8-guard-off` cross-checks equivalence with T5 for the same events and IDs. T3 provides the strict-akshara comparison. These contrasts use separate output paths and the existing plugins. Morphology stays disabled.

In the 8 MB pilot, hybrid initialization accounts for the largest measured drop in lexical akshara cuts. T4 reduced the count from 2,879 to 1,029 relative to T1 and added 143 token IDs. Picky saved 349 IDs without guarding and 373 with guarding, while the measured cut count rose from 1,029 to 1,030 without guarding and from 687 to 706 with guarding. Guarding reduced measured cuts by 342 without Picky and by 324 with Picky. The guard-off control reproduced T5's initial alphabet, atoms, chronological events, preprocessing, and control-token IDs. These comparisons involve one domain and one data split, so the interaction needs confirmation on the larger reviewed corpus.

For the single-seed integration, build the tied GPT at width 448 with 7 layers, 7 heads, 512-token context, and seed 17. This configuration has 38,799,936 parameters. Fit T1, T5 and T8 separately under a fixed source-byte slice and a fixed exact target-token budget. Include T3 if occupancy and the fidelity gate justify it. For confirmation, the proposed 12-layer, 12-head, width-768 model with the same context has 123,980,544 parameters. Use seeds 17, 23, 41 after approving a common schedule. Keep tokenizer vocabulary and control IDs, document order, initialization, optimizer, attention policy and validation membership fixed. Record actual nonpadding targets, EOS counts, updates, input positions, source bytes, tied embeddings, non-embedding parameters and total parameters, and estimate FLOPs with saved real and padded token counts. Log partial-document source exposure only where exact offsets support it.

```powershell
python -m experiments.mom_bpe.scale_proxy_plan --manifest "$root/pilot-rarity/manifest.json" --tokenizer "$root/pilot-rarity/fits/T5" --out "$root/pilot-rarity/T5-proxy-plan.json" --source-bytes 100000 --predicted-tokens 8192
```

The dry ledgers passed exact encoding checks for T1, T5 and T8. The shared source-budget slice selected 27 complete documents and 99,921 bytes. Those documents would produce 10,957, 10,896 and 10,874 targets respectively, including EOS. The 8,192-target slice used 55,128 complete-document bytes in each arm, plus 18,114, 18,624 and 18,778 verified partial-document bytes respectively. All three ledger entries use the same 999,752 validation bytes. No language-model training started.

Integration uses the project event encoder. BOS is context, EOS is one scored target per complete document, and padding contributes zero loss. A canonical development score sums held-out nats across documents, divides by ln(2), and divides by the exact canonical UTF-8 byte count. The larger run needs a verified document-local batch implementation, identical checkpoint recovery, predeclared optimizer and schedule, and a locked evaluation panel and decision rule. Keep any integration score exploratory. Before freezing a final panel, screen source ancestry, exact hashes and near duplicates against the pilot and scale-A documents as well as all later development and training sets. Final comparisons require an independently frozen panel and paired document and seed uncertainty. Physical GPU 7 is prohibited, and any future DGX GPU job needs explicit authorization plus physical GPU availability and assignment checks. [BPB scorer](../../mltok/lib/canonical_bpb.py) · [Checkpoint design](../../PIPELINE.md) · [Previous CPU integration evidence](BOUNDARY_OPTIMIZATION.md)

## Verification and artifact storage

The scoped offline CPU suite passed 204 tests. The full CPU run passed 304, skipped two CUDA-only tests and reported two architecture-test errors while reading UTF-8 files under the Windows CP1252 default. Both architecture tests passed with `PYTHONUTF8=1`. The full and scoped runs each emitted two protobuf deprecation warnings from the same SentencePiece contract test. Use UTF-8 mode for future Windows runs.

The release contains aggregate measurements and hashes. The captured text, prepared splits, trained tokenizers, and checkpoints remain in the ignored run tree pending review of storage and redistribution scope. Prior private snapshots retain their independently verified restoration receipts. [Prior backup inventory](ARTIFACT_INVENTORY_V1.md)
