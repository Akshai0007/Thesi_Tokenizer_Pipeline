# Malayalam tokenizer performance research

This study separates exploratory tokenizer measurements from the independently reviewed thesis benchmark. The archived mlbpe-48K artifacts retain the revision and hashes in [the baseline manifest](../../audit/48k-baseline/baseline-artifacts.json). The original 22,120-document evaluation corpus remains unrecovered. The release and approval conditions of the [scientific evaluation protocol](../../audit/48k-baseline/evaluation-protocol-v1.json) still apply. The [results and source receipt](performance-results-v1.json) contain provenance and aggregate measurements.

## Development benchmark

The development corpus contains attributed plain-prose excerpts from pinned Malayalam Wikipedia article revisions. The acquisition received 61 bounded HTML renders, totaling 10,487,351 decoded HTML bytes. The corpus shares a single normalizer, versioned script and provenance checks, and the previous study's duplicate index. Retained areas exclude markup, media, quoted/source-excerpt indicators, references, and low-Malayalam or short paragraphs. The source and contributor-history links identify each excerpt under the Wikimedia text-reuse terms and CC BY-SA 4.0. Extraction flags cannot establish original authorship or clearance of an unidentified embedded work. Raw pages and extracted excerpts remain local. [Wikimedia text-reuse terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) · [MediaWiki revision rendering API](https://www.mediawiki.org/wiki/API:Parsing_wikitext)

The published development metadata pins 47 training articles and 12 validation articles. They contain 614,916 and 160,695 canonical UTF-8 bytes respectively. All fitted arms used the same ordered training records. Raw text and normalized text have distinct hashes. The split excluded the seven earlier replacement-draft article IDs. It selected one record per connected component over page ancestry, raw and canonical SHA-256, and NFC-whitespace five-gram Jaccard at 0.85. No duplicate witnesses were observed among accepted records. The corpus covers encyclopedic text. Its document IDs and near-duplicate groups must be excluded from any final held-out study.

The pinned independent fastText model assigned Malayalam as the top label in 137 training and 36 validation windows. Malayalam comprised 98.80% and 99.22% of letters respectively. Training text retained 15 automatically flagged unattached marks and five legacy-chillu patterns. Both splits had zero replacement characters and repeated virama pairs. These are source and extraction diagnostics. The existing conservative normalizer is shared by the fitted arms. The package contains no independent sign-offs and has no scientific-freeze approval.

## Candidate definitions and implementation

| Arm | Configuration |
| --- | --- |
| T0 | Published mlbpe-48K, evaluated descriptively on the new validation input with its own encoder and vocabulary |
| T1 | Matched codepoint initialization |
| T2 | Extended-grapheme initialization |
| T3 | Malayalam akshara initialization |
| T4 | Frequent Malayalam akshara atoms and Unicode scalar backoff |
| T5 | T4 with Picky refinement at strict IoS threshold 0.900 |
| T6 | T4 with confidence-weighted reviewed surface morphology |
| T7 | T6 with Picky refinement at the same threshold |

The hybrid ranks multi-codepoint Malayalam aksharas by weighted occurrence, keeping those with at least eight observations, up to 512 and at most one third of the initial alphabet capacity. It reserves observed scalars from all training text, including scalars occurring only inside protected atoms. Unselected units decompose into those scalar sequences. Unavailable scalars use their exact UTF-8 bytes. A protected atom cannot be divided by a learned merge or removed by Picky. Scalar backoff makes some interior orthographic transitions possible, which is why fallback alone cannot select this arm.

The `picky_bpe` plugin can independently disable Picky. When enabled it computes IoS from actual nonoverlapping merges and residual token frequency, removes qualifying learned operands during training, and splits all residual occurrences into their live descendants. It updates affected pair, token and boundary counts before choosing the next merge. The event log retains merges, removals, and later reactivations in chronological order. The reference prototype uses strict `>`, including its self-pair convention, while the paper's pseudocode prints `≥`. The implementation follows the released trainer so 1.0 is a disabled control. The project reserves 256 fallback IDs and keeps selected initial units atomic. It uses deterministic pair ties and verifies that every event refers to symbols active at that time. During fitting it stops on vocabulary occupancy or exhaustion, with a bounded event cap to surface nontermination as a recoverable failure.

Parity was checked against the [pinned original character-level reference](https://github.com/pchizhov/picky_bpe/tree/2f6018ae9789f701e474d884a87943f1c9b0bfc1/tests/reference). Two fixtures matched chronological events, final active vocabulary, and nine inference strings. One fixture exercised residual split-back and the next frequency decision. One exercised exact threshold equality. Independent tests cover nonoverlapping self-pairs, nested splits, reactivation, malformed dependencies, protected atoms, and checkpoint recovery. A resumed fit and uninterrupted fit produced identical model bytes. The custom `picky-bpe.json` uses the project encoder. The runner and snapshot inventory use the existing plugin contract and artifact list. Stock Hugging Face BPE does not express the interleaved event sequence. [Picky BPE publication](https://aclanthology.org/2024.emnlp-main.925/)

For T6 and T7, confidence changes eligible merge ranking through fixed-point weighted crossing counts. Only independent, hash-pinned, authorized surface splits aligned with the Malayalam reference contribute. A backed-off scalar boundary inside an akshara cannot authorize a morphological split. Weights stay at absolute surface offsets and become active again if a removal exposes an originally reviewed boundary. The scorer excludes overlapping self-pair cuts that the leftmost merge does not cross. Exact, supplied sandhi realizations can be projected when licensed by the existing conservative engine. Conflicting readings and interior rewrites abstain. Zero weight with the same annotations preserves merge events and IDs. No independently adjudicated training panel was available. T6 and T7 ran on explicitly synthetic regression fixtures only.

Picky refinement is established prior work. MorphBPE reports training constraints over morpheme boundaries, while this experiment applies soft penalties from reviewed Malayalam surface offsets. Recent Indic studies include vocabulary-constrained grapheme training and multilingual work already evaluates grapheme-constrained subwords. The research question here is whether a Malayalam akshara inventory with reversible scalar backoff improves the observed compression, vocabulary and linguistic trade-off under equal inputs and capacity. Combining existing techniques does not establish novelty or downstream quality. [MorphBPE](https://aclanthology.org/2026.findings-acl.2068/) · [Indic grapheme-constrained tokenizers](https://aclanthology.org/2026.findings-emnlp.1093/) · [StructPiece](https://aclanthology.org/2026.emnlp-main.972/)

## Measured CPU results

Training-only 48K probes exhausted their eligible pairs at **8,705 / 8,951 / 8,934 / 8,762 / 7,785** occupied slots for T1 through T5. The smallest observed ceiling, rounded down to a multiple of 64, set a common exploratory budget of **7,744**. Every fitted arm reached that budget. The published T0 retained all 48,000 slots. All rows below use the same 160,695 validation bytes and 5,388 whitespace-delimited canonical words.

| Arm | Occupied | Tokens | Fertility | UTF-8 bytes/token | Byte IDs | Standalone lexical IDs used in validation | Canonical round-trip, validation / stress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T0 | 48,000 | 11,566 | 2.147 | 13.894 | 0 (0%) | 4,722 / 47,737 | 11/12 · 7/11 |
| T1 | 7,744 | 19,716 | 3.659 | 8.151 | 14 (0.071%) | 3,210 / 7,481 | 12/12 · 11/11 |
| T2 | 7,744 | 21,122 | 3.920 | 7.608 | 1,847 (8.74%) | 3,129 / 7,481 | 12/12 · 11/11 |
| T3 | 7,744 | 21,067 | 3.910 | 7.628 | 1,796 (8.53%) | 3,137 / 7,481 | 12/12 · 11/11 |
| T4 | 7,744 | 19,730 | 3.662 | 8.145 | 14 (0.071%) | 3,193 / 7,481 | 12/12 · 11/11 |
| T5 | 7,744 | 19,620 | 3.641 | 8.190 | 14 (0.071%) | 3,117 / 7,481 | 12/12 · 11/11 |

All fitted arms returned the original text for 12/12 validation records and 9/11 stress probes. Two stress strings are intentionally canonicalized by the shared normalizer. T0 raw round-trip was 11/12 and 5/11. Its higher bytes per token are descriptive given its 48K size, historical training input, and fidelity failures.

| Arm | Akshara breaks / covered Malayalam token transitions | Grapheme breaks / covered token transitions | Excluded / all token transitions | Fit time / peak process RSS | Encoding KiB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| T0 | 87 / 1,831 | 155 / 3,215 | 8,339 / 11,554 | Historical artifact | 1,830 |
| T1 | 1,339 / 7,094 | 1,407 / 19,684 | 20 / 19,704 | 12.88 s / 284.6 MiB | 144 |
| T2 | 1 / 6,677 | 0 / 19,113 | 1,997 / 21,110 | 6.14 s / 283.0 MiB | 321 |
| T3 | 0 / 6,664 | 7 / 19,114 | 1,941 / 21,055 | 5.92 s / 284.8 MiB | 275 |
| T4 | 571 / 7,171 | 580 / 19,698 | 20 / 19,718 | 7.29 s / 283.7 MiB | 255 |
| T5 | 490 / 7,059 | 502 / 19,588 | 20 / 19,608 | 6.79 s / 283.4 MiB | 352 |

A covered transition requires two contiguous exact lexical offsets. Byte fallback, special IDs and uncertain historical offsets are excluded. The independent legacy akshara grammar counted one break for T3 and 491 for T5. All seven T3 grapheme exceptions occur just before a joiner, four before ZWNJ and three before ZWJ. The only T0 validation mismatch first diverged at U+200B. Reference differences and the large variation in offset coverage rule out interpreting these as one shared transition population. Every table comparison uses the same input bytes.

T4 cut byte fallback sharply compared with T3, and it retained fewer observed orthographic breaks than T1. It required 14 more tokens than T1 and allowed more observed breaks than T3. T5 used **110 fewer validation tokens than T4, a 0.56% reduction**, with the same 14 byte IDs. Ten articles used fewer tokens, one used more, and one tied. Training refinement removed 679 lexical operands. The final vocabulary had 55 lexical items never emitted independently on training text, compared with 605 for T4. Training singleton items rose from 465 to 651, and validation observed fewer distinct lexical IDs than for T4. Vocabulary coverage across other domains and model performance remain open.

**Shortlist:** T5 is the lead hybrid for a larger 48K development replication. Keep T3 as a structure-first comparator, T1 as the matched codepoint control, and T4 for the paired Picky ablation. T2 remains valuable when a strict extended-grapheme guarantee is the research priority. Keep T6/T7 out of empirical ranking until adjudicated annotations are available. The measurements support a small exploratory improvement over the matched hybrid control and do not establish superiority over the historical baseline or any BPB improvement.

## Reproduction and controlled execution

From a clean research checkout, reconstruct the development input with the pinned source receipt. A new render has to match the archived SHA-256 even when a pinned article has externally updated transclusions. Keep any replacement capture under a new identity.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '2'
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$run = 'experiments/mom_bpe/runs/performance-replay-v1'
$pins = 'experiments/mom_bpe/performance-results-v1.json'
python -m experiments.mom_bpe.development_corpus --fetch --pins $pins --capture "$run/capture" --out "$run/development"
foreach ($arm in @('T1', 'T2', 'T3', 'T4', 'T5')) {
    python -m experiments.mom_bpe.performance --input "$run/development/manifest.json" --out "$run/probe48k/$arm" --arm $arm --vocab 48000 --fit-only
}
foreach ($arm in @('T0', 'T1', 'T2', 'T3', 'T4', 'T5')) {
    python -m experiments.mom_bpe.performance --input "$run/development/manifest.json" --out "$run/matched7744/$arm" --arm $arm --vocab 7744
}
```

The same arm command resumes archived events or reuses a hash-verified completed result. Training-only 48K probes determine the attainable common exploratory size, without consulting validation. T0 always loads the published 48K artifact. CPU throughput is one pass on this workstation and includes normalization inside the encoder. The source manifest, preprocessing, algorithm, canonical bytes, corpus and model identity are tied to each fit. Use the metadata JSON to compare hashes and supporting measurements.

For the exploratory exposure rehearsal, rerun T1, T3, T4 and T5 with `--predicted-tokens 30000`. The four arms share 614,916 canonical source bytes, which produce 65,239 / 65,386 / 65,343 / 63,686 predicted tokens with EOS. The fixed-token budget covers 281,331 / 281,331 / 281,331 / 289,454 complete-document canonical bytes and 2,475 / 2,202 / 1,963 / 927 additional partial-document bytes. These ledgers record input exposure without starting a language model.

After scientific approval, the following commands check resources and prepare a bounded, frozen DGX CPU comparison. Supply an independently approved manifest and a new private scratch directory. Fit-only checks should precede any opening of final evaluation metrics.

```bash
nvidia-smi --query-gpu=index,uuid,memory.total,memory.free,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
python -c 'import sys; from experiments.mom_bpe.verify_frozen import verify; print(verify(sys.argv[1], require_morphology=False))' "$SIGNED_MANIFEST"
for arm in T1 T3 T4 T5; do
  /usr/bin/time -v python -m experiments.mom_bpe.performance --mode frozen --input "$SIGNED_MANIFEST" --out "$DGX_SCRATCH/$arm" --arm "$arm" --vocab 48000 --fit-only
done
```

After treatment selection and review, run the same arm command with `--predicted-tokens "$COMMON_TARGET"` in place of `--fit-only`. This records full-document and partial exposure under both fixed-source-byte and fixed-training-token budgets. The scorer in `mltok/lib/canonical_bpb.py` measures summed held-out bits per canonical UTF-8 byte, resets context per document, and predicts every text token plus one EOS with BOS provided. The older continuous-stream proxy training path needs a matched document-local input and scoring integration before a thesis BPB comparison. Any GPU activation requires a fresh availability check, explicit authorization, and a verified physical device index other than 7 before masking by UUID.

The bounded laptop runs used approximately 284 MiB per tokenizer fit. The public renders, models, checkpoints, manifests and measurements occupy about 17 MiB locally. For the first larger DGX CPU capacity run, reserve four CPU threads, 8 GiB host RAM and at least 10 GiB scratch as planning headroom, then measure word-type growth, memory and checkpoint size before scale-up. The existing tiny proxy dimensions would allocate 18.43M tied token-embedding parameters at 48K and about 11M nonembedding parameters. Activation and logit memory still depend on context, precision and microbatch. Measure them on an approved device before fixing a downstream batch or GPU reservation.

For a separate 32K/48K/64K follow-up, first finalize the leading algorithm at 48K on an adequately occupied, shared development input, then vary vocabulary only. Keep train and validation IDs, preprocessing, algorithm parameters, special IDs and fallback policy fixed. Record underfilled runs and measure vocabulary usage, orthography, compression, embedding and nonembedding parameters, runtime, memory and downstream paired BPB under each exposure budget. Keep the scientific held-out set sealed until this protocol and selection are fixed.

The offline CPU regression passed 241 tests, with five proxy-training cases deselected, zero failures, zero skips and two protobuf deprecation warnings in one existing SentencePiece contract test. The study produced zero DGX or language-model runs.
