# Boundary-guarded Malayalam tokenizer optimization

**T8 reduced observed lexical akshara breaks by 159 (22.0%) with 32 additional validation tokens (0.092%) relative to T5.** Both used 20 fallback byte IDs at the matched 7,744-slot budget. T5 scored lower bits per byte under both budgets in a trained single-seed CPU proxy. Retain T5 as the downstream lead and T8 as the orthographic comparator. [Measured results, split provenance and receipts](optimization-results-v1.json)

## Development corpus

The experiment extends the attributed Wikipedia development split from `cb5c39aba71bce1f0ce7560d06d27be84ef653f6`. It keeps the existing article assignments. The added Wikisource capture contains 32 pinned chapter or section revisions totaling 1,691,575 rendered bytes. The preparation retained 27 excerpts after language, script, Unicode and minimum-length screening. Indulekha, Kundalatha and Kerala Panineeyam contributed training text. Marthandavarma supplied a previously untrained literary work for validation. The capture records each revision, upstream source hash, rendered SHA-256, author, attribution link and page history. Visible quotations, references, media and other marked non-prose were excluded. Both Wikimedia projects declare CC BY-SA 4.0 for text subject to their applicable exceptions. The project stores source renders and text locally and publishes only provenance and measurements. [Wikimedia text terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content) · [Indulekha](https://ml.wikisource.org/wiki/ഇന്ദുലേഖ) · [Kundalatha](https://ml.wikisource.org/wiki/കുന്ദലതാ) · [Kerala Panineeyam](https://ml.wikisource.org/wiki/കേരളപാണിനീയം) · [Marthandavarma](https://ml.wikisource.org/wiki/മാർത്താണ്ഡവർമ്മ)

| Development split | Wikipedia | Wikisource | Canonical UTF-8 bytes |
| --- | ---: | ---: | ---: |
| Train | 47 articles, 614,916 bytes | 16 literary and 3 grammar excerpts, 289,930 bytes | 904,846 |
| Validation | 12 articles, 160,695 bytes | 8 literary excerpts, 125,205 bytes | 285,900 |

The previous Wikipedia language and Unicode audit was reused for those exact records. The same pinned fastText model independently screened the new material. Its minimum Malayalam confidence was 0.9976 in accepted Wikisource training excerpts and 0.9989 in validation excerpts. Script coverage is recorded independently of model confidence. The filter permits mixed script and retains names, loanwords and historical letters. The common conservative normalizer is unchanged. Exact raw and canonical SHA-256, page ancestry and NFC-whitespace five-gram Jaccard at 0.85 found no duplicate witnesses among the retained records. Book families stayed within their assigned split. The 2 MiB per-split source cap, request limits and page-level atomic capture allow a bounded replay. The new sample covers encyclopedic, historical literary and grammar text, with validation in the first two domains. Contemporary news, technical, regional conversation and systematic Malayalam–English coverage remain gaps.

| Candidate source | Verified bounded inventory | Disposition |
| --- | --- | --- |
| Sangraha Verified Malayalam | `ai4bharat/sangraha` at `8b813c3f62d37b2fa174d68c31e8b35ae2fe85e8`, native `verified/mal` Parquet, `doc_id,text,type`, CC BY 4.0 at the collection level | The prior 72 sampled records remain suitable for quality diagnosis. The sampled fields lacked original page URLs and sufficient rights evidence for direct text inclusion. |
| FineWeb2 Malayalam | `HuggingFaceFW/fineweb-2` at `af9c13333eb981300149d5ca60a8e9d659b276b9`, mirrored Parquet `a092823f5cba550080cb8e56451fc9ac5def625c`, `mal_Mlym/train`, `text,id,url` plus crawl and language metadata, ODC-By 1.0 | The prior 64 sampled rows exposed original URLs, including two Malayalam Wikipedia URLs. Collection licensing covers database rights. The development corpus uses pinned first-party Wikimedia renders. |
| Malayalam Wikipedia | 59 existing articles pinned by page ID, revision, source SHA, render SHA and excerpt SHA | Included with attribution and the previous CC BY-SA 4.0 screening and separation. |

The public samples required no private Hub access or complete release download. Obtaining a balanced modern-text mixture still depends on source or rights-holder permission for selected pages and an overlap review. The historical 22,120-document benchmark remains unrecovered. The development manifest has no independent sign-offs and cannot pass the final freeze gate. [Prior corpus access and source study](CORPUS_STUDY.md) · [Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) · [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) · [ODC-By terms](https://opendatacommons.org/licenses/by/1-0/) · [Scientific freeze protocol](../../audit/48k-baseline/evaluation-protocol-v1.json)

## What caused the previous breaks

The original model bytes and validation input reproduced the 571 T4 and 490 T5 lexical breaks. The byte audit verifies raw UTF-8 IDs against their canonical spans, then distinguishes cuts between Unicode scalars from byte continuation inside a scalar. Each reported offset refers to the exact canonical text. Bad or discontinuous offset records raise an error. The existing lexical coverage metric remains intact.

| Pattern or coverage on the original 12 articles | T4 | T5 |
| --- | ---: | ---: |
| Virama or conjunct, primary lexical class | 368 | 240 |
| Dependent vowel sign, primary lexical class | 183 | 222 |
| Other combining mark, primary lexical class | 20 | 28 |
| Chillu-adjacent, overlapping context class | 3 | 3 |
| Additional lexical cuts beside ZWJ / ZWNJ | 2 / 0 | 4 / 0 |
| Byte-adjacent intra-akshara scalar cuts | 1 | 1 |
| UTF-8 continuation transitions inside a scalar | 8 | 8 |
| Original covered Malayalam lexical transitions | 7,171 | 7,059 |
| Actual pretokenizer merge-domain cuts inside an akshara | 0 | 0 |

Every counted hybrid violation lay in an unprotected akshara decomposed to scalars. The one byte-adjacent break for each hybrid contained an unavailable initial scalar. The other breaks had lexical offsets. Strict akshara T3 had zero legacy lexical breaks on the same evaluation input, but its 1,796 fallback IDs yielded 450 verified byte-adjacent scalar cuts inside reference aksharas. Its 1,196 UTF-8 continuation transitions were counted separately. A zero in the older lexical table applies only to its measured coverage.

| Original arm and document | Pattern | Adjacent codepoints | Canonical codepoint offset | Canonical UTF-8 byte offset |
| --- | --- | --- | ---: | ---: |
| T4, `mlwiki:9830:4109090` | Before virama | U+0D28 / U+0D4D | 506 | 1,326 |
| T5, `mlwiki:9830:4109090` | Before long vowel sign | U+0D39 / U+0D3E | 410 | 1,060 |
| T5, `mlwiki:9830:4109090` | Before anusvara | U+0D27 / U+0D02 | 4,878 | 13,496 |
| T5, `mlwiki:6144:4655103` | Virama before ZWJ | U+0D4D / U+200D | 5,649 | 15,487 |

## Guard design and matched results

`boundary_guarded_hybrid` retains T5's hybrid initial alphabet, scalar backoff, Picky threshold, seven special IDs and 256 byte IDs. It increases the rank of actual joins across internal reference offsets in an unprotected akshara. At strength 1000 the eligible internal occurrence contributes one additional unit of pair frequency. It adds no bonus at exterior seams or across byte barriers. Absolute offsets keep the same signal after split-back. This is a soft guard and can leave unresolved internal cuts. Setting the guard to zero reproduced the entire real-corpus T5 event sequence and its encoder specification. The four published control fingerprints were verified unchanged. No legacy implementation or artifact schema files were edited.

| Arm | Slots occupied | Validation tokens | Fertility | Canonical bytes/token | Fallback IDs | Validation lexical IDs used | Unused training lexical IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T1 codepoint | 7,744 | 34,964 | 3.740 | 8.177 | 20 | 4,109 | 375 |
| T3 strict akshara | 7,744 | 36,990 | 3.957 | 7.729 | 2,542 | 4,078 | 353 |
| T4 hybrid | 7,744 | 35,122 | 3.757 | 8.140 | 20 | 4,134 | 421 |
| T5 hybrid + Picky | 7,744 | 34,810 | 3.724 | 8.213 | 20 | 4,140 | 25 |
| T8 guarded hybrid + Picky | 7,744 | 34,842 | 3.727 | 8.206 | 20 | 4,142 | 25 |

| Arm | Lexical akshara breaks / covered Malayalam transitions | Grapheme breaks / all covered lexical transitions | Excluded lexical transitions | Additional joiner-context lexical cuts | Byte-adjacent akshara cuts |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 | 2,065 / 13,427 | 2,154 / 34,916 | 28 | 52 | 0 |
| T3 | 0 / 12,898 | 13 / 34,220 | 2,750 | 0 | 631 |
| T4 | 905 / 13,649 | 929 / 35,074 | 28 | 7 | 0 |
| T5 | 722 / 13,300 | 781 / 34,762 | 28 | 36 | 0 |
| T8 | 563 / 13,366 | 589 / 34,794 | 28 | 8 | 0 |

On Wikipedia, T8 changed the T5 totals from 19,585 to 19,619 tokens and from 442 to 342 lexical akshara breaks. On Wikisource, it changed 15,225 to 15,223 tokens and 280 to 221 breaks. Each source kept the same fallback count in both arms, 7 for Wikipedia and 13 for Wikisource. Both tables report coverage because different segmentations expose different transitions. The evidence also records all per-source and per-domain fidelity, occupancy, memory and throughput measurements.

All five fitted arms passed 20/20 exact raw and canonical validation records. On the 11 orthography probes, all matched 11/11 canonical and 9/11 raw. The normalizer accounts for those two raw differences. Across the five repeats with the published control fingerprints, fits took 2.14–5.79 seconds and 283–292 MiB peak RSS. One-pass encoding ranged from about 472 to 1,261 KiB/s. Single-process wall times and encoding throughput vary by load. The input stream included the same 904,846 training bytes for every arm.

Capped training-only 16,384-slot probes ran out of eligible pairs at 10,328 occupied entries for T5 and 10,312 for T8. At those separate capacity endpoints, T5 used 33,959 tokens and had 653 breaks across 12,459 covered Malayalam transitions. T8 used 34,017 tokens and had 585 breaks across 12,551 covered transitions. Each still emitted 20 byte IDs. Relative to its own 7,744-slot run, T8 compressed more while exposing more observed akshara breaks. These are within-arm capacity diagnostics at different sizes. 48K fitting was deferred.

StructPiece enforces grapheme boundaries and uses corpus-induced affix signals. Indic grapheme-constrained studies investigate vocabulary constraints with fixed grapheme units. MorphBPE uses morpheme boundaries during training. T8 targets internal Malayalam orthographic sites exposed by scalar backoff, with no morphological labels or inference-time boundary rewrite. This is a measured adaptation of established ideas. Its scientific contribution depends on a sufficiently powered Malayalam comparison. [StructPiece](https://aclanthology.org/2026.emnlp-main.972/) · [Indic grapheme tokenizers](https://aclanthology.org/2026.findings-emnlp.1093/) · [MorphBPE](https://aclanthology.org/2026.findings-acl.2068/) · [Unicode segmentation](https://www.unicode.org/reports/tr29/)

## Executed CPU proxy

The bounded experiment fit one 1-layer GPT per arm and per budget from seed 17, with width 16, 16-token context, microbatch 32, AdamW at 0.003, and 128,304 parameters including 123,904 tied embedding parameters. Both arms used exactly the same two validation excerpts totaling 2,397 canonical UTF-8 bytes. The scorer reset context per document, used BOS as context, scored EOS once per complete excerpt and validated right-padding masks. It sums target NLL before dividing by the same canonical bytes. All four fits backed up local optimizer, model, progress and CPU RNG generations. Synthetic regression verified exact resumed model parameters and training-position counts.

| Budget | T5 BPB | T8 BPB | T8 − T5 | Matched exposure and compute |
| --- | ---: | ---: | ---: | --- |
| Fixed source bytes | 1.260310 | 1.274335 | +0.014025 | 7,191 bytes, 747 predicted tokens, 24 updates and 11,952 padded training input positions each |
| Fixed training tokens | 1.293476 | 1.314152 | +0.020676 | 512 predicted tokens, 16 updates and 8,192 padded training input positions each |

Under the token budget, each arm consumed 3,598 complete-document bytes plus 1,441 partial bytes for T5 and 1,442 for T8. Evaluation needed 252 predictions for T5 and 256 for T8, including both EOS targets. Training wall time was under one second per fit and peak process memory was under 337 MiB. The result validates the local protocol and records a negative result for T8. One seed, two excerpts and a 16-token context provide no basis for a thesis BPB conclusion. The next downstream comparison should retain T5, T8 and the matched codepoint control, with T3 as a diagnostic for fallback and coverage. The historical T0 result still has different training and evaluation conditions.

## Reproduce or prepare the next approved run

Use Python 3.12, the repository-pinned LID model and the matching tokenizer and Torch runtimes. Use independent ignored output directories. The previous Wikipedia split can be recreated from [its capture receipt](performance-results-v1.json), and the new Wikisource source revisions and hash pins are in the optimization evidence. A changed upstream render requires an independently named capture.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '2'
$root = 'experiments/mom_bpe/runs'
$study = "$root/boundary-optimization-replay-v1"
$env:PYTHONPATH = (Resolve-Path "$root/corpus-study-v1/lid/installed").Path
python -m experiments.mom_bpe.optimization_corpus --fetch --pins experiments/mom_bpe/optimization-results-v1.json --previous "$root/performance-v1/development/manifest.json" --capture "$study/wikisource" --lid "$root/corpus-study-v1/lid/resources.json" --out "$study/development"
foreach ($arm in @('T1','T3','T4','T5','T8')) {
    python -m experiments.mom_bpe.optimization --manifest "$study/development/manifest.json" --out "$study/matched7744/$arm" --arm $arm
}
python -m experiments.mom_bpe.optimization_proxy prepare --manifest "$study/development/manifest.json" --out "$study/cpu"
foreach ($arm in @('T5','T8')) {
    foreach ($budget in @('fixed_source_bytes','fixed_training_tokens')) {
        python -m experiments.mom_bpe.optimization_proxy run --plan "$study/cpu/plan.json" --tokenizer "$study/matched7744/$arm" --out "$study/cpu/$arm/$budget" --budget $budget
    }
}
```

For a future approved DGX tokenizer job, verify the 48K frozen manifest and start with a per-arm CPU fit-only capacity probe. Set scratch, verify the source budget and available storage, and preserve all immutable checkpoint generations until recovered artifacts and their backups have been tested. A starting reservation of four CPU threads, 8 GiB RAM and 10 GiB private scratch is planning headroom for a bounded slice. Measure peak RSS, actual checkpoint and vocabulary growth and run duration before sizing a larger job. The laptop had approximately 262 GiB free at the environment check. Its CPU pilot needed no GPU authorization.

```bash
nvidia-smi --query-gpu=index,uuid,memory.total,memory.free,utilization.gpu --format=csv
df -h "$DGX_SCRATCH"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
python -c 'import sys; from experiments.mom_bpe.verify_frozen import verify; print(verify(sys.argv[1], require_morphology=False))' "$SIGNED_MANIFEST"
for arm in T1 T3 T5 T8; do
  /usr/bin/time -v python -m experiments.mom_bpe.optimization --frozen --fit-only --manifest "$SIGNED_MANIFEST" --vocab 48000 --arm "$arm" --out "$DGX_SCRATCH/$arm"
done
```

After the approved fits and treatment selection, rerun the same command without `--fit-only` to score the frozen tokenizer validation. The downstream GPU schedule still needs independent evaluation approval, a sufficiently occupied 48K training slice, a locked compute and context plan, and explicit GPU authorization. Check availability and use a verified physical UUID other than GPU 7. No DGX or large job ran in this milestone. Historical artifacts and earlier Hugging Face backups were preserved. New raw text, models and checkpoints remain in the ignored local run tree. The published backup contains code, source metadata and aggregate evidence.

The offline CPU regression suite passed 255 tests, with zero failures, zero skips, five legacy continuous-stream proxy tests deselected and two protobuf deprecation warnings in one existing SentencePiece contract test. The new tests exercise the guard-off control, protected conjuncts and signs, chillu and joiner cases, verified byte-backed offsets, checkpoint restoration, right-padding invariance and document-local BOS/EOS.
