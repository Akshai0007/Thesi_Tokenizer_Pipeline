# Malayalam 48K evaluation foundation

The published 48K tokenizer remains fixed at revision `cff0d31c84b598436087f21cc8183021993831e1`. Its `tokenizer.json` SHA-256 is `2f4b7e8b0706007f7ab3f0b8b2cc1b169094072c334585dabc4081d455e69ff5`. The original audit remains available alongside this addendum. New measurements use explicitly authored CPU diagnostics. [Original audit](REPORT.md) · [Baseline manifest](baseline-artifacts.json)

## Historical evaluation recovery

The archived DGX selection records 22,120 evaluation documents, 4,602,360 whitespace-delimited words, 122,510,792 UTF-8 bytes and the legacy fingerprint `c1321250cd78`. Local experiment paths, dataset caches, Git config history, the pinned historical archive and its inspected earlier revisions yielded no original evaluation shards or ordered document hashes. The archived tokenizer itself matches the published baseline by SHA-256.

An authenticated, metadata-only inspection located a private raw backup with three gzip streams covering FineWeb-2, Sangraha and Wikipedia. The streams total **7,518,303,142 compressed bytes**. Repository revision, blob identities and LFS SHA-256 values were recorded locally. The archived tuned YAML was inspected at a pinned revision. The scan downloaded **zero corpus bytes**. It did not verify those blob bytes or establish that the backed-up streams were the exact inputs consumed by DGX.

The archived configuration limits FineWeb-2 `mal_Mlym` to 1,000,000 documents and uses Wikimedia `20231101.ml` and Sangraha `verified/mal`. It sets length bounds of 80 and 300,000 codepoints, minimum Malayalam ratio 0.5, line repetition filtering, an evaluation split of 3 per 1,000, a nominal 64 MB shard setting, an 8,000 MB training cap, and source weights of 0.4/0.3/0.3 for FineWeb-2, Sangraha and Wikipedia. None of the archived source specifications identifies an immutable dataset revision.

The inspectable ingestion code canonicalizes line endings, strips its configured transport controls and trims text before filtering. It applies the configured filters in order, deduplicates cleaned text by BLAKE2b-96 and assigns evaluation when `int.from_bytes(digest[:4], "big") % 1000 < eval_permille`. The first accepted copy determines source attribution. Source scheduling follows accepted training bytes and continues consuming source data after the training cap to collect evaluation documents. Input order, exact reader and filter revisions, source limits, ties, duplicate attribution, and gzip/shard behavior all matter when replaying.

If original DGX shards or a trusted content manifest become available, compare ordered document SHA-256 values, raw text bytes, source attribution, per-source counts, totals, and shard byte hashes at the recorded revision. A replay from the raw backup is a candidate reconstruction. Matching the archived totals and basename/size fingerprint alone leaves content identity unverified. Exact recovery was not established in this milestone. Preserve the historical results as archived and use a separately identified corpus for new measurements. [Proposed evaluation protocol](evaluation-protocol-v1.json)

## Boundary measurement

`boundary_offsets_v2` is an opt-in metric with separate akshara and Unicode extended grapheme cluster (EGC) references. It consumes adjacent spans from the original encoding and requires contiguous, positive spans at valid scalar boundaries. It excludes transitions adjacent to registered special IDs, byte fallback when declared by the backend, invalid or overlapping spans, and sites without Malayalam context on both sides. Joiners alone cannot create Malayalam context. The historical `structure` and diagnostic `boundary_audit` implementations retain their existing definitions.

| Authored panel | Input view | Eligible | Akshara breaks | EGC breaks | Reference disagreements |
| --- | --- | ---: | ---: | ---: | ---: |
| 54 ordinary documents | Raw | 133 | 6 | 6 | 0 |
| 54 ordinary documents | Lexical canonical | 133 | 6 | 6 | 0 |
| 26 Unicode stress probes | Raw | 26 | 11 | 13 | 2 |
| 26 Unicode stress probes | Lexical canonical | 31 | 10 | 12 | 2 |

On the ordinary panel there were 555 adjacent pairs, including 24 excluded for byte fallback and 398 outside Malayalam context. Each of the six accepted violations is after virama and inside a pretokenized piece. The respective adjacent pair has neither a joined vocabulary entry nor a direct merge rule. These are genuine output splits under both references. The earlier reconstructed-surface diagnostic counted eight breaks in 135 transitions because it omitted emoji fallback pieces and joined text across the omitted regions. Explicit UTF-8 conversion gave the same counts as codepoint offsets in all four views. The offset metric cannot estimate the bias of the archived 6.9482% result without the historical inputs and metric identity.

The boundary rates are conditional on eligible transitions. For controlled comparisons also report violations per 1,000 fixed, reviewed reference units, eligible-event coverage and exclusion counts. Treat the 25 mark-start vocabulary pieces, five consecutive-vowel-sign flags and two repeated-virama flags as corpus-investigation leads. Their prevalence and causes in training text remain unknown. SentencePiece and count-only Hub backends have no supported native offsets in this metric. [Exact-offset and fidelity evidence](foundation-evidence.json)

## Unicode fidelity and user text

The lexical canonical target consists of NFC and the pinned tokenizer's declared chillu and invisible-character transformations, excluding marker transport. Record the exact input view with each score. Use `decode(encode(x)) == x` for raw fidelity and `decode(encode(x)) == canonical(x)` for canonical fidelity. Combining equivalents and legacy chillu can pass the canonical check while changing raw codepoints. The baseline ordinary probe, explicit ZWNJ, whitespace probe and empty string passed both comparisons.

| Probe | Pinned tokenizer behavior | Policy for future variants |
| --- | --- | --- |
| Literal `▁` in ordinary text | `ക▁ക` becomes `ക ക` | Preserve the literal glyph with reversible marker transport or a reversible representation. |
| User text with inline chat spelling | `മലയാളം<\|im_start\|>വാക്ക്` gains a space after the spelling | Treat user text as text. A trusted caller supplies conversation controls through a separate interface. |
| User text with padding spelling | `ക<pad>ക` becomes `ക<pad> ക` | Apply the same text/control separation to every registered control spelling. |
| Caller-inserted delimiter | The delimiter alone returns the original spelling and has its registered ID | Verify trusted control IDs independently from text round-trip. |

The inline spellings reach registered special IDs even with `add_special_tokens=False`. Evaluation therefore checks decoded text and control-ID provenance separately. Preserving both in future variants requires explicit text/control handling and reversible marker transport. Until then, the baseline cannot promise exact behavior for arbitrary literal marker characters or strings resembling controls. Eleven probes preserve the current behavior as evidence. No tokenizer code or exported artifacts changed. [Fidelity fixtures](fidelity-cases.json) · [Observed outputs](foundation-evidence.json)

## Morphology annotation

The earlier audit scored 16 proposed boundaries in 13 authored forms and observed one exact token-boundary match. All 16 offsets are carried forward as proposals. Five earlier sandhi or gemination forms lack a simple surface alignment. The expanded queue contains **96 forms** including all 24 prior forms and 72 new seeds, six per requested category. The pinned Mlmorph/SFST combination returned suggestions for 84 forms and multiple analyses for 52. The queue prioritizes five previously unresolved alignments, 49 other forms with multiple candidates and 11 forms with no candidate. Another 31 forms await review with one candidate. There are **zero independently verified forms or boundaries**.

1. Two fluent Malayalam reviewers independently supply a context or gloss and verify the form, category and family. Keep them blind to tokenizer outputs and to each other's decisions. Hide the analyzer and historical suggestion columns until each reviewer has recorded an initial judgment.
2. Annotate surface segments on the supplied string. Record internal zero-based codepoint boundaries for raw and NFC views separately. Record `[]` for an accepted form with no internal boundary. An empty field means unreviewed. For a proposed segmentation the indicated substrings must rejoin to the exact surface in that view.
3. Record lemmas, morphosyntactic features and non-concatenative allomorphs in notes. Do not force offsets from an analyzer lemma, especially across sandhi, gemination or a chillu representation change. If two readings are valid, record context-conditioned analyses or mark the form unresolved. Record abstentions explicitly.
4. Adjudicate independently recorded disagreements, save dated reviewer and adjudication revisions, and report boundary agreement, exact-form agreement and coverage by category. Only accepted, adjudicated cases enter reference scoring. Keep ambiguous and abstained cases in coverage reporting. Expand reviewed forms to the protocol target and keep lemma families together when dividing tuning and held-out cases.

The TSV is an unadjudicated template. Make reviewer copies and publish adjudicated results under a new artifact identity. Mlmorph ranking weights are heuristic and supply neither calibrated confidence nor surface offsets. [Review sheet](morphology-annotation-v1.tsv) · [Queue summary](morphology-annotation-v1.json) · [Authored seeds](morphology-candidates.json)

## Independent freeze and controlled 48K comparison

The proposed freeze has 2,400 natural documents across six domains, a separate 240-case orthography panel and 240 reviewed morphological forms from at least 80 groups. Obtain verified source permissions, stable IDs and pinned revisions. Screen duplicate groups and train/eval overlap, rank eligible groups with the protocol's stable hash, and verify source and shard identities independently before naming the final dataset. Preserve exact raw bytes alongside the frozen canonical view and report unresolved quota shortages. These counts are targets. No natural evaluation corpus or adjudicated benchmark has been frozen here. [Freeze and budgeting protocol](evaluation-protocol-v1.json)

| Arm | 48K tokenizer |
| --- | --- |
| A | Immutable published baseline for descriptive evaluation |
| A0 | Matched codepoint MLBPE control with the common fidelity policy |
| B | Extended-grapheme initial units and declared fallback |
| C | B with Malayalam-specific orthographic protection |
| D0 | Same C and morphology input path as D, with all morphology weights zero |
| D | C with validated, confidence-weighted and surface-aligned constraints |

Measure B–A0 for grapheme initialization, C–B for the Malayalam rule and D–D0 for morphology. Confirm D0 reproduces C on controlled input before using it. All newly trained arms share canonical normalization, the special-token contract, training data order and vocabulary budget. Use two budgets for each matched comparison. The fixed-source-byte budget exposes the same canonical input bytes. The fixed-training-token budget exposes the same number of predicted non-padding tokens and records complete and partial document coverage. Keep model architecture, token context, optimizer policy and paired seed schedule fixed. Record total, embedding and non-embedding parameters, canonical source-byte exposure, predicted tokens, steps, FLOP estimate, wall time, usable context bytes and the manifest for each arm. Compute BPB as total held-out bits divided by the same frozen canonical UTF-8 bytes, with document resets and a fixed BOS/EOS policy across arms.

Before using the held-out panels, ratify a minimum relevant change and sample/seed budgets using separate development data. Proposed starting thresholds are a 25% relative reduction in B's EGC break incidence and C's akshara break incidence per fixed reference units, and a gain of five absolute percentage points in D's reviewed boundary F1. Require all designated fidelity probes to pass and publish exclusions and coverage to expose apparent improvements caused by fewer measurable sites. Report paired estimates and uncertainty with document-family clusters and matched seeds. For adoption, require held-out BPB noninferiority in both budgets against the corresponding control using the proposed +0.005 BPB margin. Claim BPB improvement only when the paired interval supports lower BPB. Preregister the final thresholds, seed and interval procedures before unblinding.

Prioritize reversible text transport and trusted control separation, then test grapheme and Malayalam orthography constraints against the matched controls. Consider weighted morphology after independent adjudication with enough reviewed families. Flagged vocabulary patterns and the archived BPB are targets for investigation. No training, model comparison or linguistic improvement claim was made in this milestone.

## Reproduction and delivery

CPU verification uses the pinned baseline, the new metric and eleven explicit fidelity probes. The morphology builder uses the pinned isolated Mlmorph and SFST wheels. The scripts check the pinned baseline or package versions, record input fingerprints and refuse to replace existing reports whose bytes differ. The progress manifest identifies deliverables and their fingerprints. GitHub publication and private backup delivery can be tracked separately from the research evidence. [Checks](foundation_checks.py) · [Analyzer lock](morphology-dependencies.json) · [Progress](foundation-progress.json)
