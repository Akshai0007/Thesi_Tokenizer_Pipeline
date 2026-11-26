# Matched Malayalam 48K fits at larger local scales

This comparison uses the verified Wikipedia capacity corpus, ordered prefixes of whole documents, and a common 186-document development split of 999,950 canonical UTF-8 bytes. The original 8 MB tokenizers were rescored on those same documents. Their training split was checked for shared source groups, exact raw and canonical hashes, and near duplicates at the study's five-gram Jaccard threshold. The historical mlbpe-48K artifact is a descriptive reference alongside the new fits. Its training overlap with these documents is unknown. [Measured results, fingerprints and paired summaries](scale-next-evidence-v1.json) · [Prior manifests and source terms](SCALE_48K.md) · [Scientific evaluation protocol](../../audit/48k-baseline/evaluation-protocol-v1.json)

## Inputs and resource decisions

| Training input | Documents | Canonical bytes | Manifest SHA-256 |
| --- | ---: | ---: | --- |
| Earlier pilot | 1,872 | 7,999,910 | `49354dabf49531150935c5276a8cf0d4ae386c85667ba6fd725cbcd3368071e3` |
| Intermediate prefix | 6,400 | 31,988,482 | `698c26df7b80d99e4a039cfa687341805eb25a4add06f66310439ed42879b4ad` |
| Larger prefix | 13,669 | 63,993,376 | `10c7216d267d756b832431c11a04c6c238ab97fb49bbd710c10e0d9de16a99f3` |
| Full scale A | 19,405 | 99,999,974 | `d5875f855283a135b3c056e9bdc83a629b35f1e387be168211436b334f1ba8b8` |

The 32 MB hybrid profile had 294,062 distinct merge domains, 59,885 initial pair types and 1,581,981 initial pair-to-domain postings. At 64 MB the counts were 492,626, 75,841 and 2,707,748. At 100 MB they were 670,732, 85,992 and 3,741,493. These postings describe the initial index. The live pair index is updated throughout training.

The laptop preflight used the larger of the measured T5/T8 32 MB peaks, multiplied by the maximum growth in type count, pair postings and known-unit positions, then by 1.5. The resulting 64 MB budget was 3.08 GB of worker RAM. The 100 MB budget was 4.25 GB plus the 2 GiB system reserve, exceeding the 5.75 GB available at the first decision. After the intermediate runs and diagnostics exited, the minimum of 20 readings taken two seconds apart was 7.01 GB. That cleared the 6.40 GB combined requirement for starting the first full fit. Fits run one at a time, with active working-set, private commit and machine-headroom checks. The 335 MiB reported during the earlier 100 MB capacity profile measures profiling only.

The full T1 codepoint projection required 6.80 GB including the system reserve because its pair index is larger. Its initial and final sustained-memory checks reached minima of 6.71 and 6.37 GB. Its completed local result is the 63,993,376-byte fit. T3's full fit cleared its 5.61 GB requirement and completed.

The full scale-A DGX starting reservation is one CPU trainer, four allocated CPU threads, 32 GiB RAM and 100 GB of private scratch. Measure the first fit and revise before expanding. Provision at least 10–15 minutes of CPU wall time per arm as an initial planning allowance. Use the verified local input and confirm that its storage permissions cover the target private scratch location.

```bash
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
export PYTHONUTF8=1
# MANIFEST is the verified 99,999,974-byte scale-A manifest in approved private scratch.
sha256sum "$MANIFEST"
for arm in T5 T8 T1 T3; do
  python -m experiments.mom_bpe.scale_fit --manifest "$MANIFEST" \
    --out "$SCRATCH/mom-100mb-next-v1/$arm" --arm "$arm" || exit
done
python -m experiments.mom_bpe.scale_fit --manifest "$MANIFEST" \
  --out "$SCRATCH/mom-100mb-next-v1/T0" --arm T0
python -m experiments.mom_bpe.scale_mlbpe --manifest "$MANIFEST" \
  --out "$SCRATCH/mom-100mb-next-v1/M48"
python -m experiments.mom_bpe.scale_compare paired --manifest "$MANIFEST" \
  --reference "$SCRATCH/mom-100mb-next-v1/T5" --challenger "$SCRATCH/mom-100mb-next-v1/T8" \
  --out "$SCRATCH/mom-100mb-next-v1/paired-T8-T5"
```

The tokenizer driver reuses compatible chronological checkpoints and verifies any completed artifact before reuse. The legacy matched `scale_mlbpe` command preserves the original seeded algorithm, the common conservative normalizer, the `word` pretokenizer and the engine's metaspace marker. Metaspace can attach a word-start space to a lexical token. The event arms use script-run segmentation and literal whitespace. Consequently even the equal-data comparison retains a preprocessing difference. The legacy seeded trainer has no mid-fit recovery. Each native model export is verified and has its own completed-fit receipt.

## Measured performance

The paired T8-minus-T5 differences were −54 tokens at 32 MB and −37 at 64 MB. At 64 MB T8 used fewer tokens on 46 documents, more on 67 and the same on 73. The median change was zero and the range was −19 to +10 tokens. T8 reduced measured lexical akshara cuts from 867 to 420 and grapheme cuts from 918 to 471. Both arms used 62 byte-fallback IDs and preserved all 186 raw and canonical texts. The guard's orthographic effect is substantial in this sample, while its aggregate compression effect is small.

| Size | T5 tokens | T8 tokens | T5 / T8 fallback IDs | T5 / T8 lexical IDs used | T5 / T8 measured lexical akshara cuts |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7,999,910 bytes, rescored | 111,339 | 111,257 | 180 / 180 | 15,192 / 15,185 | 1,141 / 720 |
| 31,988,482 bytes | 107,611 | 107,557 | 114 / 114 | 16,488 / 16,453 | 841 / 411 |
| 63,993,376 bytes | 107,646 | 107,609 | 62 / 62 | 16,585 / 16,523 | 867 / 420 |
| 99,999,974 bytes | 107,662 | 107,627 | 59 / 59 | 16,557 / 16,490 | 886 / 407 |

The larger training slices reduced fallback. Lexical utilization rose from 8 MB to 64 MB and then fell slightly. Compression and boundary counts improved between 8 MB and 32 MB. Both token counts and measured lexical cut counts rose slightly from 32 MB to 64 MB. At 100 MB the token counts rose again, T5's cut count rose, and T8's cut count fell. The emitted-byte-backed cut counts for T5 and T8 stayed at zero on this split. Their 64 MB observed training peaks were 1,682 and 1,739 MiB with 110.28 and 122.38 seconds elapsed. The completed latest checkpoints reproduced the event lists in the exported models.

At the full 100 MB scale, T8 used 35 fewer tokens overall and reduced measured lexical cuts by 54.1%. It used fewer tokens on 54 articles, more on 58 and tied on 74. Its median change was zero and the range was −17 to +7 tokens. Both event arms reproduced every canonical and raw text. T5 retained 6,960,613 bytes and T8 6,941,582 bytes in 47 checkpoint generations each. Their largest individual checkpoints were 299,480 and 300,403 bytes. The full input train and development files occupy 114,315,900 bytes on disk. The full-scale fit directories add 32,113,658 bytes, including their tokenizer files and checkpoints. The latest T3, T5 and T8 checkpoints reproduced their exported event lists.

| Full-scale arm | Tokens | Bytes per token | Fallback IDs | Lexical akshara cuts | Grapheme cuts | Fit seconds | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T0 published, common validation | 82,227 | 12.1608 | 109 | 760 | 949 | — | — |
| M48 matched original MLBPE | 77,451 | 12.9107 | 74 | 559 | 615 | 92.12 | 1,299 |
| T3 strict akshara | 109,844 | 9.1034 | 678 | 0 | 23 | 124.28 | 2,034 |
| T5 hybrid + Picky | 107,662 | 9.2879 | 59 | 886 | 938 | 153.69 | 2,187 |
| T8 guarded hybrid + Picky | 107,627 | 9.2909 | 59 | 407 | 459 | 160.50 | 2,266 |

M48 filled 48,000 slots and reproduced all 186 documents after the 100 MB fit. It used 4,776 fewer tokens than the published tokenizer on this split, a 5.81% reduction, and 17 more tokens than the 64 MB M48 fit. It used fewer tokens than the published tokenizer on 173 articles, more on eight and the same on five. T3, T5 and T8 used more tokens than the published baseline on every article. The native control had the strongest measured compression among the tested configurations. Its training exposure and preprocessing limit algorithm-only attribution. T3 filled 48,000 slots and recorded 149 byte-backed intra-akshara cuts despite recording zero lexical akshara cuts.

The matched legacy MLBPE refit occupied 48,000 slots and used 77,434 tokens. Its paired difference from the published model was −4,793 tokens or 5.83%, with a median difference of −17 across the same development articles. It used fewer tokens on 175 articles, more on nine and tied on two. It reproduced all 186 texts. The published model used 82,227 tokens and reproduced 184. The two native models expose less exact offset coverage than the event encoders, which limits comparisons of their raw boundary counts. All compression gaps are specific to this development split and the specified training and preprocessing policies.

All new 64 MB fits occupied 48,000 slots. The table reports training time and fit process peak. Full metrics include fertility, throughput, fallback, raw and canonical fidelity, and covered and excluded transitions. The native tokenizers' byte-backed cut counts are unavailable under this offset policy.

| Arm | Tokens | Bytes per token | Fallback IDs | Lexical akshara cuts | Grapheme cuts | Byte-backed akshara cuts | Fit seconds | Peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T0 published | 82,227 | 12.1608 | 109 | 760 | 949 | — | — | — |
| M48 matched original MLBPE | 77,434 | 12.9136 | 77 | 536 | 590 | — | 75.92 | 1,052 |
| T1 codepoint | 107,953 | 9.2628 | 77 | 2,863 | 2,963 | 0 | 188.64 | 2,146 |
| T3 strict akshara | 109,605 | 9.1232 | 807 | 0 | 23 | 178 | 85.30 | 1,597 |
| T5 hybrid + Picky | 107,646 | 9.2892 | 62 | 867 | 918 | 0 | 110.28 | 1,682 |
| T8 guarded hybrid + Picky | 107,609 | 9.2924 | 62 | 420 | 471 | 0 | 122.38 | 1,739 |

## Development source review

The local metadata worklist contains 19,591 Wikimedia Wikipedia articles, 27 revision-pinned Wikisource records, 72 Sangraha candidates and 64 FineWeb2 candidates. Seven Sangraha and nine FineWeb2 records were screened out by the earlier survey. Sangraha declares CC BY 4.0 at collection level. Its sampled records have native identifiers and lack original source attribution. FineWeb2 declares ODC-By for the database, and its records have source links for page-level terms, attribution and crawl provenance checks. Wikimedia reuse follows the underlying licenses and attribution requirements for the source material. Wikisource's public-domain or compatible-license policy still requires work, edition and transcription evidence. The worklist records research use, redistribution and private storage as separate decisions. Each remains pending independent review. The 1.5 GB proposal still needs independently reviewed eligible bytes by domain under the existing corpus materializer. The final scientific holdout is outside these development runs. [Wikimedia text terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content) · [Malayalam Wikisource policy](https://ml.wikisource.org/wiki/Wikisource:Copyright_policy) · [Sangraha card](https://huggingface.co/datasets/ai4bharat/sangraha) · [FineWeb2 card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) · [Corpus review gates](CORPUS_STUDY.md)

## Downstream readiness

`scale_proxy_plan --verify-windows` prepared the 38,799,936-parameter, 512-context integration on T5, T8 and M48 trained at 100 MB. It checks the actual document-local target windows, counts EOS and right padding for both budgets, verifies the model parameter count without allocating weights, and exercises canonical-byte BPB arithmetic with one synthetic bit per target. The shared fixed-source budget selected eight whole documents and 99,993 canonical bytes. The fixed-token budget used exactly 8,192 targets, including EOS only where a complete document ended. M48's partial-document bytes also passed a decoded-prefix check. The checkpoint test verifies optimizer and CPU RNG recovery on a small training fixture. The proposed integration has not trained. The older continuous-stream proxy cannot provide this protocol. The 124M confirmation remains outside this run. [Document-local reference driver](optimization_proxy.py) · [Scoring and denominator](../../mltok/lib/canonical_bpb.py)

| Full-scale arm | Fixed-source targets | Whole documents at 8,192 targets | Partial canonical bytes | Validation targets |
| --- | ---: | ---: | ---: | ---: |
| T5 | 10,123 | 0 | 82,799 | 107,848 |
| T8 | 10,125 | 0 | 82,710 | 107,813 |
| M48 | 7,224 | 5 | 7,983 | 77,637 |

## Verification and storage

The final offline CPU regression suite passed 200 tests with two protobuf deprecation warnings. A focused rerun after the proxy preparation change passed 24 tests with zero warnings. Both runs kept Windows UTF-8 mode disabled and passed the two architecture scans. The scans now explicitly decode UTF-8 Python files. A longer Windows test scratch path caused failures in an earlier run. The final short-path run passed the same cases. Tokenizer files and chronological checkpoint generations remain in the ignored local run tree pending private-storage and redistribution review. The previously verified private snapshots retain their restore receipts. [Prior backups](ARTIFACT_INVENTORY_V1.md#private-backup-and-restore)
