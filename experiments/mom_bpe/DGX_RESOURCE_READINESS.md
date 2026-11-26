# DGX resource readiness and publication

## Six-file implementation review

| File | Review finding |
| --- | --- |
| `experiments/mom_bpe/dgx_resources.py` | Validates a full physical UUID and exactly one matching GPU. Physical index 7 is rejected before CUDA calls. Exclusive mode rejects other compute PIDs. Failed, ambiguous or unsupported process queries block admission. A utilization threshold above 5% requires sharing permission and the ceiling is 15%. Free memory and utilization are measured at admission time. |
| `experiments/mom_bpe/document_proxy.py` | Checks physical resources before CUDA setup and again inside the run lock immediately before training. The second check includes the physical index. The original BF16, one-device and deterministic CUBLAS checks remain active. The only additions inside the optimizer loop are observations after verified checkpoint publication. |
| `experiments/mom_bpe/dgx_handoff.py` | Forwards all three settings into the trainer preflight and checks A100 type. Updated preflight requires a source bundle containing the updated trainer. |
| `experiments/mom_bpe/DOCUMENT_PROXY.md` | Documents both policies, sample units, immutable attempt records and recovery limits. Earlier scientific milestone text reflects its original date. |
| `tests/test_dgx_resources.py` | Covers both CLIs, threshold boundaries, insufficient free memory, malformed process data, self and other PIDs, a changed second observation, and the exact digest allowed to reuse the historical scientific identity. |
| `tests/test_document_proxy.py` | Covers GPU 7 before CUDA access, checkpoint replay, and independent telemetry attempts across an interrupted CPU training fixture. |

Operational source hashes live in telemetry. The original checkpoint serializer, restore routine, experiment configuration, vocabulary, corpus, M48/F-space encoders, exposure clocks and canonical scorer are preserved. A sudden process loss can leave the latest telemetry attempt without its end record. Historical checkpoints have no retrospective allocator measurements. Admission is a point-in-time observation of compute processes reported by the host NVIDIA driver. Confirm allocation and PID visibility with the DGX operator, particularly inside containers.

## Publication checklist and exact PowerShell commands

Run from the repository root after review. The first block stages the ten release files only. Inspect the complete staged diff, including notebook cell sources, check for unexpected paths or secrets, and confirm that prior audit outputs, private JSON, data and checkpoints are outside the index. The current offline baseline suite recorded 385 passes, two CUDA skips and two protobuf warnings. Run the focused readiness tests for this revision.

```powershell
$branch = 'research/mom-bpe-dgx-resources'
$base = 'c3dd4f3ef5c2bee3462f491cbe4c17251d9b1057'
if ((git branch --show-current) -ne $branch -or (git rev-parse HEAD) -ne $base) { throw 'Review the checkout before staging' }
if (git diff --cached --name-only) { throw 'Review the existing index first' }
$files = @(
  'experiments/mom_bpe/DOCUMENT_PROXY.md',
  'experiments/mom_bpe/document_proxy.py',
  'experiments/mom_bpe/dgx_handoff.py',
  'experiments/mom_bpe/dgx_resources.py',
  'experiments/mom_bpe/dgx_readiness.py',
  'experiments/mom_bpe/DGX_RESOURCE_READINESS.md',
  'notebooks/mom_bpe_dgx_readiness.ipynb',
  'tests/test_document_proxy.py',
  'tests/test_dgx_resources.py',
  'tests/test_dgx_readiness.py'
)
git add -- $files
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
if (Compare-Object ($files | Sort-Object) (git diff --cached --name-only | Sort-Object)) { throw 'Unexpected staged membership' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Diff check failed' }
git diff --cached --stat
git diff --cached
```

After explicit commit and GitHub publication approval, verify the push destination is the project repository and run the following commands in the same terminal. An ordinary push rejects a divergent remote branch. Preserve that remote branch and review the divergence if rejected.

```powershell
if ((git branch --show-current) -ne $branch -or (git rev-parse HEAD) -ne $base) { throw 'Checkout changed' }
if (Compare-Object ($files | Sort-Object) (git diff --cached --name-only | Sort-Object)) { throw 'Index changed' }
$origin = git remote get-url --push origin
if ($origin -notin @('https://github.com/Akshai0007/Thesi_Tokenizer_Pipeline.git','git@github.com:Akshai0007/Thesi_Tokenizer_Pipeline.git','https://github.com/Akshai0007/Thesi_Tokenizer_Pipeline')) { throw 'Verify the GitHub destination' }
git commit -m "feat: prepare DGX resource admission and readiness checks"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
$pin = git rev-parse HEAD
git push origin "HEAD:refs/heads/$branch"
if ($LASTEXITCODE -ne 0) { throw 'Push rejected or unavailable' }
$remote = git ls-remote --exit-code origin "refs/heads/$branch"
if ($LASTEXITCODE -ne 0 -or ($remote -split '\s+')[0] -ne $pin) { throw 'Remote commit verification failed' }
$pin
```

## DGX execution order

Open `notebooks/mom_bpe_dgx_readiness.ipynb` from the reviewed release. The notebook pins implementation commit `9b872ffb5a66bdf4d23df58acebbba7e788915d8`. An optional `MOM_BPE_RESOURCE_REVISION` setting must match that pin. Supply a fresh session label in `MOM_BPE_READINESS_SESSION`. In an existing kernel, IPython's `%env NAME=value` sets each variable. Every cell starts a fresh process in the recorded Python 3.12.0 environment. Cells 3–10 store durable observations under the private root in `resource-readiness/<commit>/<session>/`. A failed attempt supersedes older evidence for that cell until a rerun succeeds.

1. Verify the existing Python 3.12.0 interpreter, Unicode 15.0.0 and pinned package versions.
2. Fetch the approved exact commit into a new dedicated private checkout or verify the existing clean checkout at that pin. A conflict or incomplete checkout stops execution for inspection.
3. Authenticate the four historical result/identity pairs against the private revision's recorded byte hashes.
4. List checkpoint names and sizes, scheduled generation gaps and disk consumption for the four jobs.
5. Stream SHA-256 for every retained generation and compare the final file against each authenticated result. Missing earlier generations are listed. A missing or mismatched final generation blocks this check. Hash verification establishes byte identity; restoring model and optimizer tensors on DGX is a separate operation. The seed-17 runs used training commit `a874e1700898d26ce25e0c93581010f02101f9be` under the reviewed checkout `c3dd4f3ef5c2bee3462f491cbe4c17251d9b1057`. Use that original source and recorded environment for any separately authorized state restoration.
6. Inspect all physical GPU UUIDs, free MiB, utilization and compute PIDs with read-only NVIDIA queries.
7. After the institution allocates an A100, enter its full UUID and set `allocated = True` in cell 7. Defaults are 8192 MiB, 5%, exclusive. Only after explicit shared-use approval also set `shared = True` for 5120 MiB and 10%. This cell runs the original trainer preflight and a zero-training allocator diagnostic. It may initialize a CUDA context on the approved device.
8. Validate the latest diagnostic attempt, including the recorded GPU, policy, initial/final memory, allocator peaks and elapsed time. Diagnostic peaks measure the readiness process only.
9. Review all receipts for this commit and session. After rerunning an upstream check, repeat its dependent checks. Rerun cells 6–9 when the allocation changes. The report records scientific artifact integrity and sampled operational readiness.
10. Optionally list retained storage and receipt directories. The cell only inventories files.

CPU inventory streams one checkpoint at a time with a 1 MiB hash buffer. Each final checkpoint is about 466 MB. A complete seed-17 inventory covers 24 scheduled generations, about 11.2 GB of reads. Allow at least 24 GiB additional checkpoint storage for the proposed eight future runs. The GPU thresholds are admission requirements. Training peak VRAM and the adequacy of 5120 MiB for a full run still require measurement under a separately approved training allocation. Keep historical weights and receipts in private storage.

## Windows verification

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:HF_HUB_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
$env:PYTHONDONTWRITEBYTECODE='1'
python -m pytest -q -ra -p no:cacheprovider tests/test_dgx_readiness.py tests/test_dgx_resources.py tests/test_document_proxy.py tests/test_dgx_handoff.py tests/test_dgx_session_upgrade.py tests/test_notebook.py
```

The local-first controller tests in `test_dgx_session_upgrade.py` belong to the preceding local milestone. In a fresh checkout of this release, use the same command with that file omitted.
