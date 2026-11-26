# Document-local 38.8M GPT proxy

The driver uses the existing tied GPT with seven layers, width 448, seven heads, a 512-token context and 48,000 vocabulary slots. The pinned configuration specifies AdamW, clipping, exposure-based warmup and cosine decay. Seed 17 is the first run. [Configuration](document-proxy.yaml)

Each document starts with BOS as input and contributes one EOS target if completed. Blocks have at most 512 targets and use the immediately preceding token as the first input of later blocks. Learned positions reset and earlier attention history ends at the block boundary. Different documents can share a batch as separate rows. Right padding contributes zero loss. An optimizer update groups whole blocks until it reaches the 16,384-target threshold, which may be exceeded by up to 511 targets. Gradients are normalized by the actual number of targets in the update.

Each cache pins the tokenizer and fit, dataset streams, normalizer, preprocessing, block policy, control IDs and implementation. IDs and exact cumulative canonical byte endpoints are stored separately for train and validation. The endpoint records whether a cut falls inside a UTF-8 scalar. Fixed source bytes selects the longest whole-document prefix within the cap. Fixed targets consumes exactly the requested number of predictions in the same order, counting EOS as a target only for completed documents. The ledger adds verified partial bytes, including exactly located byte-fallback fragments. The cache must contain enough ordered data to substantiate the requested budget.

Evaluation uses every verified development-validation document. Each target contributes once to a per-document NLL and the shared scorer divides total nats by `ln(2)` and the exact canonical UTF-8 bytes. The sliding-window reference agrees on equivalent block contexts. Later blocks deliberately use the documented shorter context at their first positions. This protocol therefore defines its own comparable BPB scores across arms.

## Prepare and inspect the exact plans on CPU

From the repository root, with the existing local artifacts available, use the commands below. Preparation creates distinct immutable caches. Repeating it verifies an existing cache and returns its path. The 75 MB training cache cap contains both production selections. Planning performs no model updates.

```bash
export CUDA_VISIBLE_DEVICES=-1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
r=experiments/mom_bpe/runs
manifest="$r/scale48k-v1/a-100mb/manifest.json"
mkdir -p "$r/dp/ledgers"
for arm in M48 T5 T8 F-space; do
  case "$arm" in
    F-space) fit="$r/frontend-optimization-v1/full100/fits/$arm" ;;
    *) fit="$r/scale48k-next-v1/full100/fits/$arm" ;;
  esac
  cache=$(python -m experiments.mom_bpe.document_proxy prepare --manifest "$manifest" \
    --tokenizer "$fit" --arm "$arm" --cache-root "$r/dp/cache" --train-prefix-bytes 75000000) || exit 1
  for mode in fixed_source_bytes fixed_training_tokens; do
    python -m experiments.mom_bpe.document_proxy plan --cache "$cache" \
      --mode "$mode" --out "$r/dp/ledgers/$arm-$mode.json" || exit 1
  done
done
```

The four completed caches agree on the ordered fixed-byte selection of 10,355 documents and 49,998,152 canonical bytes. Each fixed-target selection has exactly 5,000,000 predictions. Context positions equal predicted targets and right padding is zero at the default microbatch of one block. Validation uses the same ordered 186 documents and 999,950 canonical bytes across all arms.

| Arm | Fixed-byte targets | Updates | Fixed-target complete bytes | Partial bytes | Total bytes | Updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M48 | 3,857,358 | 233 | 64,643,868 | 7,186 | 64,651,054 | 302 |
| T5 | 5,383,363 | 325 | 46,503,484 | 1,233 | 46,504,717 | 302 |
| T8 | 5,386,510 | 325 | 46,477,084 | 917 | 46,478,001 | 302 |
| F-space | 3,846,319 | 233 | 64,843,093 | 1,358 | 64,844,451 | 302 |

All four partial prefixes end on UTF-8 scalar boundaries. The full local ledgers are under `experiments/mom_bpe/runs/dp/ledgers/`. [Aggregate evidence and SHA-256 receipts](document-proxy-evidence-v1.json)

## DGX GPU smoke and matched runs

Use approved private scratch with the verified caches. Check the physical GPU index, UUID, free memory and active processes before setting the mask. Select a free A100 with at least 8 GiB available and a physical index other than 7. The driver independently checks the UUID, physical index, memory, utilization and BF16 support before enabling CUDA. The production driver keeps optimizer and model tensors in FP32 and uses BF16 autocast for GPU forward passes. A checkpoint is roughly 466 MB. Allow space for all retained generations and one in-flight write. A single default-budget fit typically needs roughly 3–4 GB of writable scratch for checkpoints. Allow additional capacity for caches and separate arm, budget and seed outputs.

```bash
r=experiments/mom_bpe/runs
nvidia-smi --query-gpu=index,uuid,name,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv
# Set this to the full UUID of the verified eligible physical GPU.
export CUDA_VISIBLE_DEVICES=GPU-REPLACE-WITH-VERIFIED-UUID
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mapfile -t matches < <(find "$r/dp/cache" -mindepth 1 -maxdepth 1 -type d -name 'F-space-*')
test "${#matches[@]}" -eq 1 && test -f "${matches[0]}/cache.json" || exit 1
python -m experiments.mom_bpe.document_proxy run --cache "${matches[0]}" \
  --mode fixed_training_tokens --budget 1024 --update-targets 512 \
  --checkpoint-updates 1 --device cuda --out "$r/dp/gpu-smoke/F-space"
```

After separate approval for the matched runs, use the default budgets below. Each invocation has its own result and checkpoint directory. Repeat the same command to resume. Use a new path and `--seed 23` or `--seed 41` for confirmation.

```bash
r=experiments/mom_bpe/runs
for arm in M48 T5 T8 F-space; do
  mapfile -t matches < <(find "$r/dp/cache" -mindepth 1 -maxdepth 1 -type d -name "$arm-*")
  test "${#matches[@]}" -eq 1 && test -f "${matches[0]}/cache.json" || exit 1
  for mode in fixed_source_bytes fixed_training_tokens; do
    python -m experiments.mom_bpe.document_proxy run --cache "${matches[0]}" --mode "$mode" \
      --device cuda --out "$r/dp/seed17/$arm/$mode" || exit 1
  done
done
```

## Recovery

### GPU admission and resource records

Use one institution-approved physical `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` UUID in `CUDA_VISIBLE_DEVICES`. Physical GPU 7 is prohibited. Both `dgx_handoff preflight` and `document_proxy run` accept the same resource options. Pass the same options to both commands.

| Option | Default | Behavior |
| --- | --- | --- |
| `--min-free-mib` | `8192` | Rejects a GPU below the specified free memory. |
| `--max-gpu-util-pct` | `5` | Maximum sampled utilization from 0 through 15 percent. A value above 5 requires explicit shared use. |
| `--allow-shared-gpu` | Off | Allows other compute processes on the selected UUID when all admission checks pass. |

The default rejects other compute processes. Both modes require a verifiable process listing. The trainer checks again under the run lock just before training. A future institution-approved shared run could pass `--min-free-mib 5120 --max-gpu-util-pct 10 --allow-shared-gpu` to both commands. These are point-in-time checks. Institutional allocation and coordination are still required. The updated handoff preflight needs an updated verified source bundle. Historical bundles remain readable through the archive verification commands.

The trainer writes one immutable `resource-telemetry/<attempt-id>/` directory per CUDA training attempt. The initial record contains admission settings, physical free MiB and utilization percent, the count of other compute processes and operational code hashes. A resume record identifies the starting update and restored checkpoint. Each checkpoint sample and the end record contain physical free MiB and utilization percent, this process's PyTorch peak allocated and reserved bytes, and elapsed training-attempt seconds. The end record describes training completion or interruption before validation. Use the last available record from each attempt to estimate total attempted training time, including replay and checkpoint I/O. An abrupt process loss can leave no end record and a gap after the last durable sample. GPU free memory and utilization are device-wide snapshots. Allocator peaks are process-local. Missing optional counters remain null.

Resource settings and observations live outside the experiment identity, checkpoint and result JSON. The reviewed operational driver retains the pinned scientific driver fingerprint for compatibility, while new resource records retain the actual driver and resource module hashes.

Each generation is published under a fresh digest-bound filename after serialization, flushing, closing and archive verification. The trainer keeps prior generations, including invalid generations for inspection. Resume checks the content, experiment identity, optimizer, schedule, data cursor and exposure ledger, then restores Python, NumPy and Torch RNG states. Final evaluation can resume from the completed training checkpoint and the published result binds that checkpoint to its validation identity. Open readers of older generations cannot block publication of the next generation. The legacy rolling proxy path is separate and retains its known Windows sharing risk.

The deterministic CPU regression checks exact model parameters, optimizer state, RNG state, exposure, training NLL and evaluation after interruption, including recovery when the newest generation is corrupt. GPU reproducibility depends on the pinned environment, hardware, kernels and deterministic settings. GPU execution and development-validation BPB remain pending.

## CPU verification

The bounded offline suite passed 183 tests with zero failures or skips and two existing protobuf deprecation warnings. A separate CLI smoke used the verified F-space encoder on authored text and the actual 38,799,936-parameter GPT. It completed one optimizer update on eight targets, scored 18 targets in two authored validation documents (48 canonical bytes), and returned the identical verified result on rerun. The smoke is an integration check. Research BPB requires the matched development runs.

In a separate one-thread CPU fixture with a width-16 GPT, all 512 target NLLs for one block agreed with the sliding-window reference within `4.77e-7` nats. The median after warmup was 0.007 seconds for block scoring and 0.957 seconds for sliding windows. These timings characterize the small fixture and its equivalent contexts.
