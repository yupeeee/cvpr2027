# Beyond the Endpoint: experimental pilot

This project measures when a frozen Stable Diffusion trajectory can be read, selectively edited, and used for an online decision. It implements deterministic DDIM, small endpoint-score probes, finite latent edits, coordinate/interface controls, segment diagnostics, and validation-calibrated policies. **No research results are included.** The presets are exploratory examples, not justified final sample sizes.

## Setup and reading order

Use a compatible existing PyTorch/CUDA environment and install missing packages from `requirements.txt`. Do not replace a working Torch/CUDA installation merely to reproduce the checks. The checked environment is Python **3.13.13**, Torch **2.11.0+cu130**, Diffusers **0.38.0**, Transformers **5.8.0**, Accelerate **1.13.0**, NumPy **2.4.4**, Matplotlib **3.10.9**, pytest **9.1.1**, Pillow **12.2.0**, SentencePiece **0.2.2**, Protobuf **6.33.6**, and tqdm **4.67.3**. Requirement ranges are broader than this one checked combination.

The default SigLIP tokenizer requires `sentencepiece` and `protobuf`, both included in `requirements.txt`. To repair an existing environment without changing Torch/CUDA, use its Python executable:

```bash
python -m pip install --no-deps --only-binary=:all: "sentencepiece>=0.2,<0.3" "protobuf>=4.25,<7"
```

Preparation checks the actual local processors/tokenizers before any model weights or GPU workers are loaded, so missing tokenizer dependencies fail early with an install command. Rerun the launcher in a fresh process after installation; existing completed records are resumed.

The default generator is `stable-diffusion-v1-5/stable-diffusion-v1-5`; evaluator A is `openai/clip-vit-base-patch32`, and evaluator B is `google/siglip-base-patch16-224`. The default and demo configuration require local/cached pretrained files. Pass `--no-local-files-only` when you run the launcher to permit downloading missing files. For a new run, the launcher prepares all three checkpoints before starting GPU workers; cached files are reused. On a restart, each unfinished stage checks model files only after validating its saved outputs. Supply existing local directories through `--model-id`, `--evaluator-a-id`, and `--evaluator-b-id`, or pass `--local-files-only` to require cached checkpoints. Model identifiers/revisions and evaluator preprocessing are configurable. Unsupported architectures or non-DDIM schedulers fail explicitly. Imports, help, plotting, and dry runs need no weights or CUDA.

`LocalEntryNotFoundError` means the selected checkpoint/revision is absent from the local Hugging Face cache while local-only loading is enabled. It is unrelated to torchrun's informational `OMP_NUM_THREADS` message. Pass `--no-local-files-only` when running to allow the missing files to download, or supply all three existing local checkpoint directories. The code respects `HF_HUB_OFFLINE` and other library offline settings; enabling downloads does not override those environment variables. An offline machine must have the requested files provisioned in advance.

Read the code in this order:

1. `utils/sampler.py`: frozen components, fixed schedule, differentiable prediction, exact suffix continuation.
2. `utils/tasks.py`: task definitions, tensor preprocessing, cached text features, separate A/B score scales.
3. `utils/probes.py`, `utils/editors.py`: pooled spatial features, DDP heads, projected finite edits.
4. `utils/experiments.py`, `utils/policies.py`, then `exps/`: experiment orchestration and the prospective/post-hoc boundary.
5. `utils/data.py`, `utils/metrics.py`, `utils/plotting.py`: provenance, costs, saved-record analysis.

| Location | Contents |
|---|---|
| `configs/` | Fully specified `demo.json`, tiny `smoke.json`, prompt bank, task texts and thresholds |
| `logs/<run_id>/` | Manifest, resolved configuration, component identities, full-precision trajectory tensors, per-rank JSONL and merged measurements, audit image pairs, frozen policy settings |
| `logs/<run_id>/setup/` | Unique per-rank shared evaluator-text and head preparation records, referenced by trials |
| `logs/<run_id>/progress/<session>/` | Per-rank JSON progress snapshots for one invocation; display state is separate from research measurements |
| `ckpts/<run_id>/` | Fitted heads, training-only feature/label statistics, prompt means, training stage RMS |
| `figs/<run_id>/` | PNG/PDF figures, exact plotted CSV tables, captions/manifest, optional human audit tables |

## Definitions and interfaces

The indexing is `x_k` **before** update `k`, for `k=0,...,N`. Stage `0` is initial noise and stage `N` is terminal. The final continuation at `N` decodes without a denoiser call. Paused states always resume the original timestep suffix; changing the number of DDIM steps defines another experiment.

| Mathematical object | Implementation |
|---|---|
| `Psi_k(x_k) = x_(k+1)` | `Sampler.step` |
| `C_k(x_k)` | `Sampler.continue_from`, followed by `Sampler.decode` |
| Predicted clean latent | `Sampler.preview`; uses `pred_original_sample`, not the actual future endpoint |
| Restricted readout `h_k` | `probes.features`, `Probe.predict` |
| `delta = upsample(u)`; `RMS(delta)/training_stage_RMS <= budget` | `editors.field`, `editors.project`, `editors.edit` |
| Coordinate map `g_k` and inverse | `controls.coordinate` |
| `g_(k+1) o Psi_k o g_k_inverse` | `controls.wrapped_step` |
| `g_k o I_k o g_k_inverse` | `controls.conjugate_editor` |
| Segment edit residual | `exps.diagnose` |
| Inspect–decide–act | `Runtime.online`, `policies.decide` |

Task scores are positive-minus-negative matching logits. Appearance compares red with blue; composition compares close-up/partial with whole-object framing. For request `r` in `{-1,+1}`, success requires `r * (score - decision_threshold) >= margin`. A/B decision thresholds, success margins, and protected tolerances are separate fields in `configs/tasks.json`; these scores are **not calibrated probabilities**. The other task and object-category matching are protected. Numerical image drift is recorded separately and does not establish object identity or quality.

Requests derive deterministically from sample/task IDs, independently of future outcomes. All variants of a root seed share one persisted split. Preprocessing statistics and stage RMS use training seeds only. Probe fitting selects epochs using validation data; policy calibration also uses validation only. Evaluation concerns held-out seeds on the declared prompt bank, not unseen-prompt generalization.

## Configuration and commands

Every entry point uses the same `argparse` configuration: **explicit CLI arguments > JSON > defaults in `utils/config.py`**. Unknown keys/options, invalid stages, duplicate trial settings, and incompatible caches fail. Boolean options support both forms, for example `--overwrite`/`--no-overwrite`. Saved work is reused automatically; `--resume` and `--no-resume` remain accepted compatibility flags and neither forces recomputation. `--help` lists settings and defaults.

`configs/demo.json` specifies 128 root seeds, 30 updates, stages `[0,5,15,25,30]`, budgets `[0.05,0.1]`, linear heads and MLP widths `[32,128]`. Intervention/diagnostic/control/policy subsets are limited to 16/4/4/8 samples; policy calibration uses 8 validation samples. A limit of `0` means all samples in the relevant split. Both stage `0` and stage `N` must be collected. The smoke preset is for configuration/error checks, not a research benchmark.

These are **user-run research commands**; they were not executed during implementation. Run the complete workflow with one executable script. This command explicitly permits downloading missing checkpoints:

```bash
./run_pilot.sh --config configs/demo.json --run-id pilot01 --device auto --no-local-files-only
```

The script runs `prepare → collect → fit → intervene → controls → diagnose → policy → plot`, labels its stages `[1/8]` through `[8/8]`, stops on the first failure, and defaults to `--config configs/demo.json --device auto`. All supplied arguments are forwarded unchanged to every stage and override those defaults. On a restart, `prepare` immediately defers to the owning stages; it does not reread trajectory or image tensors. Each stage visibly validates its expected records and referenced files, skips complete commands before CUDA/model preparation, and repairs missing or damaged outputs. Pass `--overwrite` to force a full rerun for the selected run ID. The script changes to its repository directory, so relative configuration and output paths are resolved there. Set `PYTHON=/path/to/env/bin/python` to choose a Python environment. Omit `--no-local-files-only` to require existing local/cached weights. The final plot command runs once and uses saved records only.

Progress is enabled by default in every entry point. **One combined tqdm bar summarizes all devices**, with completed/total work, elapsed time, and estimated time remaining. In `controls`, the total covers the entire command: wrapped probe fitting, all three readout variants, frozen-model setup, and endpoint/edit trials. Work estimates weight probe jobs by epochs and trials by remaining sampler/edit steps; the bar stays on the same total as its operation description changes. Nested training and evaluation details appear alongside each device's counts. The ETA estimates remaining work from observed throughput and may shift as differently priced phases begin; it does not predict the entire eight-stage workflow. Other commands show their current phase total. Bars are shown on standard error even when output is redirected.

Use `--no-progress` to hide progress output or `--progress-mininterval 2` to set the periodic refresh interval to two seconds. Both settings can also be supplied in JSON as `progress` and `progress_mininterval`; changing them does not invalidate a scientific cache. Completed trials and compatible probe checkpoints are reused by default and contribute to the completed count without repeating work; reused work is excluded from the throughput estimate. Uneven shards and ranks with no assigned samples are supported. GPU workers publish small asynchronous JSON snapshots under `logs/<run_id>/progress/<session>/`; rank zero renders their aggregate without adding collective synchronization to inference loops. Snapshots belong to one invocation and are not experimental result records.

With `--device auto`, each compute-heavy command discovers visible CUDA devices that can initialize, uses all of them with one worker per GPU, and falls back to CPU if none is usable. Set `CUDA_VISIBLE_DEVICES` to restrict which GPUs it can choose:

```bash
CUDA_VISIBLE_DEVICES=0,2 ./run_pilot.sh --run-id pilot01 --device auto --no-local-files-only
```

Use `--device cpu` for CPU execution. Device initialization does not establish that a GPU has enough free memory for the full frozen-model replica. Help and dry runs skip CUDA discovery, model loading, and downloads:

```bash
./run_pilot.sh --help
./run_pilot.sh --config configs/demo.json --device auto --dry-run
```

Individual stages remain available, for example `python -m exps.collect --config configs/demo.json --run-id pilot01 --device auto --resume`. Existing `torchrun --standalone --nproc-per-node=2 -m exps.collect ... --device auto` launches keep their assigned ranks and world size without launching nested workers. Use the same configuration and run ID for every stage. Plot after any completed stage: it reads available saved measurements and lists missing commands rather than requiring the full suite. Incomplete commands continue automatically, including with a different world size. For an individual command, `--overwrite` clears that command's saved outputs and invalidates dependent outputs; `plot --overwrite` replaces only generated files in `figs/<run_id>/`. A selected `--audit-labels` input is preserved even when stored inside that directory. Upstream artifacts and pretrained weights are retained. A standalone downstream stage still requires the upstream scientific configuration to match; changing it requires a new run ID or rerunning the full launcher with `--overwrite`. Per-rank files are merged by stable keys with duplicate/missing checks.

Artifact validation stores SHA-256 receipts under `logs/<run_id>/validation/`, bound to the scientific and component identities. The first pass validates and hashes existing tensors/checkpoints with visible progress. Later passes reuse that successful validation when file size, modification/change timestamps, and file identity are unchanged, without reading tensor contents. Changed metadata triggers rehashing; an unchanged digest reuses validation, while changed content is checked again. Missing or malformed files still trigger repair. Deleting validation receipts only removes this optimization. Hashing a file from scratch must still read every byte, so the first pass through existing outputs can take time. To explicitly check or acquire model files for an existing run, use `python -m exps.prepare --config YOUR_CONFIG --overwrite` (add `--no-local-files-only` to allow missing-file downloads). `exps.prepare` never deletes experiment outputs; its overwrite flag forces the model preflight.

Plotting also reuses complete outputs. `figures_manifest.json` stores input fingerprints and checksums for each figure group and its PNG/PDF/CSV/audit artifacts. Matching groups skip bootstrap summaries and image loading; missing or damaged figures are repaired while intact sibling figures remain untouched. Changes to source records, image tensors, imported audit labels, or plotting settings refresh the affected groups. Existing manifests are migrated after checking configuration, source records, image timestamps, and output formats.

Scheduler metadata checks ignore Diffusers' `_use_default_values` bookkeeping list, whose order can differ across processes or reloads. They still strictly compare the actual DDIM parameters and original timestep array. Existing compatible trajectory caches can resume after this fix; a real schedule change still fails validation.

To reduce work, copy `demo.json` to a new JSON file before collecting. For example, retain the schedule but set `num_samples` to `32`, `stages` to `[0,15,30]`, `intervention_stages` to `[15]`, `diagnostic_stages` to `[0,15]`, `control_stages` to `[15]`, `policy_stages` to `[15,30]`, subset limits to `2`, and `policy_calibration_limit` to `2`. One edit budget, one probe head, fewer edit steps, and fewer probe epochs further reduce work. Keep enough seeds for all splits. Use that same configuration for every command and choose a **new run ID**. The scientific fingerprint deliberately rejects changed scientific settings, task/prompt contents, or incompatible resolved component identities. Plot formatting/audit options and output paths are operational settings. Changing only a later command's stage list is not compatible cache reuse.

## What each experiment does

`collect` stores actual states, clean predictions/previews, final images, and endpoint scores. Collection can reuse the clean prediction from an unedited step and the terminal decode; reused calls are counted once. Standalone preview-observation work is still recorded so a preview is not treated as free. Filmstrips use one fixed raw-state display scale across stages.

`fit` trains one stage/access/head at a time, comparing spatially pooled raw and predicted-clean latents with equal prompt-bank information. It also records a prompt-only training-mean baseline, raw-scale errors, classification/balanced accuracy where defined, parameter counts, A/B agreement, and training/calibration costs.

`intervene` compares no-op, norm-matched random, raw-probe gradient, and preview gradient controllers. Random norm matching includes the work of the nominated controller. Preview editing differentiates only through the current prediction, VAE, and A; terminal editing uses the VAE alone. Protected references come from the current observation. Each edited latent is then actually continued and evaluated by A and B. All trials and originally-unsatisfied requests are reported; no-op is valid for already-satisfied requests.

`controls` refits restricted raw readouts in wrapped coordinates, reuses native heads with the inverse applied before pooling, and checks finite conjugated edits and paired endpoints. The coupling is explicitly identity at both boundaries. Clock-only plots relabel the same states and work. These are coordinate/interface controls; they do not show equal intermediate marginals, better learned dynamics, or an intrinsic ordering of paths. Changing the DDIM step count is not this control.

`diagnose` compares editing before a real sampler segment with editing after it, using the same absolute request and relative budget. Nonadjacent checkpoints produce a **segment compatibility residual**. Both branches reach endpoints. A zero no-op residual may still fail a request; neither a small residual nor a failed optimizer is a semantic certificate or evidence of unreachability.

`policy` generates from initial noise and inspects only available states. Its final act/no-act assessment defines the joint event; earlier deferrals assert nothing about necessity. Validation selects the fixed stage and adaptive confidence distance by joint success, then fewer UNet sample-calls and smaller candidate as tie-breakers. The threshold is a score distance, not a probability. The adaptive rule acts on the first sufficiently confident predicted violation and uses `--policy-deadline` plus `--policy-fallback predicted|edit|skip`. The time-only rule uses validation necessity prevalence, with `--policy-time-only-default` for an unseen request group. Settings are saved before test evaluation.

Policies include never edit, always edit at the selected stage, time-only, fixed-stage inspection, adaptive inspection, single-stage profiles, and endpoint inspection followed by terminal-latent preview editing. The endpoint policy sees its completed image. B is never used for online control. Cached unedited endpoints are loaded only after online decisions finish. Joint success requires correct necessity assessment, final request success, protected preservation, and a respected budget on the **same trial**; it is not a product of separate averages. Necessary-action and already-satisfied cases remain separate.

## Saved figures and audits

Every figure has PNG, PDF, and the exact plotted CSV; `figures_manifest.json` records captions. Stages run noise → endpoint. Confidence intervals bootstrap complete root-seed clusters; a one-root interval and missing measurements remain undefined. CSV tables include denominators and failure counts. No smoothing hides failures.

| Figure family | Question and interpretation |
|---|---|
| `filmstrip_*`, `before_after_audit` | What do fixed-seed states, clean predictions, and actual endpoints show? Predictions may look readable without predicting final properties well. |
| `reading_*` | When do linear/MLP raw/clean observers predict endpoint scores? Stage and observation-inclusive UNet-work views may give different rankings. |
| `intervention_*`, `preservation_tradeoff`, `protected_errors` | Can edits satisfy requests while preserving declared proxies? Target success alone may hide collateral changes. |
| `endpoint_drift`, `coordinate_reading_*`, `coordinate_finite_control`, `clock_only_*` | Does coordinate transport recover reading/control and preserve paired endpoints within tolerance? Clock relabeling cannot improve compute-based scores. |
| `paired_raw_clean_reading`, `paired_coordinate_reading` | Do paired differences survive root-seed uncertainty? Unmatched records and undefined intervals are explicit. |
| `human_audit_success` | Do optional human labels agree with the proxies? Ambiguous, malformed, duplicate, and missing annotations remain counted. |
| `diagnostic_*` | Does segment residual track endpoint disagreement or failure? No association is a valid outcome. |
| `policy_reading_*`, `policy_action_*`, `policy_joint_*` | Do assessment and selective action work together on the same requests? A useful-stage estimate may be empty. |
| `policy_comparison_*` | Does adaptation help relative to fixed-stage and endpoint policies after actual repeated inspection/edit costs? Baselines may perform equally well or better. |

Rings in joint-success profiles mark stages whose lower confidence bound meets `--useful-success`. They are empirical estimates with sample counts, not guarantees.

Audit selection is deterministic by sorted IDs/keys, including failures. Copy `figs/<run_id>/audit_template.csv` before labeling; retain `key` and enter `valid` and `protected_preserved` as `0`/`1`, and `target_class` as `-1`/`+1`. Blank, `?`, `ambiguous`, or `unknown` remain unresolved. Import labels with:

```bash
python -m exps.plot --config configs/demo.json --run-id pilot01 --audit-labels path/to/labels.csv
```

Optional human-success figures are separate from proxy figures. Imported labels, malformed/unknown entries, and disagreements with both proxies remain in `audit_imported.csv` and `audit_summary.json`; annotations do not silently replace the proxy figures. Unaudited captions say **proxy task success**. Text matching does not establish factual object count, localization, validity, or overall image quality.

## Distributed execution and accounting

Each inference rank loads one complete frozen generator/evaluator replica on its own local GPU. **Each GPU must fit that replica**; more GPUs increase throughput, not memory available to one sample. Stable IDs use strided, non-padding inference shards. Frozen models are not wrapped in DDP, and edit variables remain local. Odd counts and fewer samples than ranks are supported.

Probe heads use actual `DistributedDataParallel`, `.backward()`, and an epoch-seeded `DistributedSampler`. Training pads to equal rank lengths, so some training examples can repeat; losses use sums/counts. Evaluation uses unwrapped heads on non-padding shards. CUDA uses NCCL; the tiny CPU distributed check uses Gloo. Process groups have finite timeouts and are cleaned up without failure-path barriers. Cross-hardware/precision bitwise identity is not promised.

Records separate shared text/head setup, training/calibration, online generation/decisions, and offline endpoint evaluation. They retain UNet calls and CFG sample-work, backwards, VAE decodes, image/text scorer calls, probe work, edit iterations, synchronized device timing, and peak memory. Setup records are shared once per replica, not added anew to every policy. Cached unedited-reference generation belongs to research evaluation; replayed prefixes are marked as cached. Counters and per-rank timing describe different quantities: total GPU work is not distributed elapsed time, and cached replay or nominal NFE does not demonstrate a wall-clock speedup.

## Bounded checks

Current verification: `python -m compileall -q utils exps tests` and `bash -n run_pilot.sh` passed. The offline `python -m pytest -q -rs` suite passed **158 tests, with 2 optional tests skipped** (42.98 seconds). Coverage includes preparation without redundant tensor scans, persistent SHA-256 validation receipts, changed-file detection, warm-cache reuse without deserialization, whole-command controls progress, automatic complete-stage reuse before model/CUDA startup, selective repair of missing/corrupt outputs, figure caching, overwrite cleanup and human-label preservation, policy-settings repair, separate-process DDIM metadata compatibility, mocked CUDA discovery, and a tiny SigLIP/SentencePiece tokenizer round trip. The suite includes the two-process CPU/Gloo and synthetic workflow-restart tests, including deliberately late workers during statistics/settings repair and unchanged valid artifacts. All eight entry points passed `--help`; the actual shell launcher completed all eight stages with `--device auto --overwrite --dry-run` while hub access and CUDA visibility were disabled. The skips were the unrequested CUDA and existing-local-pretrained opt-ins described below. No pretrained weights were downloaded and no research commands were executed. Launcher regression tests use a logging executable stand-in to check stage order, quoting, argument forwarding, and failure propagation.

The implementation checks use tiny locally instantiated Diffusers/Transformers components and temporary artifacts. They cover schedule suffixes/indexing, terminal decode, branch/no-op equivalence, preview gradients with frozen weights, evaluator preprocessing, edit projection, coordinate inversion/conjugation, cache/split/resume behavior, saved-record figures, online endpoint isolation, and a bounded two-process CPU/Gloo test comparing a genuine DDP optimizer step with a global-batch reference. The ordinary suite cannot contact the model hub.

```bash
python -m compileall -q utils exps tests
python -m pytest -q
python -m exps.collect --config configs/demo.json --dry-run
bash -n run_pilot.sh
./run_pilot.sh --config configs/demo.json --device auto --dry-run
```

`tests/test_optional_integration.py` contains two default skips: CUDA execution requires `PILOT_CHECK_CUDA=1`; real-pretrained integration requires `PILOT_CHECK_PRETRAINED=1` plus existing `PILOT_MODEL_DIR`, `PILOT_CLIP_DIR`, and `PILOT_SIGLIP_DIR`. `PILOT_CHECK_DEVICE` selects its device. These opt-ins never download weights. They were not requested or executed during implementation. Offline fixtures are not experimental results and do not validate pretrained image quality or multi-GPU CUDA execution.
