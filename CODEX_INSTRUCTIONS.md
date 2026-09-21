# Codex task: Beyond the Endpoint — minimal experimental pilot

## 1. Deliverable and execution limit

Read `plan.md`, especially Sections 4–7. Implement a small, complete Python project that lets me generate figures investigating **reading, selective intervention, and their joint availability during generation**. This is an exploratory implementation of the plan, not the entire publication agenda.

Write the project, not another proposal. Implement every advertised experiment without placeholders. Keep code explicit, short, typed where helpful, and understandable without a framework. Spend tokens on working code and essential tests, not repeated explanations, source dumps, or speculative abstractions.

**Run error checks only. Do not download pretrained weights, generate a research dataset, fit production probes, run experiment sweeps, or execute the README's research commands.** Tiny offline unit tests and a tiny distributed smoke test are permitted. State exactly which checks passed or were skipped. Never present synthetic test fixtures as experimental results.

Scope: one pretrained Stable Diffusion backend, deterministic DDIM, small readouts, two inexpensive controllers, endpoint-preserving controls, and one online decision protocol. Defer FLUX/SD3, foundation-model training, learned synthetic-scene generators, real-image inversion, dense Jacobians, and full-continuation backpropagation. Do not build placeholder infrastructure for them.

## 2. Project layout

```text
README.md
requirements.txt
configs/
  demo.json                 # Fully specified, editable exploratory preset
  smoke.json                # Tiny offline-check settings
  prompts.jsonl             # prompt_id, text, object name/category
  tasks.json                # Task texts, protected fields, evaluator thresholds
utils/
  __init__.py
  config.py                 # Argument parsing, validation, resolved configuration
  distributed.py            # Process setup, partitioning, reductions, cleanup
  sampler.py                # Load frozen components, step, preview, resume, decode
  data.py                   # IDs, splits, caches, manifests, atomic I/O
  tasks.py                  # Frozen image/text scorers and task definitions
  probes.py                 # Features, small heads, fitting and prediction
  editors.py                # Bounded interventions and online decisions
  controls.py               # Coordinate wrapper and equivalence checks
  metrics.py                # Metrics, compute accounting, bootstrap summaries
  plotting.py               # Figures from saved records only
exps/
  __init__.py
  collect.py
  fit.py
  intervene.py
  controls.py
  diagnose.py
  policy.py
  plot.py
tests/
  conftest.py
  test_sampler.py
  test_controls.py
  test_metrics.py
  test_distributed.py
logs/<run_id>/              # Configs, trajectories, observations, scores, result rows
ckpts/<run_id>/             # Small trained probes and their preprocessing statistics
figs/<run_id>/              # PNG/PDF figures and their plotted CSV tables
```

Reusable functionality belongs in `utils/`. Scripts in `exps/` contain readable orchestration and a guarded `main()`, not duplicated algorithms. No computation or downloads on import. No notebooks, Hydra, Lightning, experiment registries, external tracking services, or omnibus subprocess runner. Add only genuinely necessary dependencies; use standard JSON/CSV rather than a database.

## 3. Arguments and reproducibility

Use `argparse`. Support `--config FILE`; precedence is **explicit CLI > JSON configuration > documented parser defaults**. Every experimental or operational setting must have an argument: model/evaluator IDs and revisions, paths, seeds, counts, splits, resolution, stages, DDIM settings, guidance, precision, batch sizes, probe capacity/training settings, edit budgets/steps, thresholds, diagnostic subsets, bootstrap settings, and plot settings. Defaults belong in one place, never scattered through functions. Mathematical constants, schema field names, and tensor indices need not become arguments.

Pass settings explicitly to utility functions; do not read a global `args`. Include `--help`, `--dry-run`, `--local-files-only`, `--resume`, and configurable numerical tolerances. Help/dry-run must work without weights, network access, or CUDA. Reject unknown options/config keys and unsupported model/scheduler combinations rather than silently substituting another method.

Use stable sample IDs and deterministic seeds independent of rank, batch boundaries, and world size. Persist a train/validation/test split by **root seed**, keeping all related prompts, stages, coordinate variants, and edits in the same split. Fit preprocessing on training data only; select settings on validation only; test once. This pilot measures held-out-seed performance on a declared prompt bank, not unseen-prompt generalization.

Write a small demo preset, for example: 128 samples, 30 denoising steps, stages `[0,5,15,25,30]`, two nonzero edit budgets, two probe sizes, and explicitly limited intervention/diagnostic/policy subsets. These are configurable exploratory examples, not statistically justified final sample sizes. Include stage 0 and the terminal stage. Never launch the preset during implementation.

## 4. Frozen model and exact continuation

Use `StableDiffusionPipeline.from_pretrained`, with the example model ID `stable-diffusion-v1-5/stable-diffusion-v1-5`; accept another compatible checkpoint or local directory through arguments. Freeze the UNet, VAE, and text encoder with `.eval()` and `requires_grad_(False)`. Preserve the same prompt, negative prompt, and guidance for all branches of a sample.

Construct `DDIMScheduler.from_config(pipe.scheduler.config)` with `eta=0`. Restrict this implementation to DDIM; do not pretend it handles history-dependent solvers. Follow the installed, compatible Diffusers API. Derive latent dimensions, VAE scaling, scheduler prediction type, and preprocessing from component configuration—not literals such as four channels or `0.18215`.

Use this indexing throughout:

```text
x_k = latent BEFORE denoising update k, k = 0,...,N
Psi_k(x_k) = x_(k+1)
C_k(x_k) = decode the result of updates k,...,N-1
C_N(x_N) = decode(x_N), with no denoising update
```

Implement a short component-level loop with clear `step`, `preview`, `continue_from`, and `decode` functions. Use the actual fixed timestep array, CFG, model-input scaling, and `scheduler.step`. Obtain the predicted-clean latent from `pred_original_sample`; label its decoded image a **prediction**, not the actual remaining-sampler output.

Cache states from actual generation, not forward-corrupted final images. Store the full scheduler configuration, original timestep array, stage index, conditioning identity, and computational dtype. Resume the **suffix of that same schedule**. Never set a new schedule with `N-k` steps or pass a paused latent into a fresh pipeline call as though it were initial noise. Branch from independent copies and recalculate the denoiser output after an edit.

Collection and ordinary continuation use no-grad execution. Differentiable preview editing calls the individual components with gradients enabled with respect to the edit only. Do not call a no-grad-decorated pipeline entry point, detach the prediction, or pass through PIL/NumPy on this gradient path. Use differentiable tensor resizing/normalization and test it against the evaluator's ordinary preprocessing. Avoid retaining collection graphs or using inference-mode tensors directly as trainable edit variables.

Record preview work separately: producing a prediction at `x_k` can require another model call. Reuse it for an unedited update where valid, but do not count reuse twice or reuse a stale prediction after intervention.

## 5. Tasks and measurements

Provide two task families in `tasks.json`: **appearance** (red versus blue object) and **composition** (close-up versus whole-object framing). Supply a small prompt bank about ordinary objects that does not explicitly specify these answers. These are starting examples, not a redefinition of the research around one edit.

Use frozen `openai/clip-vit-base-patch32` as evaluator A for training targets and differentiable preview control. Use frozen `google/siglip-base-patch16-224` as evaluator B for independent endpoint evaluation, never for test-time control. IDs, revisions, text templates, preprocessing, and thresholds are configurable. Cache text features. Implement the appropriate APIs/preprocessing for each scorer; do not assume their logits have the same scale or probability interpretation.

For each binary task, save the positive-minus-negative matching score. Define decisions and success thresholds separately for each evaluator. Normalize regression targets using training statistics, but convert predictions back to the declared score scale for task decisions. Include object-category matching and the non-target task as protected measurements. Numerical image drift is a companion measurement, not proof of preserved identity or image quality.

These are **semantic proxies**, not ground truth. Export deterministic, non-cherry-picked audit grids and a simple CSV for optional human labels (`valid`, target class, protected-property preservation). Permit importing those labels during plotting/evaluation; do not silently discard disagreements, malformed images, or ambiguous cases. Without audited labels, captions must say “proxy task success.” Do not equate text matching with factual object count, localization, or overall validity.

Use final-image scores as training labels and post-hoc evaluation references only. A prospective observer/controller must not receive that example's final image, future features, final scores, or a request secretly chosen from its future label. Generate requests deterministically from sample/task IDs independently of endpoints.

## 6. Experiments and required figures

### A. `collect`: what is visible along generation?

Save selected `x_k`, their predicted-clean latents/previews, final images, endpoint scores, and costs. Save tensors without unnecessary precision loss; endpoint-equivalence checks must not be confounded by lossy caches. Produce records for a same-seed filmstrip, distinguishing raw-state visualizations, clean predictions, and actual endpoints. Fix display scaling across stages rather than normalizing every image to look equally readable.

### B. `fit`: when can a cheap observer predict the endpoint?

Fit per-stage linear and small one-hidden-layer MLP regressors for endpoint task scores. Compare spatially pooled raw latents with pooled predicted-clean latents. Keep the spatial pooling grid configurable and retain spatial layout; do not reduce everything to a channel average. Give both observers the same prompt information, for example a prompt-bank one-hot vector. Include a prompt-only training-mean baseline.

Fit one stage/head at a time to keep DDP and the code simple. Save model weights, feature/label normalizers, split IDs, and training configuration. Record parameter counts and calibration/training costs. Report held-out score error and classification accuracy, including balanced accuracy where defined. Show uncertainty and evaluator-A/B agreement rather than calling matching scores calibrated probabilities.

**Figures:** reading performance versus native stage and versus observation-inclusive compute; linear versus MLP; raw versus predicted-clean access. Better preview performance is not free extra information: show its extraction cost.

### C. `intervene`: when can an action still change the outcome selectively?

Implement four editors: no-op, norm-matched random perturbation, **probe-gradient editing**, and **preview-gradient editing**.

Represent an edit as `delta = upsample(u)`, where `u` is a configurable low-resolution latent field. Apply a bounded number of projected gradient steps. Bound

```text
RMS(delta) / training_stage_RMS <= edit_budget
```

in original latent coordinates; log actual RMS and maximum displacement. Compute the stage scale from training trajectories and reuse it for validation/test. Use the same feasible edit field/budget when comparing controllers at a stage.

The probe editor differentiates only the fitted raw-state readout. The preview editor differentiates current UNet prediction → predicted-clean latent → VAE → evaluator A. Neither differentiates through the full continuation. At stage N, the preview editor operates through VAE decoding alone.

A request specifies the desired side of a binary task. Optimize a target-margin hinge loss plus configurable protected-score and edit penalties. Protected references available to the controller come from its **current** readout/preview, not the final unedited sample. Then run the actual remaining sampler and assess the resulting final image with A and B.

Save target success, each protected error, perturbation magnitude, image drift, and costs separately. Report all trials and the predeclared originally-unsatisfied subset. No-op must fail an unmet request, but is legitimate when the request was already satisfied. Never infer controllability from surrogate-loss reduction or treat optimizer failure as proof of unreachability.

**Figures:** stage-by-budget target and selective-success heatmaps; success–preservation tradeoffs; reproducible same-seed before/after grids. Preserve failures in the records and figures.

### D. `controls`: can an interface change explain an apparent path difference?

Implement an invertible, time-dependent channel coupling:

```text
g_k(x_A,x_B) = (x_A, x_B + a_k*tanh(x_A))
a_k = amplitude*sin(pi*k/N)
g_0 = g_N = identity explicitly
g_k_inverse subtracts the same coupling
wrapped_Psi_k = g_(k+1) o Psi_k o g_k_inverse
```

Split compatible channels from the actual shape and validate it. Never feed a wrapped latent directly into the original UNet. In exact arithmetic this construction preserves paired endpoints; measure numerical drift rather than promising bitwise equality.

Compare native restricted readouts, refitted restricted readouts on wrapped coordinates, and correctly transported readouts `h_k(g_k_inverse(w))`. Apply the inverse **before lossy pooling**. Reuse the native head for the transported case. For finite-control sanity checks use `wrapped_I_k = g_k o I_k o g_k_inverse` and measure the budget in original coordinates. Show recovered capability and endpoint equivalence within tolerance.

Run the coordinate-readout comparison on raw-state observations. A correctly decoded predicted-clean endpoint is not a stage-k coordinate tensor; do not apply `g_k` to it and call that an equivalent observation. Use cached trajectories for fitting and a configurable small real-model subset for endpoint checks. Also implement a clock-only control that relabels exactly the same states/work. It must not improve any compute-based score. Changing the number of DDIM steps is a separate numerical experiment with potentially different endpoints.

**Figures:** paired endpoint drift; native/naive/transported reading profiles; clock-only curves against stage labels and compute. Label these as **coordinate/interface controls**, not evidence of inferior learned dynamics, equal intermediate marginals, or an intrinsic ordering of paths.

### E. `diagnose`: does editing retain its meaning across stages?

For adjacent selected checkpoints `k<j`, compute

```text
r_(k,j) = F_(j<-k)(I_k(x_k)) - I_j(F_(j<-k)(x_k))
```

where `F` is the actual sampler segment. Use the same absolute request and budget at both stages. When `j != k+1`, call it a **segment compatibility residual**, not a one-step residual. Continue both branches to endpoints. Record normalized residual, endpoint semantic disagreement, target failures, and protected errors.

**Figures:** residual versus endpoint disagreement/failure, stratified by stage and edit budget. Include no-op: it may have zero residual while failing a requested change. Treat these as exploratory diagnostics, not semantic certificates or causal explanations. Do not add dense Jacobians or a new prediction framework merely for this plot.

### F. `policy`: can reading and action help the same sample?

Implement an actual online inspect–decide–act loop. At selected stages, inspect only the available state and fitted observer, decide whether the request needs intervention, and optionally edit once. Freeze confidence thresholds and the fixed-stage baseline using validation data. An adaptive policy edits at the first sufficiently confident predicted violation, with a documented configurable deadline/fallback.

Compare never-edit, always-edit at a validation-selected stage, time-only/fixed-stage decision rules, the adaptive rule, and **wait-until-endpoint then edit the terminal latent through the VAE** using the same preview controller. The endpoint baseline may inspect its completed image; do not deliberately weaken its interface. Match declared budgets and report tradeoffs across total compute, not just number of edits.

Log every inspection. For single-stage joint success use the assessment at that stage; for an adaptive policy use its final act/no-act decision. Earlier deferrals are not assertions that the requirement is satisfied. The policy must not load the test sample's cached endpoint while deciding. Obtain the unedited reference only in separate post-hoc evaluation. Charge its generation to research evaluation, not secretly to one policy's online budget.

Define the per-trial joint event as

```text
correct assessment of intervention necessity
AND final request satisfied
AND declared protected properties preserved
AND budget respected
```

Compute it on the same seeds/requests at each stage and use the same evaluator/thresholds for all three curves within a comparison. Do not multiply separately averaged reading and editing success. Report necessary-action and already-satisfied cases separately so class imbalance cannot manufacture success. An empty useful-stage set is valid.

**Figures:** reading, action, and joint-success profiles; joint success versus online compute; adaptive versus fixed-stage and endpoint-only policies. Only show a useful window when the chosen success criterion is supported, with sample counts and uncertainty. Call any highlighted useful-stage set an empirical estimate, not a guarantee. Actual repeated inspection costs belong to the adaptive policy.

### G. `plot`: reproduce figures without models

Read saved records/checkpoints' metadata only; never load pretrained networks, regenerate samples, or recompute interventions. Save each figure as PNG and PDF, plus the exact CSV table used. Use consistent stage orientation (noise → endpoint), units, legends, and seed-level confidence intervals. Document how to read every figure and plausible null outcomes. Missing measurements remain missing, not zero; no smoothing that conceals failures.

## 7. Multi-GPU execution: real DDP, not a label

All compute-heavy entry points must work with both `python -m exps.NAME` and `torchrun ... -m exps.NAME`. Initialize from `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`; bind each CUDA process to its own local device. Use NCCL for CUDA and Gloo for CPU tests. Clean up process groups, set a finite timeout, and avoid failure-path barriers that hang other ranks.

For collection, intervention, diagnostics, and policies, load one frozen model/evaluator replica per process and partition stable sample IDs with non-padding strided shards. Do not wrap entirely frozen generators in DDP or add dummy trainable parameters. This is distributed data-parallel inference; the expensive generation work must actually be distributed.

For fitting, wrap the trainable head in **`torch.nn.parallel.DistributedDataParallel`** and use a `DistributedSampler`, `set_epoch`, and equal numbers of backward steps. Document any training padding/drop-last behavior. Use `.backward()` for DDP training. At evaluation, use unwrapped heads on non-padded shards; do not trigger DDP forward collectives inside uneven-length loops. Independent per-image edit variables are local optimizations, never all-reduced.

Reduce sums and counts, not unweighted means of rank means. Handle non-divisible sample counts, final partial batches, and fewer samples than ranks. Do not place collectives inside uneven inference loops. Rank zero alone writes shared manifests/checkpoints/final summaries; per-rank result files have unique names. Merge by stable keys and check duplicates/missing IDs. Model initialization must agree across ranks; random sample seeds must not depend on rank. Do not promise bitwise identity across hardware/precision choices.

No `DataParallel`, `device_map="auto"`, multi-GPU model sharding, or hard-coded `cuda:0`. Each GPU must fit one model replica; this design increases throughput, not the memory available to one sample. Document this clearly.

## 8. Logs, costs, and statistical safeguards

Save resolved configuration, command, software versions, model revisions, scheduler state, split manifest, hardware/precision, and a scientific cache fingerprint. Exclude rank/world-size/output paths from sample identity; include anything that changes the actual experiment. Reject incompatible cache reuse. Use atomic writes and resume completed IDs safely even with a different world size.

Use per-rank JSONL result rows and tensor trajectory files under `logs/`. Every measurement must identify sample/root seed, prompt, split, task/request, stage and native timestep, observation/controller/coordinate variant, budget, and status. Save raw scores, decisions, errors, successes, and failure reasons—not only aggregates. Store fitted heads under `ckpts/`, not mixed into source code.

Account for UNet forward calls and per-sample work including CFG branches, backward passes, VAE decodes, image/text scorer calls, probe work, editing iterations, synchronized GPU timing, and peak memory. Separate training/calibration, online decision/generation, and offline evaluation costs. Report work counters alongside measured time; never turn cached replay or nominal NFE into a claimed wall-clock speedup. Distinguish total GPU work from multi-GPU elapsed time.

Bootstrap complete root-seed clusters, retaining all related stages/actions, and use paired comparisons where possible. Expose bootstrap seed/count/confidence level. Show denominators and undefined metrics explicitly. Unsupported inputs should fail clearly; malformed generated samples should remain recorded failures. Never swallow arbitrary exceptions to make an experiment appear successful.

## 9. Required checks — and then stop

Run syntax/import checks, CLI help/dry-run checks, and small offline tests. Inspect installed versions and local APIs instead of dumping package sources. Do not replace the user's Torch/CUDA installation or download large dependencies just to claim a check passed; report unavailable checks honestly.

Test the real utility paths using tiny locally instantiated Diffusers components and deterministic fixtures, not solely mocked return values. Cover schedule-suffix replay, stage indexing, terminal decoding, branch isolation, no-op equivalence, preview calculation, gradient flow to edits with frozen component weights, norm projection, coordinate inversion/conjugation, endpoint recovery, split isolation, weighted aggregation, cache compatibility/resume, and figure output from saved toy records.

Run a bounded two-process CPU/Gloo test: one genuine DDP optimizer step must agree with a single-process global-batch reference within tolerance; verify parameter synchronization and non-padding inference partitioning for odd/tiny sample counts. GPU checks and real-pretrained integration checks are opt-in and require already available hardware/weights. Report skips precisely; offline tests do not certify pretrained image quality or multi-GPU CUDA execution.

Document commands such as:

```bash
python -m compileall -q utils exps tests
python -m pytest -q
python -m exps.collect --config configs/demo.json --dry-run
```

The test suite must not contact the model hub. Keep synthetic check artifacts in test temporary directories, never in research result folders.

## 10. README and final response

Explain the data flow and reading order: sampler → tasks → probes/editors → experiments → saved metrics/figures. Include a small equation-to-function map and a figure-to-question map. Specify installation requirements, the actual tested software versions, the per-GPU replica requirement, and the limits of unaudited proxy semantics.

Provide these **user-run commands**, without executing them during implementation:

```bash
# Set GPUS, RUN, and CFG to the desired values first.
torchrun --standalone --nproc-per-node="$GPUS" -m exps.collect --config "$CFG" --run-id "$RUN"
torchrun --standalone --nproc-per-node="$GPUS" -m exps.fit --config "$CFG" --run-id "$RUN"
torchrun --standalone --nproc-per-node="$GPUS" -m exps.intervene --config "$CFG" --run-id "$RUN"
torchrun --standalone --nproc-per-node="$GPUS" -m exps.controls --config "$CFG" --run-id "$RUN"
torchrun --standalone --nproc-per-node="$GPUS" -m exps.diagnose --config "$CFG" --run-id "$RUN"
torchrun --standalone --nproc-per-node="$GPUS" -m exps.policy --config "$CFG" --run-id "$RUN"
python -m exps.plot --config "$CFG" --run-id "$RUN"
```

Explain the equivalent single-process commands and how to reduce the sample/stage/edit subsets. Figures available after each stage should be plottable without waiting for the entire suite.

Your final reply should contain only a compact project summary, exact checks passed/skipped, and the next commands for me to run. Do not paste every file, invent results, promise a publication, or say the real-model project was validated when only offline tests ran.

## API references

Consult these only when resolving an API discrepancy; do not turn implementation into another literature review.

```text
https://huggingface.co/docs/diffusers/api/schedulers/ddim
https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img
https://huggingface.co/docs/transformers/model_doc/clip
https://huggingface.co/docs/transformers/model_doc/siglip
https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html
https://docs.pytorch.org/docs/2.9/elastic/run.html
```
