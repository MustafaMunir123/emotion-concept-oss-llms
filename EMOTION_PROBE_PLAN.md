# Emotion Probe Project Plan (Heretic-Inspired)

## Scope and Origin

This plan captures decisions made in discussion starting from:

> "ok lets discuss only a bit... tell me if you find any similarity between the paper and the heretic project approach?"

through the latest decision:

> replace `Disgust ↔ Attraction` with `Anxious ↔ Relaxed`.

The goal is to build a practical, Heretic-inspired workflow for estimating emotion activation profiles in open-source LLMs.

---

## 0) Locked Decisions (Latest)

### 0.1 Locked Model Set

We will use the following model family targets:

- `Qwen/Qwen3-4B-Instruct-2507`
- `Mistral 7B` (phase-1 default: `mistralai/Mistral-7B-Instruct-v0.3`)
- `Falcon 7B` (phase-1 default: `tiiuae/falcon-7b-instruct`)
- `Zephyr 7B` (phase-1 default: `HuggingFaceH4/zephyr-7b-beta`)
- `OpenChat 7B` (phase-1 default: `openchat/openchat_3.5`)

If access restrictions apply for a checkpoint, replace with the closest publicly accessible instruct variant and log the substitution in run metadata.

### 0.2 Locked Runtime/Tooling

Execution environment is fixed to:

- **Jupyter notebooks**
- run from **VSCode**
- backing compute from **Kaggle kernel runtime** (GPU session)

Implementation and reporting must assume a **single-notebook workflow** (not long-running local scripts only).

---

## 0.3 Implementation Status (Current)

- **Step 0 - Setup/Reproducibility:** **Implemented**
  - single notebook scaffold in place
  - run metadata + step status persisted to `logs/`
  - notebook-first Kaggle workflow enabled
- **Step 1 - Model Selection:** **Implemented**
  - locked model targets wired
  - single-model execution filter (`selected_model`) implemented (default `qwen_4b`)
  - model/tokenizer load path with dtype fallback implemented
- **Step 2 - Probe Design + Data Contract:** **Implemented (Phase-1)**
  - pairwise probe spec generation implemented
  - JSONL schema/layout validator implemented
  - starter contextual train JSONL data added for all 5 pairs
  - currently configured for `train` split validation first
- **Step 3 - Layer/Token Residual Extraction:** **Implemented (Phase-1)**
  - token policy: first generated token
  - layer policy: all layers
  - batched residual extraction + artifact saving implemented
- **Step 4 - Detection/Measurement Definition + Probe Construction:** **Implemented (Phase-1)**
  - pair probe vectors built from saved residual artifacts (`mu_left - mu_right`, normalized)
  - signed scoring defined as cosine projection with layer-mean aggregation
  - pair percentages defined as `sigmoid(k*s)` (baseline `k=1.0`, calibration deferred to Step 6)
  - probe artifacts + per-pair measurement reports persisted
- **Step 5 - Layer Strategy Selection:** **Implemented (Phase-1, train-first)**
  - per-layer sweep metrics implemented (balanced sign accuracy + score separation)
  - best single layer and best contiguous layer band selection implemented
  - per-pair `layer_policy.json` and global model policy report persisted
  - configured to use `train` split until `val` split is added
- **Step 6 - Score-to-Percentage Calibration:** **Implemented (Phase-1, train-first)**
  - temperature `k` grid-sweep calibration implemented using binary NLL objective
  - calibration diagnostics added (Brier score, ECE)
  - per-pair and global calibration artifacts persisted
  - currently configured for `train` split; rerun on `val` once available
- **Step 7 - Evaluation per Model:** **Implemented (Phase-1, train-first)**
  - per-pair evaluation metrics implemented (balanced accuracy, confidence margins)
  - model-level aggregate metrics and pair CSV export implemented
  - overlap/confusion diagnostics implemented for `fear_vs_confidence` and `anxious_vs_relaxed`
  - currently configured for `train` split; rerun on `test` when available
- **Step 8 - Cross-Model Comparison Outputs:** **Implemented (pipeline ready; partial until all models run)**
  - comparison aggregator implemented (`model_summary.csv`, `pair_comparison.csv`, JSON bundle)
  - supports partial runs (warns if some expected models are missing)
  - rerun after Mistral/Falcon/Zephyr/OpenChat evaluation to produce final complete comparison
- **Step 9 - Final Visualization Tables/Charts:** **Implemented (presentation layer)**
  - colorful emotion-percentage table per model
  - all-model summary table
  - heatmap + comparison charts exported for reporting

No duplicate step headers exist in this plan: each of `Step 0` through `Step 8` appears once.

---

## 1) What We Agreed Conceptually

### 1.1 Similarity Between Anthropic Paper and Heretic

Both approaches follow a common pattern:

1. identify a behavior/concept direction in representation space,
2. measure activation relative to that direction,
3. intervene/steer (optional) and evaluate behavior shifts.

### 1.2 What Is Achievable

It is achievable to run this on multiple open-source models (at least 3) and report emotion profiles.

Important caveat: output percentages are **representation-activation percentages** (proxy metrics), not ground-truth "felt emotions."

### 1.3 Role of Heretic

Heretic is a strong starting scaffold because it already has:

- residual extraction,
- contrast-style vector construction,
- batch evaluation flow,
- multi-layer handling and optimization mindset.

But it is not a complete emotion benchmarking framework out-of-the-box. We need targeted extensions.

---

## 2) Probe Design Decisions

### 2.1 Why We Need Probes

We need probes/vectors because models do not expose direct, interpretable "emotion neurons."

Probes are the measurement instrument for:

- quantifying activation strength,
- comparing across prompts/models (after normalization),
- enabling optional causal checks via steering.

### 2.2 Why Baseline/Contrast Is Needed

We agreed that a single raw mean per emotion is possible but weaker/noisier.

Preferred strategy: contrastive probe construction to remove non-emotion artifacts (style/template/length/global tone).

### 2.3 Pairwise Opposite-Emotion Strategy (Final)

We will use pairwise contrast probes and measure both sides:

- `Sad ↔ Happy`
- `Angry ↔ Calm`
- `Fear ↔ Confidence`
- `Love ↔ Hate`
- `Anxious ↔ Relaxed`  (**final replacement**)

For each pair `(A, B)`:

- build `v_A_vs_B = normalize(mean(A) - mean(B))`
- represent opposite side with sign flip (or explicit mirror probe):
  - `v_B_vs_A = -v_A_vs_B`

This is Heretic-like and gives cleaner directional signals.

---

## 3) Layer/Token Strategy Decisions

### 3.1 Do We Manually Pick One Layer?

No hard manual single-layer choice by default.

### 3.2 How Heretic Does It

Heretic computes residuals for all layers (for the first generated token), then:

- supports per-layer directions,
- supports global direction via index/interpolation,
- uses optimization instead of fixed handpicked layer in many cases.

### 3.3 What We Will Do

Use Heretic-style all-layer extraction, then choose one of:

1. layer sweep and select best validation band, or
2. per-layer aggregation (recommended for stability), optionally
3. optimize layer weights later.

Phase 1 default: **layer sweep + aggregate best layer band**.

---

## 4) Detection and Measurement Definition

For each input prompt:

1. run model and extract residual activation(s) at chosen token/layer setup,
2. compute projection score per pair direction,
3. convert signed score to pair percentages.

### 4.1 Pair Score

- `s_pair = cosine(residual, v_A_vs_B)` (or normalized dot product)
- `s_pair > 0`: leaning toward `A`
- `s_pair < 0`: leaning toward `B`

### 4.2 Pair Percentages

For each pair:

- `p_A = sigmoid(k * s_pair)`
- `p_B = 1 - p_A`

Where `k` is calibration temperature/sensitivity.

### 4.3 Triggered Emotion (Per Pair / Overall)

- For a given pair: triggered side = max(`p_A`, `p_B`)
- Confidence margin = `abs(p_A - p_B)`

Optional global view: combine pair-side scores into a ranked profile across all 10 emotion labels.

---

## 5) Implementation Plan (Achievable Steps)

## Step 0 - Project Setup and Reproducibility

**Objective:** establish repeatable environment and experiment structure.

Tasks:

- define experiment folder layout (`data/`, `configs/`, `results/`, `plots/`, `logs/`).
- pin model list and prompt templates.
- define run metadata schema (model, date, commit hash, config id).
- define single-notebook structure for Kaggle execution:
  - `emotion_probe_pipeline.ipynb` with ordered sections (pipeline order, not sub-steps of any single step):
    - setup/config
    - data prep/splits
    - residual extraction
    - probe construction
    - layer sweep + calibration
    - evaluation + comparison
    - plots + report tables
- define in-notebook persistence policy (save vectors/results/checkpoints within the same run; do not depend on cross-notebook artifacts).

Deliverables:

- single notebook pipeline scaffold (`emotion_probe_pipeline.ipynb`)
- reusable helper modules for config/runtime/model loading
- run metadata and step-status artifacts in `logs/`

Acceptance:

- one command reproduces a baseline run end-to-end.
- single-notebook workflow reproduces baseline end-to-end in Kaggle runtime.

---

## Step 1 - Model Selection (3-5 Open Models)

**Objective:** choose 3-5 practical instruct models to compare.

Locked phase-1 targets:

- `Qwen/Qwen3-4B-Instruct-2507`
- `mistralai/Mistral-7B-Instruct-v0.3` (7B Mistral target)
- `tiiuae/falcon-7b-instruct` (7B Falcon target)
- `HuggingFaceH4/zephyr-7b-beta` (7B Zephyr target)
- `openchat/openchat_3.5` (7B OpenChat target)

Operational criteria:

- must run within Kaggle GPU memory/runtime limits,
- instruction/chat checkpoints only,
- chat template handling must be standardized across models.

Deliverables:

- model list with exact HF IDs and precision settings.

Acceptance:

- all selected models load and generate deterministic test outputs.

---

## Step 2 - Dataset Construction for Emotion Pairs

**Objective:** create clean contrast sets for each side of each pair.

For each pair `(A, B)`:

- build prompt set `D_A` and `D_B`,
- include diverse templates and lexical variation,
- avoid explicit leakage where possible (do not always name emotion directly),
- equalize count/length distributions as much as feasible.

Recommended minimum:

- 200-500 samples per side for phase 1.

Deliverables:

- `data/emotions/{pair}/{A}.jsonl`
- `data/emotions/{pair}/{B}.jsonl`
- split into train/validation/test.

Acceptance:

- balanced counts and quality spot-check pass.

---

## Step 3 - Residual Extraction Pipeline

**Objective:** extract residuals similarly to Heretic style.

Tasks:

- run each prompt through model,
- capture hidden/residual activations across all layers,
- standardize token-position policy (phase 1: first generated token or fixed assistant boundary token).

Deliverables:

- cached residual tensors with metadata.

Acceptance:

- deterministic re-run reproduces same statistics.

---

## Step 4 - Build Pairwise Probes

**Objective:** construct directional probes for each emotion pair.

For each pair `(A, B)`:

- compute per-layer means `mu_A[layer]`, `mu_B[layer]`,
- compute `v_A_vs_B[layer] = normalize(mu_A - mu_B)`,
- keep opposite side via sign inversion.

Deliverables:

- saved probe vectors by layer and pair.

Acceptance:

- non-degenerate vector norms and stable cosine structure across random subsamples.

---

## Step 5 - Layer Strategy Selection

**Objective:** pick robust layer usage policy.

Tasks:

- evaluate per-layer classification quality on validation set,
- choose:
  - best single layer, or
  - best layer band + weighted average (recommended).

Phase 1 default:

- best layer band with simple mean aggregation.

Deliverables:

- `layer_policy.json` per model.

Acceptance:

- validation metrics beat random and are consistent across pair tasks.

---

## Step 6 - Scoring and Percentage Calibration

**Objective:** convert projections into interpretable percentages.

Per pair:

- compute signed score `s_pair`,
- map using `sigmoid(k * s_pair)` to `(p_A, p_B)`.

Calibration:

- tune `k` on validation set for sensible confidence spread.

Deliverables:

- calibrated scorer module with documented formula.

Acceptance:

- confidence behaves sensibly on easy vs hard prompts.

---

## Step 7 - Evaluation per Model

**Objective:** produce robust comparable model-level results.

Metrics:

- pairwise accuracy on held-out labeled prompts,
- average confidence margin,
- confusion behavior between overlapping pairs (notably `Fear/Confidence` vs `Anxious/Relaxed`).

Deliverables:

- evaluation reports per model.

Acceptance:

- stable results across reruns; no catastrophic pair collapse.

---

## Step 8 - Cross-Model Comparison (Final Output)

**Objective:** generate final emotion profile per LLM.

Outputs for each model:

- per-pair percentages:
  - `Sad vs Happy`
  - `Angry vs Calm`
  - `Fear vs Confidence`
  - `Love vs Hate`
  - `Anxious vs Relaxed`
- dominant side and confidence margin per pair,
- aggregated profile table and chart.

Deliverables:

- `results/final/{model}_emotion_profile.csv`
- optional visualizations (`bar`, `radar`, `heatmap`).

Acceptance:

- all selected models have complete comparable profiles.

---

## 6) Validation and Quality Gates

Before trusting final percentages:

1. **Prompt sanity checks**: obvious prompts should activate expected side.
2. **Lexical confound checks**: paraphrase prompts with different wording.
3. **Template robustness**: evaluate multiple prompt templates.
4. **Layer robustness**: nearby-layer agreement check.
5. **Cross-run stability**: repeated runs should be close.

Optional causal check:

- steer along a pair direction and verify output tendency shifts in expected direction.

---

## 7) Risks and Mitigations

Risk: percentages interpreted as true emotions.  
Mitigation: label clearly as representation-activation proxy.

Risk: prompt/template confounds.  
Mitigation: diverse templates, leakage controls, paraphrase tests.

Risk: pair overlap (`Fear` with `Anxious`).  
Mitigation: report overlap diagnostics; add decorrelation later if needed.

Risk: cross-model incomparability due to scale differences.  
Mitigation: per-model calibration and standardized evaluation protocol.

---

## 8) Phase-1 Minimum Deliverable (MVP)

Must-have:

- 3-5 models evaluated,
- all 5 emotion pairs implemented,
- pairwise percentages per model,
- documented formulas and configs,
- reproducible run command.

Nice-to-have:

- steering-based causal sanity check,
- plots and concise interpretation report.

---

## 9) Immediate Next Actions

1. add `val`/`test` JSONL splits (currently `train` first),
2. add `val`/`test` JSONL splits and rerun **Step 5** on `val`,
3. rerun **Step 6** calibration on `val` once validation split is available,
4. rerun **Step 7** on `test` split once available and review overlap confusion stability,
5. run Steps 3-7 for `mistral_7b`, `falcon_7b`, `zephyr_7b`, and `openchat_7b`, then rerun **Step 8** and **Step 9** for final complete comparison visuals.

---

## 10) Non-Negotiable Reporting Language

In every report/result:

- Say "emotion activation percentage" or "probe-based percentage."
- Do **not** claim "model feels X% emotion."
- Include caveat that values are operational proxies from internal representations.

