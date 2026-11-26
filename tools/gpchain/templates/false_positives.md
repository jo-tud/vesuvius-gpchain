# False-positive mitigation and held-out validation

<!-- GP package template (34_prizes.md "False-positive mitigation", "Held-out validation"; VES-3585 part 2).
     package.py falsepos fills every field in braces from a JSON of measured numbers; a field with no number stays visible
     as MISSING and the package build refuses it. Text in this file must stay factual: every claim has a number and
     the file it came from. -->

Scroll: {scroll}. Scan: {scan}. Region: {region}. Package commit: {commit}.

<!-- VES-3629: package.py build --chain fills model_ref, window_px, window_mm, seeds, training_run,
     inference_runs, kept_rounds, docker_image, requirements, repro_cmd, orientation from records (records.py). -->

## 1. Why we believe the letters are real

| check | what it tests | result | source |
|---|---|---|---|
| Positive control, same pipeline | the chain recovers known ink on labelled public data before it is run on new data | {pc_summary} | {pc_source} |
| Label-shift null (N_shift) | ink AUC on labels moved by ±{shift_um} µm; real ink must beat every shift | AUC {auc} vs shifted max {shift_max} | {score_source} |
| Permutation null (N_perm) | the scorer gives 0.5 on shuffled labels | mean {perm_mean}, max {perm_max} | {score_source} |
| Frozen baseline | the model against a public baseline on the same px | {model} {auc} vs baseline {baseline_auc} | {score_source} |
| Frame check (FC) | the render samples the CT where the mesh says | r {fc_min}-{fc_max} on every piece (bar {fc_bar}) | {chain_source} |

## 2. Model window size and hallucination risk

Model: {model} ({model_ref}); input window {window_px} px = {window_mm} mm on the surface; letter height about
{letter_mm} mm. {window_note}

## 3. No overlap between training and prediction regions

{overlap_note} Check: {overlap_source}.

## 4. Held-out results on public ground truth

| data | labels | AUC (ours) | AUC (baseline) | px (ink / no ink) | source |
|---|---|---|---|---|---|
{heldout_rows}

## 5. Reproduction

`{repro_cmd}` (Docker image {docker_image}). Seeds: {seeds}. Human input: {human_minutes} min ({ledger_source}).

Tracking runs (34_prizes.md :133-134): training {training_run}; inference {inference_runs}.
Kept datasets and checkpoints (:131-132): {kept_rounds}. System requirements (:149): {requirements}.
Image orientation: {orientation}.
