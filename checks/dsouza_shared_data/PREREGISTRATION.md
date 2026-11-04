# Pre-registration: the D'Souza et al. data shared by the authors

Written Nov 4 2026, before any model or simulator was run on the shared files.

## Context
On Nov 2 2026 Neville D'Souza sent `Well1_average_data.csv` and `Well2_average_data.csv`, through Gaurav Mirlekar, in
answer to this audit's question 4. With them came the configuration: the gas injection was varied from 10% to 100% in
each well, giving ten datasets per well, at a sample time of one minute, and the mean of the ten was used as each
well's dataset. The paper's data section (2.3) gives neither the shares nor the averaging. It says only that an
open-loop simulation of the Janatian et al. (2022) model gave 5,762 samples per well, split 70/30.

Each file has 5,762 rows and seven columns: `Time [hrs]`, `W_ga [kg/s]`, `P_wf [bar]`, `P_wh [bar]`,
`W_op [kg/s]`, `W_gp [kg/s]` and `W_lp [kg/s]`. The files are not redistributed: they stay out of this repository
and out of every public notebook until the authors say otherwise.

### Seen before writing this
This was inspection only. No model was fitted and no simulator was run.
- **Window means of liquid rate.** Means over whole supply steps, so the start-up transient is included:

  | Well | Step 1 | Step 2 | Test window | Test window vs step 1 |
  |---|---|---|---|---|
  | 1 | 78.8 | 63.8 | 71.1 | -9.8% |
  | 2 | 75.0 | 50.4 | 69.3 | -7.6% |

  The test window is the rows from 4,033 onward.
- **Gas-lift rate.** In supply steps 1 and 3, `W_ga` is 5.0722 kg/s. That equals the audit's 30% DataverseNO run
  scaled to a mean share of 55%, which is 2.767 × 55/30.

## Check A: Table 1 on the shared files
**Primary protocol:**
- **Inputs:** `W_ga`, `P_wf` and `P_wh`, standardised on all rows as in the notebooks.
- **Targets:** `W_op`, `W_gp` and `W_lp`.
- **Split rows:** training 0-2,303, validation 2,304-4,032 and test 4,033-5,761, as in the notebooks.
- **Metric:** scikit-learn's `mean_absolute_percentage_error`, averaged uniformly over the three targets, times 100.
- **Models:** the audit's `paper_models`, which follow the paper's section 3.1 and the GitHub notebooks.
- **LSTM:** the audit's `lstm_mape` at ten seeds (0 to 9). The median is reported with its range.

**Sensitivities, reported but not judged:**
- the scaler fit on the training rows only, as the paper's section 2.4 states, with the LSTM at three seeds;
- the MLP with two hidden layers of two neurons, as the paper's section 3.1 states;
- per-target MAPE under the primary protocol.

**Judgement:**
- **A1, the regime.** Holds if every one of the 20 model-well values is above 1% MAPE.
- **A2, agreement.**
  - A cell agrees when |ours − Table 1| ≤ max(0.5, 0.25 × Table 1), in MAPE points.
  - Table 1 reproduces if at least 16 of the 20 cells agree, partly if 10-15 agree, and not if fewer than 10 agree.
- **A3, the ranking.** Whether the LSTM is the best model on each well, the paper's headline. Reported per well.

## Check B: the released simulator with the authors' recipe
**Recipe:**
- **Copy:** the DataverseNO two-well copy, with its `Theta_nom` and `m0`.
- **Shares:** `w_ga = [s s]/100 · w_gc` for s = 10, 20, ..., 100.
- **Supply:** 40,000, then 20,000, then 40,000 Sm³/h, in three equal steps over 96 h.
- **Outputs:** taken from `oil_field_model` as in the audit.
- **Averaging:** the mean of the ten runs at each time step.

**Variants:**
1. dt = 60 s, one row per step. This is the authors' "sample time of 1 minute".
2. dt = 20 s, averaged to one-minute rows. This is the audit's original setting.
3. Fallback, run in the same kernel: the GitHub single-well copy with `w_ga = s/100 · w_gc` at dt = 60 s.

**Alignment.** CSV row i is compared with t = i minutes for i = 0 to 5,760. Whether the CSV's last row (5,761)
repeats row 5,760 is reported, not compared.

**Judgement.** For each variant and well, over the six columns `W_ga`, `P_wf`, `P_wh`, `W_op`, `W_gp` and `W_lp`:
- **reproduces** if every column's maximum relative error is ≤ 1% and its median is ≤ 0.1%;
- **close** if every column's maximum relative error is ≤ 5%;
- **does not reproduce** otherwise.

## Check C: why the test window is a new level
- **Per-run levels.** For each single run, compare `W_lp` over the test window (85-95 h) with its level in step 1
  (20-30 h). A run returns if |Δ| < 0.1%.
- **What arithmetic requires, stated before the run.** An average of runs that each return must itself return. So the
  shift in the shared files needs at least one run that does not return.
- **Reported:**
  - which shares return;
  - each non-returning run's shift;
  - the oscillation in step 2 for each share, as the standard deviation of `W_lp` over 52-62 h.

## What decides the correction
- **If B reproduces, in any variant:**
  - "Neither released copy reproduces them" becomes "the released copy reproduces them with the configuration the
    authors supplied, which is in neither the paper nor the release", naming the variant.
  - The values read off the figures in section 2.2 are replaced by the files' values.
- **If A2 reproduces:**
  - The summary's "does not reproduce" for D'Souza et al. becomes "reproduces on the authors' data", with our numbers
    beside Table 1.
- **If A1 holds but A2 is partly or not met:** the regime claim is corrected, and the per-model disagreement is stated
  cell by cell.
- **Whatever the outcome:**
  - the original claim and the correction stand side by side, equally prominent;
  - question 4 is marked answered, with thanks to the authors.
