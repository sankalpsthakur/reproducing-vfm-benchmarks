# D'Souza, Pfeiffer & Mirlekar (SIMS EUROSIM 2024): the two CSVs the authors shared on Nov 2 2026, checked against
# Table 1 (check A), against the released simulator run with the configuration they supplied (check B), and for why
# the test window sits at a new level (check C). The criteria are fixed in PREREGISTRATION.md, written before this
# ran. The script reads Well1_average_data.csv and Well2_average_data.csv from the working directory and never copies
# them into its outputs; its one figure plots them, so the figure stays private until the authors agree to publication.
import json
import re
import subprocess
import time
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings('ignore')
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 30)
T0 = time.time()
OUT = {}
UA = {'User-Agent': 'Mozilla/5.0 (benchmark reproducibility check)'}
SRC = Path('/tmp/sources')
SRC.mkdir(parents=True, exist_ok=True)


def fetch(url, timeout=120):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def save():
    OUT['seconds'] = round(time.time() - T0)
    Path('dsouza_shared_check.json').write_text(json.dumps(OUT, indent=1, default=float))


def log(msg):
    print(f'[{time.time() - T0:6.0f} s] {msg}', flush=True)


# --- the shared files -----------------------------------------------------------------------------------------------------
COLS = ['t_h', 'W_ga', 'P_wf', 'P_wh', 'W_op', 'W_gp', 'W_lp']
DATA = ['W_ga', 'P_wf', 'P_wh', 'W_op', 'W_gp', 'W_lp']
WINDOWS = {'step 1 (20-30 h)': (20, 30), 'step 2 (52-62 h)': (52, 62), 'test window (85-95 h)': (85, 95)}
shared = {}
for w in (1, 2):
    df = pd.read_csv(f'Well{w}_average_data.csv')
    assert len(df.columns) == 7, df.columns
    df.columns = COLS
    shared[w] = df
OUT['shared_files'] = {
    f'well {w}': {'rows': len(df), 'header_ok': True, 't_first_h': float(df.t_h.iloc[0]), 't_last_h': float(df.t_h.iloc[-1]),
                  'last_row_repeats_previous': bool((df.iloc[-1, 1:].to_numpy() == df.iloc[-2, 1:].to_numpy()).all())}
    for w, df in shared.items()}


def window_levels(df, cols):
    out = {}
    for c in cols:
        r = {k: float(df.loc[(df.t_h >= a) & (df.t_h < b), c].mean()) for k, (a, b) in WINDOWS.items()}
        s1, te = r['step 1 (20-30 h)'], r['test window (85-95 h)']
        r['test vs step 1, %'] = (te / s1 - 1) * 100 if abs(s1) > 1e-9 else float('nan')   # a dead well has no step-1 level
        out[c] = r
    return out


OUT['shared_window_levels'] = {f'well {w}': window_levels(df, DATA) for w, df in shared.items()}
log('shared files read: ' + json.dumps(OUT['shared_files']))
for w in (1, 2):
    print(pd.DataFrame(OUT['shared_window_levels'][f'well {w}']).T.round(3), flush=True)
save()

# --- check A: Table 1 on the shared files -------------------------------------------------------------------------------
from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_percentage_error  # noqa: E402
from sklearn.multioutput import MultiOutputRegressor  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

TABLE1 = {'LSTM': (1.96, 1.53), 'MLP NN': (2.43, 5.49), 'MV linear regression': (2.14, 7.57), 'SVR': (5.04, 4.31),
          'KNN': (8.05, 5.41), 'Decision tree': (9.26, 5.43), 'Gradient boosting': (4.95, 5.55),
          'XGBoost': (4.23, 5.56), 'PLS': (9.54, 7.57), 'PCR': (9.52, 16.69)}
TR, VA = 2304, 4033
X_COLS, Y_COLS = ['W_ga', 'P_wf', 'P_wh'], ['W_op', 'W_gp', 'W_lp']
LSTM_SEEDS_PRIMARY, LSTM_SEEDS_SENS = list(range(10)), [0, 1, 2]


def paper_models(well, mlp_layers=2):
    """The audit's paper_models: the paper's section 3.1 and the GitHub notebooks (identical settings)."""
    w1 = well == 1
    return {
        'MLP NN': MLPRegressor(hidden_layer_sizes=mlp_layers, activation='identity' if w1 else 'logistic', solver='lbfgs',
                               alpha=0.0005, max_iter=1000, random_state=42),
        'MV linear regression': MultiOutputRegressor(LinearRegression()),
        'SVR': MultiOutputRegressor(SVR(kernel='rbf', C=1.0, gamma=0.1) if w1 else SVR(kernel='rbf', C=10.0, gamma=0.01)),
        'KNN': KNeighborsRegressor(n_neighbors=8, p=2),
        'Decision tree': DecisionTreeRegressor(max_depth=1, max_features='sqrt', min_samples_leaf=1, min_samples_split=2,
                                               random_state=0) if w1 else
        DecisionTreeRegressor(max_depth=10, max_features='log2', min_samples_leaf=1, min_samples_split=10, random_state=0),
        'Gradient boosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                                            learning_rate=0.1 if w1 else 0.05)),
        'XGBoost': MultiOutputRegressor(XGBRegressor(objective='reg:squarederror', n_estimators=300 if w1 else 100,
                                                     max_depth=7 if w1 else 3, learning_rate=0.01 if w1 else 0.1,
                                                     random_state=42)),
        'PLS': PLSRegression(n_components=1),
        'PCR': make_pipeline(StandardScaler(), PCA(n_components=1), LinearRegression())}


def mape(Y, P, per_target=False):
    if per_target:
        return [float(mean_absolute_percentage_error(Y[:, j], P[:, j]) * 100) for j in range(Y.shape[1])]
    return float(mean_absolute_percentage_error(Y, P) * 100)


def lstm_pred(X, Y, well, seed):
    """The audit's lstm_mape (the notebook's settings), returning the test predictions."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    inp = tf.keras.Input(shape=(1, 3))
    out = tf.keras.layers.Dense(3, activation='linear')(tf.keras.layers.LSTM(32 if well == 1 else 40)(inp))
    model = tf.keras.Model(inp, out)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X[:TR, None, :], Y[:TR], validation_data=(X[TR:VA, None, :], Y[TR:VA]), epochs=100, batch_size=32,
              verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])
    return model.predict(X[VA:, None, :], verbose=0)


cells, per_target, lstm_runs, sens = [], [], [], []
for w in (1, 2):
    df = shared[w].iloc[:5762]
    Xraw, Y = df[X_COLS].to_numpy(), df[Y_COLS].to_numpy()
    for scaler in ('all rows', 'training rows'):
        X = StandardScaler().fit(Xraw if scaler == 'all rows' else Xraw[:TR]).transform(Xraw)
        for name, model in paper_models(w).items():
            P = np.asarray(model.fit(X[:TR], Y[:TR]).predict(X[VA:]))
            if scaler == 'all rows':
                cells.append({'well': w, 'model': name, 'ours': mape(Y[VA:], P), 'table1': TABLE1[name][w - 1]})
                per_target.append({'well': w, 'model': name, **dict(zip(Y_COLS, mape(Y[VA:], P, True)))})
            else:
                sens.append({'well': w, 'model': name, 'sensitivity': 'scaler fit on training rows', 'ours': mape(Y[VA:], P)})
        if scaler == 'all rows':
            P = np.asarray(paper_models(w, mlp_layers=(2, 2))['MLP NN'].fit(X[:TR], Y[:TR]).predict(X[VA:]))
            sens.append({'well': w, 'model': 'MLP NN', 'sensitivity': 'two hidden layers of two', 'ours': mape(Y[VA:], P)})
        for seed in (LSTM_SEEDS_PRIMARY if scaler == 'all rows' else LSTM_SEEDS_SENS):
            try:
                P = lstm_pred(X, Y, w, seed)
                lstm_runs.append({'well': w, 'scaler': scaler, 'seed': seed, 'mape': mape(Y[VA:], P),
                                  **dict(zip(Y_COLS, mape(Y[VA:], P, True)))})
            except Exception as e:                       # recorded, never hidden
                lstm_runs.append({'well': w, 'scaler': scaler, 'seed': seed, 'mape': float('nan'), 'error': str(e)})
        log(f'check A: well {w}, scaler on {scaler} done')

lstm = pd.DataFrame(lstm_runs)
for w in (1, 2):
    prim = lstm[(lstm.well == w) & (lstm.scaler == 'all rows')]['mape'].dropna()
    cells.append({'well': w, 'model': 'LSTM', 'ours': float(prim.median()) if len(prim) else float('nan'),
                  'table1': TABLE1['LSTM'][w - 1], 'lstm_min': float(prim.min()) if len(prim) else float('nan'),
                  'lstm_max': float(prim.max()) if len(prim) else float('nan'), 'lstm_seeds': int(len(prim))})
    sensl = lstm[(lstm.well == w) & (lstm.scaler == 'training rows')]['mape'].dropna()
    sens.append({'well': w, 'model': 'LSTM', 'sensitivity': 'scaler fit on training rows (median of 3 seeds)',
                 'ours': float(sensl.median()) if len(sensl) else float('nan')})
cells = pd.DataFrame(cells)
cells['tolerance'] = np.maximum(0.5, 0.25 * cells.table1)
cells['agrees'] = (cells.ours - cells.table1).abs() <= cells.tolerance
n_agree = int(cells.agrees.sum())
best = {w: cells[cells.well == w].sort_values('ours').iloc[0]['model'] for w in (1, 2)}
# the pre-registered sensitivity "scaler fit on the training rows" (the paper's section 2.4), counted with the same
# tolerance: reported, not judged
sens_df = pd.DataFrame(sens)
tr = sens_df[sens_df.sensitivity.str.startswith('scaler fit on training rows')].copy()
tr['table1'] = [TABLE1[m][w - 1] for m, w in zip(tr.model, tr.well)]
tr['agrees'] = (tr.ours - tr.table1).abs() <= np.maximum(0.5, 0.25 * tr.table1)
OUT['check_A'] = {
    'A1_every_value_above_1pct': bool((cells.ours > 1.0).all()),
    'A1_values_at_or_below_1pct': cells.loc[cells.ours <= 1.0, ['well', 'model', 'ours']].round(4).to_dict('records'),
    'A2_cells_agreeing_of_20': n_agree,
    'A2_verdict': 'reproduces' if n_agree >= 16 else ('partly' if n_agree >= 10 else 'not'),
    'A3_lstm_best': {f'well {w}': best[w] == 'LSTM' for w in (1, 2)},
    'best_model': {f'well {w}': best[w] for w in (1, 2)},
    'reported_not_judged_cells_agreeing_with_training_rows_scaler': int(tr.agrees.sum()),
    'reported_not_judged_training_rows_scaler_cells': tr[['well', 'model', 'ours', 'table1', 'agrees']].round(4).to_dict(
        'records'),
    'cells': cells.round(4).to_dict('records'), 'per_target_primary': pd.DataFrame(per_target).round(4).to_dict('records'),
    'sensitivities': pd.DataFrame(sens).round(4).to_dict('records'), 'lstm_runs': lstm.round(4).to_dict('records')}
cells.to_csv('dsouza_shared_table1.csv', index=False)
save()
log(f"check A: A1 {OUT['check_A']['A1_every_value_above_1pct']}, A2 {n_agree}/20 ({OUT['check_A']['A2_verdict']}), "
    f"best {OUT['check_A']['best_model']}")
print(cells.pivot(index='model', columns='well', values=['table1', 'ours', 'agrees']).round(3), flush=True)

# --- check B: the released simulator with the authors' recipe ----------------------------------------------------------
if subprocess.run('command -v octave', shell=True).returncode != 0:
    subprocess.run('apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq octave >/dev/null',
                   shell=True, check=True, timeout=1800)
OUT['octave'] = subprocess.run(['octave', '--version'], capture_output=True, text=True).stdout.splitlines()[0]
log(OUT['octave'])
GH_RAW = 'https://raw.githubusercontent.com/dsouzaneville/FMH606-1-Masters-Thesis/main/'
GH_DIR, DV_DIR = SRC / 'github', SRC / 'dataverse'
GH_DIR.mkdir(exist_ok=True)
DV_DIR.mkdir(exist_ok=True)
for f in ('openloop_simulation.m', 'oil_field_model.m', 'make_w_gc.m', 'update_states.m', 'calculate_output.m'):
    (GH_DIR / f).write_bytes(fetch(GH_RAW + 'Simulator/' + f))
DV_IDS = {'oil_field_model.m': 268788, 'make_w_gc.m': 268786, 'update_states.m': 268787,
          'openloop_simulation.m': 268789, 'calculate_output.m': 268790}
try:
    dv = json.loads(fetch('https://dataverse.no/api/datasets/:persistentId/versions/:latest/files'
                          '?persistentId=doi:10.18710/KIJEWJ'))
    DV_IDS.update({f['dataFile']['filename']: f['dataFile']['id'] for f in dv['data'] if f['dataFile']['filename'] in DV_IDS})
except Exception as e:
    log(f'DataverseNO listing failed, using the audit ids: {e}')
for f, fid in DV_IDS.items():
    (DV_DIR / f).write_bytes(fetch(f'https://dataverse.no/api/access/datafile/{fid}'))
dv_model = (DV_DIR / 'oil_field_model.m').read_text().replace('\r', '').splitlines()
dv_model[0] = ('function  [dm, w_op, w_gop, P_a, P_ainj, rho_l, V_G, rho_m, P_tinj, Y_2, rho_ga, w_ginj, P_wf, '
               'w_l, w_o, w_g, P_wh, Y_3, w_gp, w_lp, w_wp] = oil_field_model(m, w_ga, Theta)')
(DV_DIR / 'oil_field_model.m').write_text('\n'.join(l.replace('%     w_wp = w_lp -w_op;', '    w_wp = w_lp - w_op;')
                                                    for l in dv_model) + '\n')
OUTS = ('[~, w_op, w_gop, P_a, P_ainj, rho_l, V_G, rho_m, P_tinj, Y_2, rho_ga, w_ginj, P_wf, w_l, w_o, w_g, P_wh, Y_3, '
        'w_gp, w_lp, w_wp]')
DV_TEMPLATE = """
Theta_nom = [2.51 1.63 0.08 0.07 0.15 0.15]; dt = {dt}; timespan = 96;
t = 0:dt:timespan*60*60;
w_gc = make_w_gc('2step', [40000 1; 20000 1; 40000 1], t);
w_ga = [{s} {s}]/100 .* w_gc;
m0 = [16422 15445 1669 1517 15229 13458];
m = zeros(length(t), 6); m(1,:) = m0;
for k = 1:length(t)-1
  m(k+1,:) = update_states(m(k,:), w_ga(k,:), Theta_nom, dt);
end
Theta_OL = Theta_nom .* ones(length(t),1);
{outs} = oil_field_model(m, zeros(length(t),2), Theta_OL);
w_ga = [w_ga; w_ga(end,:)];
for wi = 1:2
  dlmwrite(sprintf('{tag}_well%d.csv', wi), [t'/3600, w_ga(:,wi), P_wf(:,wi), P_wh(:,wi), w_op(:,wi), w_gp(:,wi), w_lp(:,wi)], 'precision', 12);
end
"""
GH_TEMPLATE = """
Theta_nom = [2.51 0.05 0.2]; dt = {dt}; timespan = 96;
t = 0:dt:timespan*60*60;
w_gc = make_w_gc('2step', [40000 1; 20000 1; 40000 1], t);
w_ga = {s}/100 .* w_gc;
m0 = [16422, 1669, 15229];
m = zeros(length(t), length(m0)); m(1,:) = m0;
for k = 1:length(t)-1
  m(k+1,:) = update_states(m(k,:), w_ga(k,:), Theta_nom, dt);
end
Theta_OL = Theta_nom .* ones(length(t),1);
{outs} = oil_field_model(m, zeros(length(t),1), Theta_OL);
dlmwrite('{tag}_well1.csv', [t'/3600, w_ga, P_wf, P_wh, w_op, w_gp, w_lp], 'precision', 12);
"""
SHARES = list(range(10, 101, 10))
VARIANTS = {'DataverseNO two-well, dt 60 s': (DV_DIR, DV_TEMPLATE, 60, 'dv60', (1, 2)),
            'DataverseNO two-well, dt 20 s averaged to 1 min': (DV_DIR, DV_TEMPLATE, 20, 'dv20', (1, 2)),
            'GitHub single-well, dt 60 s': (GH_DIR, GH_TEMPLATE, 60, 'gh60', (1,))}
jobs = []
for vname, (d, tmpl, dt, code, wells) in VARIANTS.items():
    for s in SHARES:
        tag = f'{code}_s{s}'
        (d / f'run_{tag}.m').write_text(tmpl.format(dt=dt, s=s, outs=OUTS, tag=tag))
        jobs.append((vname, d, tag, s))


def run_octave(job):
    vname, d, tag, s = job
    t1 = time.time()
    r = subprocess.run(['octave', '--no-gui', '--quiet', f'run_{tag}.m'], cwd=d, capture_output=True, text=True,
                       timeout=5400)
    return {'variant': vname, 'tag': tag, 'share': s, 'exit': r.returncode, 'seconds': round(time.time() - t1),
            'tail': (r.stdout + r.stderr)[-300:] if r.returncode else ''}


with ThreadPoolExecutor(max_workers=4) as ex:
    runs = list(ex.map(run_octave, jobs))
OUT['octave_runs'] = runs
save()
log(f"check B: {sum(r['exit'] == 0 for r in runs)} of {len(runs)} simulator runs succeeded; "
    f"slowest {max(r['seconds'] for r in runs)} s")
for r in runs:
    if r['exit']:
        print('FAILED', r, flush=True)

RCOLS = ['t_h'] + DATA


def read_run(d, tag, well, dt):
    raw = pd.read_csv(d / f'{tag}_well{well}.csv', header=None, names=RCOLS)
    if dt == 20:
        raw = raw.groupby(np.arange(len(raw)) // 3).mean()
    return raw.reset_index(drop=True)


errors, per_run, averaged, all_series = [], [], {}, {}
for vname, (d, tmpl, dt, code, wells) in VARIANTS.items():                 # check B first, saved before check C
    for w in wells:
        series = {}
        for s in SHARES:
            try:
                series[s] = read_run(d, f'{code}_s{s}', w, dt)
            except Exception as e:
                log(f'{vname} share {s} well {w}: no output ({e})')
        all_series[(vname, w)] = series
        if len(series) < len(SHARES):
            errors.append({'variant': vname, 'well': w, 'verdict': 'incomplete', 'runs': len(series)})
            continue
        n = min(len(x) for x in series.values())
        avg = sum(x.iloc[:n][RCOLS].to_numpy() for x in series.values()) / len(series)
        avg = pd.DataFrame(avg, columns=RCOLS)
        averaged[(vname, w)] = avg
        csv = shared[w]
        m = min(n, 5761)
        row = {'variant': vname, 'well': w, 'rows_compared': m}
        ok_tight, ok_close = True, True
        for c in DATA:
            rel = np.abs(avg[c].to_numpy()[:m] - csv[c].to_numpy()[:m]) / np.abs(csv[c].to_numpy()[:m])
            row[f'{c} max rel %'] = float(rel.max() * 100)
            row[f'{c} median rel %'] = float(np.median(rel) * 100)
            ok_tight &= rel.max() <= 0.01 and np.median(rel) <= 0.001
            ok_close &= rel.max() <= 0.05
        row['row 0 max rel %'] = float(np.max(np.abs(avg[DATA].to_numpy()[0] - csv[DATA].to_numpy()[0])
                                              / np.abs(csv[DATA].to_numpy()[0])) * 100)
        row['verdict'] = 'reproduces' if ok_tight else ('close' if ok_close else 'does not reproduce')
        errors.append(row)
errors = pd.DataFrame(errors)
errors.to_csv('dsouza_shared_regeneration.csv', index=False)
OUT['check_B'] = errors.round(5).to_dict('records')
save()
print(errors[['variant', 'well', 'rows_compared', 'row 0 max rel %'] +
             [f'{c} max rel %' for c in DATA] + ['verdict']].round(4).to_string(), flush=True)

for (vname, w), series in all_series.items():                             # check C, per single run
    for s, x in series.items():
        lv = window_levels(x, ['W_lp'])['W_lp']
        s1, te = lv['step 1 (20-30 h)'], lv['test window (85-95 h)']
        dead1 = abs(s1) <= 1e-6
        returns = (abs(te) <= 1e-6) if dead1 else abs(te / s1 - 1) < 0.001
        sd2 = float(x.loc[(x.t_h >= 52) & (x.t_h < 62), 'W_lp'].std())
        per_run.append({'variant': vname, 'well': w, 'share %': s, 'W_lp step 1': s1, 'W_lp step 2': lv['step 2 (52-62 h)'],
                        'W_lp test window': te, 'test vs step 1, %': lv['test vs step 1, %'], 'dead in step 1': dead1,
                        'returns': bool(returns), 'step-2 sd of W_lp': sd2, 'W_lp min over run': float(x.W_lp.min()),
                        'hours with W_lp at zero': float((x.W_lp <= 1e-6).sum() * (x.t_h.iloc[1] - x.t_h.iloc[0]))})
per_run = pd.DataFrame(per_run)
per_run.to_csv('dsouza_shared_runs.csv', index=False)
OUT['check_C'] = per_run.round(4).to_dict('records')
OUT['check_C_summary'] = {
    f'{v}, well {w}': {'non_returning_shares': g.loc[~g.returns, 'share %'].tolist(),
                       'shifts_of_non_returning, %': g.loc[~g.returns, 'test vs step 1, %'].round(3).tolist(),
                       'dead_in_step_1_shares': g.loc[g['dead in step 1'], 'share %'].tolist(),
                       'test_minus_step1_of_the_mean_kg_s': float(g['W_lp test window'].mean() - g['W_lp step 1'].mean())}
    for (v, w), g in per_run.groupby(['variant', 'well'])}
OUT['averaged_window_levels'] = {f'{v}, well {w}': window_levels(a, ['W_lp', 'W_op', 'W_gp', 'P_wf', 'P_wh'])
                                 for (v, w), a in averaged.items()}
save()
print(per_run.round(3).to_string(), flush=True)
print(json.dumps(OUT['check_C_summary'], indent=1), flush=True)

# figure: the shared files against each averaged regeneration (the files themselves are not written out)
fig, axes = plt.subplots(1, 2, figsize=(16, 4.5), sharey=False)
for ax, w in zip(axes, (1, 2)):
    ax.plot(shared[w].t_h, shared[w].W_lp, color='k', lw=2.2, label='shared CSV, liquid')
    for (v, ww), a in averaged.items():
        if ww == w:
            ax.plot(a.t_h, a.W_lp, lw=1, ls='--', label=f'mean of ten runs: {v}')
    ax.axvline(shared[w].t_h.iloc[VA], color='grey', ls=':', lw=1)
    ax.set_title(f'Well {w}: liquid mass rate')
    ax.set_xlabel('time, h (dotted: first test row)')
    ax.set_ylabel('kg/s')
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig('dsouza_shared_regeneration.png', dpi=120)
log('done')
save()
