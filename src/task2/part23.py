"""
Task 2 — Parts 2 and 3 (vectorised, corrected)
- Part 2: Real-valued GA (tournament + single-point crossover) for FLC optimisation
          with 120/30 train/test split, 5 independent runs, per-output RMSE
- Part 3: Real-valued GA vs PSO on CEC'2005 F1 (bounds [-100,100]) and F10 (bounds [-5,5])
          15 independent runs each, Wilcoxon test
- Scenarios: Actual rule firing strengths computed from vectorised FLC
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from scipy.stats import wilcoxon
import warnings
import os
warnings.filterwarnings('ignore')

_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIG_DIR  = os.path.join(_ROOT, 'docs', 'figures')
DATA_DIR = os.path.join(_ROOT, 'data')

print("=" * 60)
print("TASK 2 — PARTS 2 & 3 (corrected)")
print("=" * 60)

# ── Load expert dataset (reference-policy labels) ────────────────
print("\n[Data] Loading reference-policy dataset...")
data = np.loadtxt(f"{DATA_DIR}/task2_expert_dataset.csv", delimiter=',', skiprows=1)

rng = np.random.RandomState(42)
idx = rng.permutation(len(data))
data = data[idx]

N_TRAIN = 120
train_data = data[:N_TRAIN]
test_data  = data[N_TRAIN:]

inputs_train  = train_data[:, :6]
outputs_train = train_data[:, 6:]
inputs_test   = test_data[:, :6]
outputs_test  = test_data[:, 6:]

print(f"  Total: {len(data)} examples → train: {N_TRAIN}, test: {len(test_data)}")

# ── MF structure (mirrors Part 1 skfuzzy definitions) ───────────
INIT_PARAMS = {
    'temperature':  [('trap',[10,10,14,19]),('tri',[16,22,27]),('tri',[24,28,33]),('trap',[30,36,40,40])],
    'humidity':     [('trap',[20,20,30,40]),('tri',[35,50,65]),('trap',[60,70,90,90])],
    'activity':     [('trap',[0,0,.2,.35]),('tri',[.25,.5,.75]),('trap',[.65,.8,1,1])],
    'light_level':  [('trap',[0,0,100,250]),('tri',[150,400,650]),('trap',[550,750,1000,1000])],
    'time_of_day':  [('trap',[0,0,5,7]),('tri',[6,9,12]),('tri',[11,14,18]),('trap',[16,20,24,24])],
    'comfort_pref': [('trap',[0,0,.2,.4]),('tri',[.3,.5,.7]),('trap',[.6,.8,1,1])],
    'heater':       [('trap',[0,0,5,15]),('tri',[10,30,50]),('tri',[40,60,80]),('trap',[70,85,100,100])],
    'fan':          [('trap',[0,0,5,15]),('tri',[10,30,50]),('tri',[40,60,80]),('trap',[70,85,100,100])],
    'dimmer':       [('trap',[0,0,5,15]),('tri',[10,30,50]),('tri',[40,60,80]),('trap',[70,85,100,100])],
    'blinds':       [('trap',[0,0,5,20]),('tri',[15,50,85]),('trap',[70,90,100,100])],
}

VAR_ORDER   = ['temperature','humidity','activity','light_level','time_of_day','comfort_pref',
               'heater','fan','dimmer','blinds']
BOUNDS_MAP  = {
    'temperature':(10,40),'humidity':(20,90),'activity':(0,1),'light_level':(0,1000),
    'time_of_day':(0,24),'comfort_pref':(0,1),'heater':(0,100),'fan':(0,100),
    'dimmer':(0,100),'blinds':(0,100),
}
INPUT_VARS  = ['temperature','humidity','activity','light_level','time_of_day','comfort_pref']
OUTPUT_VARS = ['heater','fan','dimmer','blinds']

def params_to_chrom(params):
    return np.concatenate([np.array(mp) for var in VAR_ORDER for _, mp in params[var]])

def chrom_to_params(chrom):
    params, idx = {}, 0
    for var in VAR_ORDER:
        lst = []
        for mtype, mp0 in INIT_PARAMS[var]:
            n = len(mp0)
            lst.append((mtype, list(chrom[idx:idx+n])))
            idx += n
        params[var] = lst
    return params

CHROM_LEN = len(params_to_chrom(INIT_PARAMS))
print(f"  Chromosome length: {CHROM_LEN} genes")

# ── Vectorised MF evaluators ─────────────────────────────────────
def trimf_v(x, a, b, c):
    e = 1e-10
    L = np.where(b > a, (x-a)/(b-a+e), (x>=a).astype(float))
    R = np.where(c > b, (c-x)/(c-b+e), (x<=c).astype(float))
    return np.clip(np.minimum(L, R), 0, 1)

def trapmf_v(x, a, b, c, d):
    e = 1e-10
    L = np.where(b > a, (x-a)/(b-a+e), (x>=a).astype(float))
    R = np.where(d > c, (d-x)/(d-c+e), (x<=d).astype(float))
    return np.clip(np.minimum(L, np.minimum(1.0, R)), 0, 1)

def mf_centroid(mtype, mp):
    return mp[1] if mtype == 'tri' else (mp[1]+mp[2])/2

def eval_mf(x, mtype, mp):
    return trimf_v(x, *mp) if mtype == 'tri' else trapmf_v(x, *mp)

# Rule table: (input_col_indices, input_set_indices, output_col, output_set)
# Columns: 0=temp,1=hum,2=act,3=light,4=time,5=comfort
# Sets: temp→0=cold,1=comfortable,2=warm,3=hot
#       hum/act→0=low,1=medium,2=high
#       light→0=dark,1=dim,2=bright
#       time→0=night,1=morning,2=afternoon,3=evening
#       comfort→0=cool,1=normal,2=warm
# Outputs: 0=heater,1=fan,2=dimmer,3=blinds
#   heater/fan/dimmer sets→0=off,1=low,2=medium,3=high; blinds→0=closed,1=part_open,2=open

RULES = [
    ([0,2,5],[0,0,2],0,3),           # R1
    ([0,1,2,5],[0,0,0,1],0,2),       # R2
    ([0,1,2,5],[0,2,0,1],0,1),       # R3
    ([0,2,5],[0,0,0],0,1),           # R4
    ([0,2],[0,2],0,1),               # R5
    ([0,2,5],[1,0,2],0,2),           # R6
    ([0,5],[1,1],0,1),               # R7
    ([0,5],[1,0],0,0),               # R8
    ([0,1,2,5],[2,2,0,2],0,1),       # R9
    ([0,5],[2,0],0,0),               # R10
    ([0],[3],0,0),                   # R11
    ([0,1,2,5],[0,2,2,2],0,2),       # R12
    ([0,1,2,5],[3,2,2,0],1,3),       # R13
    ([0,5],[3,0],1,2),               # R14
    ([0,5],[3,1],1,1),               # R15
    ([0,5],[3,2],1,0),               # R16
    ([0,1,2,5],[2,2,2,0],1,2),       # R17
    ([0,1,5],[2,2,0],1,1),           # R18
    ([0,2,5],[1,2,0],1,1),           # R19
    ([0],[0],1,0),                   # R20
    ([3,4],[0,1],2,3),               # R21
    ([3,4,5],[0,2,2],2,2),           # R22
    ([3,4,5],[0,2,0],2,0),           # R23
    ([3,4,5],[0,3,2],2,3),           # R24
    ([3,4,5],[0,3,0],2,2),           # R25
    ([3,4],[0,0],2,1),               # R26
    ([3,4,5],[1,1,2],2,2),           # R27
    ([3,4,5],[1,3,2],2,2),           # R28
    ([3,5],[1,0],2,0),               # R29
    ([3],[2],2,0),                   # R30
    ([3,0,4],[2,3,2],3,0),           # R31
    ([3],[2],3,0),                   # R32
    ([3],[0],3,2),                   # R33
    ([3,0,4],[1,2,2],3,1),           # R34
    ([3,4],[1,1],3,1),               # R35
]

RULE_LABELS = [
    "R1:  temp=cold, act=low, comfort=warm → heater=high",
    "R2:  temp=cold, hum=low, act=low, comfort=normal → heater=medium",
    "R3:  temp=cold, hum=high, act=low, comfort=normal → heater=low",
    "R4:  temp=cold, act=low, comfort=cool → heater=low",
    "R5:  temp=cold, act=high → heater=low",
    "R6:  temp=comfortable, act=low, comfort=warm → heater=medium",
    "R7:  temp=comfortable, comfort=normal → heater=low",
    "R8:  temp=comfortable, comfort=cool → heater=off",
    "R9:  temp=warm, hum=high, act=low, comfort=warm → heater=low",
    "R10: temp=warm, comfort=cool → heater=off",
    "R11: temp=hot → heater=off",
    "R12: temp=cold, hum=high, act=high, comfort=warm → heater=medium",
    "R13: temp=hot, hum=high, act=high, comfort=cool → fan=high",
    "R14: temp=hot, comfort=cool → fan=medium",
    "R15: temp=hot, comfort=normal → fan=low",
    "R16: temp=hot, comfort=warm → fan=off",
    "R17: temp=warm, hum=high, act=high, comfort=cool → fan=medium",
    "R18: temp=warm, hum=high, comfort=cool → fan=low",
    "R19: temp=comfortable, act=high, comfort=cool → fan=low",
    "R20: temp=cold → fan=off",
    "R21: light=dark, time=morning → dimmer=bright",
    "R22: light=dark, time=afternoon, comfort=warm → dimmer=medium",
    "R23: light=dark, time=afternoon, comfort=cool → dimmer=off",
    "R24: light=dark, time=evening, comfort=warm → dimmer=bright",
    "R25: light=dark, time=evening, comfort=cool → dimmer=medium",
    "R26: light=dark, time=night → dimmer=low",
    "R27: light=dim, time=morning, comfort=warm → dimmer=medium",
    "R28: light=dim, time=evening, comfort=warm → dimmer=medium",
    "R29: light=dim, comfort=cool → dimmer=off",
    "R30: light=bright → dimmer=off",
    "R31: light=bright, temp=hot, time=afternoon → blinds=closed",
    "R32: light=bright → blinds=closed",
    "R33: light=dark → blinds=open",
    "R34: light=dim, temp=warm, time=afternoon → blinds=partial",
    "R35: light=dim, time=morning → blinds=partial",
]

def fast_flc(X, params):
    """Vectorised FLC over N examples. X: (N,6), returns (N,4)."""
    N = X.shape[0]
    mu = [[eval_mf(X[:,vi], mt, mp) for mt, mp in params[v]]
          for vi, v in enumerate(INPUT_VARS)]
    centroids = [[mf_centroid(mt,mp) for mt,mp in params[v]] for v in OUTPUT_VARS]
    ws = np.zeros((N,4)); wt = np.zeros((N,4))
    for col_idx, set_idx, oc, os_ in RULES:
        f = mu[col_idx[0]][set_idx[0]].copy()
        for ci, si in zip(col_idx[1:], set_idx[1:]):
            f = np.minimum(f, mu[ci][si])
        ws[:,oc] += f * centroids[oc][os_]
        wt[:,oc] += f
    return np.where(wt > 1e-10, ws / np.maximum(wt,1e-10), 50.0)

def compute_firing_strengths(input_vec, params):
    """Firing strength for each rule at a single input point."""
    x = np.array(input_vec, dtype=float).reshape(1,-1)
    mu = [[eval_mf(x[:,vi], mt, mp)[0] for mt, mp in params[v]]
          for vi, v in enumerate(INPUT_VARS)]
    strengths = []
    for col_idx, set_idx, oc, os_ in RULES:
        f = mu[col_idx[0]][set_idx[0]]
        for ci, si in zip(col_idx[1:], set_idx[1:]):
            f = min(f, mu[ci][si])
        strengths.append(float(f))
    return strengths

def compute_memberships(input_vec, params):
    """Input memberships for a single point."""
    x = np.array(input_vec, dtype=float).reshape(1,-1)
    result = {}
    for vi, v in enumerate(INPUT_VARS):
        sets = [mt for mt, _ in params[v]]
        vals = [eval_mf(x[:,vi], mt, mp)[0] for mt, mp in params[v]]
        result[v] = list(zip(sets, vals))
    return result

# ── SCENARIO ANALYSIS ─────────────────────────────────────────────
SCENARIOS = [
    {"name":   "Cold winter morning, resting",
     "inputs": [12.0, 40.0, 0.1, 200.0, 9.0, 0.85],
     "desc":   "temp=12°C, hum=40%, act=0.1, light=200lux, time=9h, comfort=0.85"},
    {"name":   "Hot summer afternoon, active",
     "inputs": [32.0, 75.0, 0.8, 950.0, 14.0, 0.1],
     "desc":   "temp=32°C, hum=75%, act=0.8, light=950lux, time=14h, comfort=0.1"},
    {"name":   "Cool evening, moderate activity",
     "inputs": [18.0, 55.0, 0.5, 150.0, 19.0, 0.5],
     "desc":   "temp=18°C, hum=55%, act=0.5, light=150lux, time=19h, comfort=0.5"},
]

print("\n[Scenarios] Computing rule firing strengths from FLC...")
scenario_output_lines = []
for sc in SCENARIOS:
    xvec = sc["inputs"]
    strengths = compute_firing_strengths(xvec, INIT_PARAMS)
    outputs   = fast_flc(np.array(xvec).reshape(1,-1), INIT_PARAMS)[0]
    memberships = compute_memberships(xvec, INIT_PARAMS)

    line = f"\nScenario: {sc['name']}"
    print(line); scenario_output_lines.append(line)
    line = f"  Inputs: {sc['desc']}"
    print(line); scenario_output_lines.append(line)

    line = "  Input memberships (>0.05):"
    print(line); scenario_output_lines.append(line)
    var_short = ['temp','hum','act','light','time','comfort']
    for vi, v in enumerate(INPUT_VARS):
        pairs = memberships[v]
        active = [(mt, val) for mt, val in pairs if val > 0.05]
        if active:
            parts = ", ".join(f"{mt}={val:.3f}" for mt, val in active)
            line = f"    {var_short[vi]:8s}: {parts}"
            print(line); scenario_output_lines.append(line)

    line = "  Top activated rules:"
    print(line); scenario_output_lines.append(line)
    ranked = sorted(enumerate(strengths), key=lambda kv: -kv[1])
    for ridx, strength in ranked:
        if strength > 0.01:
            line = f"    {RULE_LABELS[ridx]:<60s}  α={strength:.4f}"
            print(line); scenario_output_lines.append(line)

    outs = f"  Outputs: heater={outputs[0]:.1f}%, fan={outputs[1]:.1f}%, dimmer={outputs[2]:.1f}%, blinds={outputs[3]:.1f}%"
    print(outs); scenario_output_lines.append(outs)

with open(f"{DATA_DIR}/task2_scenarios.txt", 'w') as f:
    f.write("\n".join(scenario_output_lines))
print(f"\n  Scenario firing strengths saved to data/task2_scenarios.txt")

# ── Part 2: GA for FLC optimisation ─────────────────────────────
def fitness(chrom, X_in, Y_out):
    p   = chrom_to_params(chrom)
    pr  = fast_flc(X_in, p)
    rms = np.sqrt(np.mean((pr - Y_out)**2, axis=0))
    return -float(np.mean(rms))

def per_output_rmse(chrom, X_in, Y_out):
    p  = chrom_to_params(chrom)
    pr = fast_flc(X_in, p)
    return np.sqrt(np.mean((pr - Y_out)**2, axis=0))

def constrain(chrom):
    p = chrom_to_params(chrom)
    out = []
    for var in VAR_ORDER:
        lo, hi = BOUNDS_MAP[var]
        for mtype, mp in p[var]:
            out.extend(np.clip(np.sort(mp), lo, hi))
    return np.array(out)

def tournament_select(pop, scores, k=3):
    idx = np.random.choice(len(pop), k, replace=False)
    return pop[idx[np.argmax(scores[idx])]]

def run_ga_flc(seed, n_gen=100, pop_size=50):
    """Real-valued GA for FLC MF optimisation."""
    rng_ga = np.random.RandomState(seed)
    init_c = params_to_chrom(INIT_PARAMS)
    pop    = np.array([constrain(init_c + rng_ga.randn(CHROM_LEN)*0.3) for _ in range(pop_size)])
    pop[0] = init_c.copy()

    scores   = np.array([fitness(c, inputs_train, outputs_train) for c in pop])
    best_hist = [float(np.max(scores))]

    for g in range(n_gen):
        new_pop = []
        for _ in range(pop_size):
            p1 = tournament_select(pop, scores, k=3)
            p2 = tournament_select(pop, scores, k=3)
            # single-point crossover
            if rng_ga.rand() < 0.8:
                pt = rng_ga.randint(1, CHROM_LEN)
                child = np.concatenate([p1[:pt], p2[pt:]])
            else:
                child = p1.copy()
            # Gaussian mutation
            mask = rng_ga.rand(CHROM_LEN) < 0.05
            child[mask] += rng_ga.randn(mask.sum()) * 0.5
            child = constrain(child)
            new_pop.append(child)

        new_pop    = np.array(new_pop)
        new_scores = np.array([fitness(c, inputs_train, outputs_train) for c in new_pop])

        # elitism: preserve best
        best_idx = np.argmax(scores)
        worst_new = np.argmin(new_scores)
        if scores[best_idx] > new_scores[worst_new]:
            new_pop[worst_new]    = pop[best_idx].copy()
            new_scores[worst_new] = scores[best_idx]

        pop    = new_pop
        scores = new_scores
        best_hist.append(float(np.max(scores)))

    best_chrom = pop[np.argmax(scores)]
    return best_chrom, best_hist

POP, NGEN = 50, 100
N_GA_RUNS = 5

print(f"\n[Part 2] Running GA ({N_GA_RUNS} independent runs, pop={POP}, gen={NGEN})...")
init_c     = params_to_chrom(INIT_PARAMS)
init_rmse_train = per_output_rmse(init_c, inputs_train, outputs_train)
init_rmse_test  = per_output_rmse(init_c, inputs_test,  outputs_test)
print(f"  Initial train RMSE: heater={init_rmse_train[0]:.2f} fan={init_rmse_train[1]:.2f} "
      f"dimmer={init_rmse_train[2]:.2f} blinds={init_rmse_train[3]:.2f} "
      f"mean={np.mean(init_rmse_train):.3f}")
print(f"  Initial test  RMSE: heater={init_rmse_test[0]:.2f} fan={init_rmse_test[1]:.2f} "
      f"dimmer={init_rmse_test[2]:.2f} blinds={init_rmse_test[3]:.2f} "
      f"mean={np.mean(init_rmse_test):.3f}")

all_best_chroms = []
all_train_rmse  = []
all_test_rmse   = []
all_hist        = []

for run in range(N_GA_RUNS):
    bchrom, hist = run_ga_flc(seed=run*17+42)
    tr = per_output_rmse(bchrom, inputs_train, outputs_train)
    te = per_output_rmse(bchrom, inputs_test,  outputs_test)
    all_best_chroms.append(bchrom)
    all_train_rmse.append(tr)
    all_test_rmse.append(te)
    all_hist.append([-v for v in hist])  # convert back to RMSE (positive)
    print(f"  Run {run+1}: train RMSE={np.mean(tr):.3f}  test RMSE={np.mean(te):.3f}")

train_means = np.array([np.mean(r) for r in all_train_rmse])
test_means  = np.array([np.mean(r) for r in all_test_rmse])
best_run    = int(np.argmin(train_means))
best_chrom  = all_best_chroms[best_run]
opt_params  = chrom_to_params(best_chrom)

best_tr = all_train_rmse[best_run]
best_te = all_test_rmse[best_run]

print(f"\n  Best run (run {best_run+1}):")
print(f"    Train RMSE: heater={best_tr[0]:.2f} fan={best_tr[1]:.2f} "
      f"dimmer={best_tr[2]:.2f} blinds={best_tr[3]:.2f} mean={np.mean(best_tr):.3f}")
print(f"    Test  RMSE: heater={best_te[0]:.2f} fan={best_te[1]:.2f} "
      f"dimmer={best_te[2]:.2f} blinds={best_te[3]:.2f} mean={np.mean(best_te):.3f}")
print(f"  Train RMSE over {N_GA_RUNS} runs: mean={train_means.mean():.3f} std={train_means.std():.3f}")
print(f"  Test  RMSE over {N_GA_RUNS} runs: mean={test_means.mean():.3f}  std={test_means.std():.3f}")

# ── GA convergence plot (mean ± 1 SD across 5 runs) ──────────────
hist_arr = np.array([h[:NGEN+1] for h in all_hist])
mn = hist_arr.mean(axis=0)
sd = hist_arr.std(axis=0)
xs = np.arange(NGEN+1)

fig, ax = plt.subplots(figsize=(9,5))
ax.plot(xs, mn, color='#1565C0', lw=2, label=f'GA mean RMSE ({N_GA_RUNS} runs)')
ax.fill_between(xs, mn-sd, mn+sd, color='#1565C0', alpha=0.15, label='±1 SD')
ax.axhline(np.mean(init_rmse_train), color='#F44336', ls='--', lw=1.5,
           label=f'Initial RMSE ({np.mean(init_rmse_train):.2f})')
ax.set_xlabel("Generation", fontsize=11)
ax.set_ylabel("RMSE (lower = better)", fontsize=11)
ax.set_title("GA Fitness Convergence — FLC Membership Function Optimisation", fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/task2_ga_convergence.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: task2_ga_convergence.png")

# ── Before/after control surface ─────────────────────────────────
print("[Part 2] Before/after control surface...")

temperature2  = ctrl.Antecedent(np.linspace(10,40,300), 'temperature')
humidity2     = ctrl.Antecedent(np.linspace(20,90,300), 'humidity')
activity2     = ctrl.Antecedent(np.linspace(0,1,300),   'activity')
light_level2  = ctrl.Antecedent(np.linspace(0,1000,300),'light_level')
time_of_day2  = ctrl.Antecedent(np.linspace(0,24,300),  'time_of_day')
comfort_pref2 = ctrl.Antecedent(np.linspace(0,1,300),   'comfort_preference')
heater2       = ctrl.Consequent(np.linspace(0,100,300),  'heater')
fan2          = ctrl.Consequent(np.linspace(0,100,300),  'fan')
dimmer2       = ctrl.Consequent(np.linspace(0,100,300),  'dimmer')
blinds2       = ctrl.Consequent(np.linspace(0,100,300),  'blinds')

temperature2['cold']        = fuzz.trapmf(temperature2.universe, [10,10,14,19])
temperature2['comfortable'] = fuzz.trimf(temperature2.universe,  [16,22,27])
temperature2['warm']        = fuzz.trimf(temperature2.universe,  [24,28,33])
temperature2['hot']         = fuzz.trapmf(temperature2.universe, [30,36,40,40])
humidity2['low']    = fuzz.trapmf(humidity2.universe, [20,20,30,40])
humidity2['medium'] = fuzz.trimf(humidity2.universe,  [35,50,65])
humidity2['high']   = fuzz.trapmf(humidity2.universe, [60,70,90,90])
activity2['low']    = fuzz.trapmf(activity2.universe, [0,0,.2,.35])
activity2['medium'] = fuzz.trimf(activity2.universe,  [.25,.5,.75])
activity2['high']   = fuzz.trapmf(activity2.universe, [.65,.8,1,1])
light_level2['dark']   = fuzz.trapmf(light_level2.universe, [0,0,100,250])
light_level2['dim']    = fuzz.trimf(light_level2.universe,  [150,400,650])
light_level2['bright'] = fuzz.trapmf(light_level2.universe, [550,750,1000,1000])
time_of_day2['night']     = fuzz.trapmf(time_of_day2.universe, [0,0,5,7])
time_of_day2['morning']   = fuzz.trimf(time_of_day2.universe,  [6,9,12])
time_of_day2['afternoon'] = fuzz.trimf(time_of_day2.universe,  [11,14,18])
time_of_day2['evening']   = fuzz.trapmf(time_of_day2.universe, [16,20,24,24])
comfort_pref2['cool']   = fuzz.trapmf(comfort_pref2.universe, [0,0,.2,.4])
comfort_pref2['normal'] = fuzz.trimf(comfort_pref2.universe,  [.3,.5,.7])
comfort_pref2['warm']   = fuzz.trapmf(comfort_pref2.universe, [.6,.8,1,1])
heater2['off']   = fuzz.trapmf(heater2.universe, [0,0,5,15])
heater2['low']   = fuzz.trimf(heater2.universe,  [10,30,50])
heater2['medium']= fuzz.trimf(heater2.universe,  [40,60,80])
heater2['high']  = fuzz.trapmf(heater2.universe, [70,85,100,100])
fan2['off']   = fuzz.trapmf(fan2.universe, [0,0,5,15])
fan2['low']   = fuzz.trimf(fan2.universe,  [10,30,50])
fan2['medium']= fuzz.trimf(fan2.universe,  [40,60,80])
fan2['high']  = fuzz.trapmf(fan2.universe, [70,85,100,100])
dimmer2['off']   = fuzz.trapmf(dimmer2.universe, [0,0,5,15])
dimmer2['low']   = fuzz.trimf(dimmer2.universe,  [10,30,50])
dimmer2['medium']= fuzz.trimf(dimmer2.universe,  [40,60,80])
dimmer2['bright']= fuzz.trapmf(dimmer2.universe, [70,85,100,100])
blinds2['closed']        = fuzz.trapmf(blinds2.universe, [0,0,5,20])
blinds2['partially_open']= fuzz.trimf(blinds2.universe,  [15,50,85])
blinds2['open']          = fuzz.trapmf(blinds2.universe, [70,90,100,100])

r2 = [None]*36
r2[1]  = ctrl.Rule(temperature2['cold'] & activity2['low'] & comfort_pref2['warm'],   heater2['high'])
r2[2]  = ctrl.Rule(temperature2['cold'] & humidity2['low'] & activity2['low'] & comfort_pref2['normal'], heater2['medium'])
r2[3]  = ctrl.Rule(temperature2['cold'] & humidity2['high'] & activity2['low'] & comfort_pref2['normal'], heater2['low'])
r2[4]  = ctrl.Rule(temperature2['cold'] & activity2['low'] & comfort_pref2['cool'],   heater2['low'])
r2[5]  = ctrl.Rule(temperature2['cold'] & activity2['high'],                           heater2['low'])
r2[6]  = ctrl.Rule(temperature2['comfortable'] & activity2['low'] & comfort_pref2['warm'], heater2['medium'])
r2[7]  = ctrl.Rule(temperature2['comfortable'] & comfort_pref2['normal'],             heater2['low'])
r2[8]  = ctrl.Rule(temperature2['comfortable'] & comfort_pref2['cool'],               heater2['off'])
r2[9]  = ctrl.Rule(temperature2['warm'] & humidity2['high'] & activity2['low'] & comfort_pref2['warm'], heater2['low'])
r2[10] = ctrl.Rule(temperature2['warm'] & comfort_pref2['cool'],                      heater2['off'])
r2[11] = ctrl.Rule(temperature2['hot'],                                                heater2['off'])
r2[12] = ctrl.Rule(temperature2['cold'] & humidity2['high'] & activity2['high'] & comfort_pref2['warm'], heater2['medium'])
r2[13] = ctrl.Rule(temperature2['hot'] & humidity2['high'] & activity2['high'] & comfort_pref2['cool'], fan2['high'])
r2[14] = ctrl.Rule(temperature2['hot'] & comfort_pref2['cool'],  fan2['medium'])
r2[15] = ctrl.Rule(temperature2['hot'] & comfort_pref2['normal'],fan2['low'])
r2[16] = ctrl.Rule(temperature2['hot'] & comfort_pref2['warm'],  fan2['off'])
r2[17] = ctrl.Rule(temperature2['warm'] & humidity2['high'] & activity2['high'] & comfort_pref2['cool'], fan2['medium'])
r2[18] = ctrl.Rule(temperature2['warm'] & humidity2['high'] & comfort_pref2['cool'], fan2['low'])
r2[19] = ctrl.Rule(temperature2['comfortable'] & activity2['high'] & comfort_pref2['cool'], fan2['low'])
r2[20] = ctrl.Rule(temperature2['cold'],                          fan2['off'])
r2[21] = ctrl.Rule(light_level2['dark'] & time_of_day2['morning'],                               dimmer2['bright'])
r2[22] = ctrl.Rule(light_level2['dark'] & time_of_day2['afternoon'] & comfort_pref2['warm'],     dimmer2['medium'])
r2[23] = ctrl.Rule(light_level2['dark'] & time_of_day2['afternoon'] & comfort_pref2['cool'],     dimmer2['off'])
r2[24] = ctrl.Rule(light_level2['dark'] & time_of_day2['evening'] & comfort_pref2['warm'],       dimmer2['bright'])
r2[25] = ctrl.Rule(light_level2['dark'] & time_of_day2['evening'] & comfort_pref2['cool'],       dimmer2['medium'])
r2[26] = ctrl.Rule(light_level2['dark'] & time_of_day2['night'],                                 dimmer2['low'])
r2[27] = ctrl.Rule(light_level2['dim'] & time_of_day2['morning'] & comfort_pref2['warm'],        dimmer2['medium'])
r2[28] = ctrl.Rule(light_level2['dim'] & time_of_day2['evening'] & comfort_pref2['warm'],        dimmer2['medium'])
r2[29] = ctrl.Rule(light_level2['dim'] & comfort_pref2['cool'],                                   dimmer2['off'])
r2[30] = ctrl.Rule(light_level2['bright'],                                                         dimmer2['off'])
r2[31] = ctrl.Rule(light_level2['bright'] & temperature2['hot'] & time_of_day2['afternoon'],     blinds2['closed'])
r2[32] = ctrl.Rule(light_level2['bright'],                                                         blinds2['closed'])
r2[33] = ctrl.Rule(light_level2['dark'],                                                           blinds2['open'])
r2[34] = ctrl.Rule(light_level2['dim'] & temperature2['warm'] & time_of_day2['afternoon'],       blinds2['partially_open'])
r2[35] = ctrl.Rule(light_level2['dim'] & time_of_day2['morning'],                                blinds2['partially_open'])

sys2 = ctrl.ControlSystem([r2[i] for i in range(1,36)])
sim2 = ctrl.ControlSystemSimulation(sys2)

def skfuzzy_heater(t, c):
    try:
        sim2.input['temperature'] = float(t)
        sim2.input['humidity']    = 55.0
        sim2.input['activity']    = 0.3
        sim2.input['light_level'] = 300.0
        sim2.input['time_of_day'] = 12.0
        sim2.input['comfort_preference'] = float(c)
        sim2.compute()
        return sim2.output.get('heater', 5.0)
    except Exception:
        return 5.0

N_surf = 20
T_v = np.linspace(10, 40, N_surf)
C_v = np.linspace(0,  1,  N_surf)
Z_before = np.zeros((N_surf, N_surf))
Z_after  = np.zeros((N_surf, N_surf))

batch_inputs = np.array([[t,55,0.3,300,12,c] for t in T_v for c in C_v])
after_batch  = fast_flc(batch_inputs, opt_params)[:,0].reshape(N_surf, N_surf)

for i, t in enumerate(T_v):
    for j, c in enumerate(C_v):
        Z_before[i,j] = skfuzzy_heater(t, c)
Z_after = after_batch

fig, axes = plt.subplots(1, 2, figsize=(13,5), subplot_kw={'projection':'3d'})
fig.suptitle("Heater Control Surface: Before vs After GA Optimisation", fontsize=12, fontweight='bold')
for ax, Z, title in [(axes[0],Z_before,'Before Optimisation'),(axes[1],Z_after,'After Optimisation')]:
    X,Y = np.meshgrid(T_v, C_v)
    ax.plot_surface(X, Y, Z.T, cmap='YlOrRd', alpha=0.85, edgecolor='none')
    ax.set_xlabel("Temperature (°C)", fontsize=8)
    ax.set_ylabel("Comfort Preference", fontsize=8)
    ax.set_zlabel("Heater (%)", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=7)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/task2_before_after_surface.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: task2_before_after_surface.png")

# ─────────────────────────────────────────────────────────────────
# PART 3: CEC'2005 BENCHMARK COMPARISON
# ─────────────────────────────────────────────────────────────────
print("\n[Part 3] CEC'2005 benchmarks (GA vs PSO, 15 runs each)...")
print("  Note: rotation matrix for F10 uses fixed-seed QR decomposition (official data files not available)")

def f1(x, o):
    z = x - o
    return float(np.sum(z**2) - 450)

def make_rot(D, seed):
    Q, _ = np.linalg.qr(np.random.RandomState(seed).randn(D,D))
    return Q

def f10(x, o, R):
    z = R @ (x - o)
    return float(np.sum(z**2 - 10*np.cos(2*np.pi*z) + 10) - 330)

def run_real_ga(func, lo, hi, D, maxeval, pop_size=50):
    """Real-valued GA: tournament selection, arithmetic crossover, Gaussian mutation, elitism."""
    pop = np.random.uniform(lo, hi, (pop_size, D))
    fit = np.array([func(xi) for xi in pop])
    hist = [float(np.min(fit))]
    ev = pop_size
    sigma = 0.1 * (hi - lo)

    while ev < maxeval:
        new_pop = []
        new_fit = []
        for _ in range(pop_size):
            # Tournament selection (k=3) — minimisation
            idx1 = np.random.choice(pop_size, 3, replace=False)
            p1   = pop[idx1[np.argmin(fit[idx1])]]
            idx2 = np.random.choice(pop_size, 3, replace=False)
            p2   = pop[idx2[np.argmin(fit[idx2])]]
            # Arithmetic crossover
            if np.random.rand() < 0.8:
                alpha = np.random.rand()
                child = alpha * p1 + (1 - alpha) * p2
            else:
                child = p1.copy()
            # Gaussian mutation (rate 0.05 per gene)
            mask = np.random.rand(D) < 0.05
            child[mask] += np.random.randn(mask.sum()) * sigma
            child = np.clip(child, lo, hi)
            new_pop.append(child)
            f_ = func(child)
            new_fit.append(f_)
            ev += 1
            if ev >= maxeval:
                break

        new_pop = np.array(new_pop[:pop_size])
        new_fit = np.array(new_fit[:pop_size])
        # Elitism: keep best from previous generation
        best_prev = np.argmin(fit)
        worst_new = np.argmax(new_fit)
        if fit[best_prev] < new_fit[worst_new]:
            new_pop[worst_new] = pop[best_prev].copy()
            new_fit[worst_new] = fit[best_prev]

        pop = new_pop
        fit = new_fit
        hist.append(float(np.min(fit)))
        if ev >= maxeval:
            break

    return float(np.min(fit)), hist

def run_pso(func, lo, hi, D, maxeval, n=40):
    """Particle Swarm Optimisation: w=0.7, c1=c2=1.5, velocity clamping."""
    vmax = (hi - lo) * 0.2
    pos  = np.random.uniform(lo, hi, (n, D))
    vel  = np.random.uniform(-vmax, vmax, (n, D))
    pb   = pos.copy()
    pf   = np.array([func(p) for p in pos])
    gi   = np.argmin(pf); gf = pf[gi]; gb = pb[gi].copy()
    hist = [float(gf)]
    ev   = n; w, c1, c2 = 0.7, 1.5, 1.5
    while ev < maxeval:
        r1, r2 = np.random.rand(n, D), np.random.rand(n, D)
        vel    = np.clip(w*vel + c1*r1*(pb-pos) + c2*r2*(gb-pos), -vmax, vmax)
        pos    = np.clip(pos+vel, lo, hi)
        for i in range(n):
            f_ = func(pos[i]); ev += 1
            if f_ < pf[i]:
                pb[i] = pos[i].copy(); pf[i] = f_
                if f_ < gf: gb = pos[i].copy(); gf = f_
        hist.append(float(gf))
        if ev >= maxeval:
            break
    return float(gf), hist

N_RUNS  = 15
DIMS    = [2, 10]
MAX_EV  = 10000   # × D

results     = {}
convergence = {}

for D in DIMS:
    rng_cec = np.random.RandomState(D * 7)
    o1  = rng_cec.uniform(-50, 50, D)
    o10 = rng_cec.uniform(-2.5, 2.5, D)   # shift inside [-5,5]
    R10 = make_rot(D, seed=D)
    mev = MAX_EV * D

    # F1 bounds: [-100, 100]^D; F10 bounds: [-5, 5]^D
    for fname, func, lo_b, hi_b in [
        ('F1_Sphere',     lambda x, _o=o1:  f1(x, _o),       -100.0, 100.0),
        ('F10_Rastrigin', lambda x, _o=o10, _R=R10: f10(x, _o, _R), -5.0,   5.0),
    ]:
        for oname, runner in [('GA', run_real_ga), ('PSO', run_pso)]:
            key = (fname, oname, D)
            finals, convs = [], []
            for run in range(N_RUNS):
                np.random.seed(run * 100 + D)
                val, hist = runner(func, lo_b, hi_b, D, mev)
                finals.append(val); convs.append(hist)
            results[key]     = finals
            convergence[key] = convs
            m, s, b, w_ = np.mean(finals), np.std(finals), np.min(finals), np.max(finals)
            print(f"  {fname:16s} {oname} D={D:2d} | mean={m:11.3f} std={s:9.3f} "
                  f"best={b:11.3f} worst={w_:11.3f}")

print("\n[Part 3] Wilcoxon tests (GA vs PSO):")
wilcoxon_results = {}
for fname in ['F1_Sphere', 'F10_Rastrigin']:
    for D in DIMS:
        ga_r  = results[(fname, 'GA',  D)]
        pso_r = results[(fname, 'PSO', D)]
        diffs = np.array(ga_r) - np.array(pso_r)
        if np.all(diffs == 0):
            wilcoxon_results[(fname, D)] = (None, False)
            print(f"  {fname} D={D}: N/A — all paired differences are zero")
        else:
            try:
                _, p = wilcoxon(ga_r, pso_r)
                wilcoxon_results[(fname, D)] = (p, p < 0.05)
                sig = "significant" if p < 0.05 else "not significant"
                print(f"  {fname} D={D}: p={p:.4f} ({sig})")
            except ValueError as e:
                wilcoxon_results[(fname, D)] = (None, False)
                print(f"  {fname} D={D}: {e}")

# ── 4-panel convergence plot ─────────────────────────────────────
print("[Part 3] Plotting convergence...")

fig, axes = plt.subplots(2, 2, figsize=(13,9))
fig.suptitle("CEC'2005 Benchmark: GA vs PSO Convergence (15 runs, mean ±1 SD)",
             fontsize=12, fontweight='bold')

configs = [
    ('F1_Sphere',      2,  axes[0,0], 'F1 Shifted Sphere (D=2)'),
    ('F1_Sphere',      10, axes[0,1], 'F1 Shifted Sphere (D=10)'),
    ('F10_Rastrigin',  2,  axes[1,0], 'F10 Shifted Rotated Rastrigin (D=2)'),
    ('F10_Rastrigin',  10, axes[1,1], 'F10 Shifted Rotated Rastrigin (D=10)'),
]

for fname, D, ax, title in configs:
    mev = MAX_EV * D
    for opt, col, lbl in [('GA','#1565C0','GA'), ('PSO','#C62828','PSO')]:
        convs = convergence[(fname, opt, D)]
        L     = min(len(c) for c in convs)
        mat   = np.array([c[:L] for c in convs])
        mn    = np.mean(mat, 0); sd = np.std(mat, 0)
        xs    = np.linspace(0, mev, L)
        ax.plot(xs, mn, color=col, lw=2, label=lbl)
        ax.fill_between(xs, mn-sd, mn+sd, color=col, alpha=0.15)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel("Function Evaluations", fontsize=9)
    ax.set_ylabel("Best Fitness", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/task2_cec_convergence.png", dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: task2_cec_convergence.png")

# ── Summary table ────────────────────────────────────────────────
print("\n[Summary] Results table:")
print(f"{'Function':<20}{'Opt':>5}{'D':>4} {'Mean':>12} {'Std':>10} {'Best':>12} {'Worst':>12}")
print("-"*75)
rows = []
for fname in ['F1_Sphere', 'F10_Rastrigin']:
    for D in DIMS:
        for opt in ['GA', 'PSO']:
            r = results[(fname, opt, D)]
            row = (fname, opt, D, np.mean(r), np.std(r), np.min(r), np.max(r))
            rows.append(row)
            print(f"{fname:<20}{opt:>5}{D:>4} {np.mean(r):>12.3f} {np.std(r):>10.3f} "
                  f"{np.min(r):>12.3f} {np.max(r):>12.3f}")

# ── Save summary ─────────────────────────────────────────────────
init_tr_mean = float(np.mean(init_rmse_train))
init_te_mean = float(np.mean(init_rmse_test))
best_tr_mean = float(np.mean(best_tr))
best_te_mean = float(np.mean(best_te))

with open(f"{DATA_DIR}/task2_summary.txt", 'w') as f:
    f.write(f"Chromosome length: {CHROM_LEN}\n")
    f.write(f"GA runs: {N_GA_RUNS}, train/test split: {N_TRAIN}/{len(test_data)}\n\n")
    f.write("Per-output RMSE — Initial FLC:\n")
    f.write(f"  train: heater={init_rmse_train[0]:.2f} fan={init_rmse_train[1]:.2f} "
            f"dimmer={init_rmse_train[2]:.2f} blinds={init_rmse_train[3]:.2f} mean={init_tr_mean:.3f}\n")
    f.write(f"  test:  heater={init_rmse_test[0]:.2f} fan={init_rmse_test[1]:.2f} "
            f"dimmer={init_rmse_test[2]:.2f} blinds={init_rmse_test[3]:.2f} mean={init_te_mean:.3f}\n\n")
    f.write(f"Per-output RMSE — GA-optimised FLC (best of {N_GA_RUNS} runs):\n")
    f.write(f"  train: heater={best_tr[0]:.2f} fan={best_tr[1]:.2f} "
            f"dimmer={best_tr[2]:.2f} blinds={best_tr[3]:.2f} mean={best_tr_mean:.3f}\n")
    f.write(f"  test:  heater={best_te[0]:.2f} fan={best_te[1]:.2f} "
            f"dimmer={best_te[2]:.2f} blinds={best_te[3]:.2f} mean={best_te_mean:.3f}\n\n")
    f.write(f"GA mean train RMSE across {N_GA_RUNS} runs: {train_means.mean():.3f} ± {train_means.std():.3f}\n")
    f.write(f"GA mean test  RMSE across {N_GA_RUNS} runs: {test_means.mean():.3f} ± {test_means.std():.3f}\n\n")
    f.write("CEC Results:\n")
    for fname, opt, D, m, s, b, w_ in rows:
        f.write(f"{fname},{opt},{D},{m:.3f},{s:.3f},{b:.3f},{w_:.3f}\n")
    f.write("\nWilcoxon:\n")
    for (fname, D), (p, sig) in wilcoxon_results.items():
        f.write(f"{fname},D={D},p={p},sig={sig}\n")

print(f"\nSaved summary to {DATA_DIR}/task2_summary.txt")
print("\n" + "=" * 60)
print("PARTS 2 & 3 DONE")
print("=" * 60)
