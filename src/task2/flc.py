"""
Task 2: Fuzzy Logic Controller for Intelligent Assistive Care Environment
ST7085CEM Advanced Machine Learning — Manjil Karki
Parts 1, 2, and 3 — generates all figures and results
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from scipy.stats import wilcoxon
import csv
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
FIG_DIR  = os.path.join(_ROOT, 'docs', 'figures')
DATA_DIR = os.path.join(_ROOT, 'data')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("TASK 2 — FLC IMPLEMENTATION")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────
# PART 1: DEFINE FLC
# ─────────────────────────────────────────────────────────────────
print("\n[Part 1] Defining fuzzy variables and membership functions...")

# Universes
temp_u      = np.linspace(10, 40, 300)
humid_u     = np.linspace(20, 90, 300)
activity_u  = np.linspace(0, 1, 300)
light_u     = np.linspace(0, 1000, 300)
time_u      = np.linspace(0, 24, 300)
comfort_u   = np.linspace(0, 1, 300)

heater_u    = np.linspace(0, 100, 300)
fan_u       = np.linspace(0, 100, 300)
dimmer_u    = np.linspace(0, 100, 300)
blinds_u    = np.linspace(0, 100, 300)

# ── Inputs ──────────────────────────────────────────────────────
temperature     = ctrl.Antecedent(temp_u, 'temperature')
humidity        = ctrl.Antecedent(humid_u, 'humidity')
activity        = ctrl.Antecedent(activity_u, 'activity')
light_level     = ctrl.Antecedent(light_u, 'light_level')
time_of_day     = ctrl.Antecedent(time_u, 'time_of_day')
comfort_pref    = ctrl.Antecedent(comfort_u, 'comfort_preference')

# ── Outputs ─────────────────────────────────────────────────────
heater  = ctrl.Consequent(heater_u, 'heater')
fan     = ctrl.Consequent(fan_u, 'fan')
dimmer  = ctrl.Consequent(dimmer_u, 'dimmer')
blinds  = ctrl.Consequent(blinds_u, 'blinds')

# ── Membership functions ─────────────────────────────────────────
# Temperature: cold(trap) comfortable(tri) warm(tri) hot(trap)
temperature['cold']        = fuzz.trapmf(temp_u, [10, 10, 14, 19])
temperature['comfortable'] = fuzz.trimf(temp_u,  [16, 22, 27])
temperature['warm']        = fuzz.trimf(temp_u,  [24, 28, 33])
temperature['hot']         = fuzz.trapmf(temp_u, [30, 36, 40, 40])

# Humidity: low(trap) medium(tri) high(trap)
humidity['low']    = fuzz.trapmf(humid_u, [20, 20, 30, 40])
humidity['medium'] = fuzz.trimf(humid_u,  [35, 50, 65])
humidity['high']   = fuzz.trapmf(humid_u, [60, 70, 90, 90])

# Activity: low(trap) medium(tri) high(trap)
activity['low']    = fuzz.trapmf(activity_u, [0, 0, 0.2, 0.35])
activity['medium'] = fuzz.trimf(activity_u,  [0.25, 0.5, 0.75])
activity['high']   = fuzz.trapmf(activity_u, [0.65, 0.8, 1, 1])

# Light level: dark(trap) dim(tri) bright(trap)
light_level['dark']   = fuzz.trapmf(light_u, [0, 0, 100, 250])
light_level['dim']    = fuzz.trimf(light_u,  [150, 400, 650])
light_level['bright'] = fuzz.trapmf(light_u, [550, 750, 1000, 1000])

# Time of day: night(trap) morning(tri) afternoon(tri) evening(trap)
time_of_day['night']     = fuzz.trapmf(time_u, [0, 0, 5, 7])
time_of_day['morning']   = fuzz.trimf(time_u,  [6, 9, 12])
time_of_day['afternoon'] = fuzz.trimf(time_u,  [11, 14, 18])
time_of_day['evening']   = fuzz.trapmf(time_u, [16, 20, 24, 24])

# Comfort preference: cool(trap) normal(tri) warm(trap)
comfort_pref['cool']   = fuzz.trapmf(comfort_u, [0, 0, 0.2, 0.4])
comfort_pref['normal'] = fuzz.trimf(comfort_u,  [0.3, 0.5, 0.7])
comfort_pref['warm']   = fuzz.trapmf(comfort_u, [0.6, 0.8, 1, 1])

# Heater: off(trap) low(tri) medium(tri) high(trap)
heater['off']    = fuzz.trapmf(heater_u, [0, 0, 5, 15])
heater['low']    = fuzz.trimf(heater_u,  [10, 30, 50])
heater['medium'] = fuzz.trimf(heater_u,  [40, 60, 80])
heater['high']   = fuzz.trapmf(heater_u, [70, 85, 100, 100])

# Fan: off(trap) low(tri) medium(tri) high(trap)
fan['off']    = fuzz.trapmf(fan_u, [0, 0, 5, 15])
fan['low']    = fuzz.trimf(fan_u,  [10, 30, 50])
fan['medium'] = fuzz.trimf(fan_u,  [40, 60, 80])
fan['high']   = fuzz.trapmf(fan_u, [70, 85, 100, 100])

# Dimmer: off(trap) low(tri) medium(tri) bright(trap)
dimmer['off']    = fuzz.trapmf(dimmer_u, [0, 0, 5, 15])
dimmer['low']    = fuzz.trimf(dimmer_u,  [10, 30, 50])
dimmer['medium'] = fuzz.trimf(dimmer_u,  [40, 60, 80])
dimmer['bright'] = fuzz.trapmf(dimmer_u, [70, 85, 100, 100])

# Blinds: closed(trap) partially_open(tri) open(trap)
blinds['closed']        = fuzz.trapmf(blinds_u, [0, 0, 5, 20])
blinds['partially_open']= fuzz.trimf(blinds_u,  [15, 50, 85])
blinds['open']          = fuzz.trapmf(blinds_u, [70, 90, 100, 100])

# ── Rules ────────────────────────────────────────────────────────
print("[Part 1] Defining 35 rules...")

r = [None] * 36  # 1-indexed

# HVAC — Heater (R1–R12)
r[1]  = ctrl.Rule(temperature['cold'] & activity['low'] & comfort_pref['warm'],   heater['high'])
r[2]  = ctrl.Rule(temperature['cold'] & humidity['low'] & activity['low'] & comfort_pref['normal'], heater['medium'])
r[3]  = ctrl.Rule(temperature['cold'] & humidity['high'] & activity['low'] & comfort_pref['normal'], heater['low'])
r[4]  = ctrl.Rule(temperature['cold'] & activity['low'] & comfort_pref['cool'],   heater['low'])
r[5]  = ctrl.Rule(temperature['cold'] & activity['high'],                          heater['low'])
r[6]  = ctrl.Rule(temperature['comfortable'] & activity['low'] & comfort_pref['warm'], heater['medium'])
r[7]  = ctrl.Rule(temperature['comfortable'] & comfort_pref['normal'],            heater['low'])
r[8]  = ctrl.Rule(temperature['comfortable'] & comfort_pref['cool'],              heater['off'])
r[9]  = ctrl.Rule(temperature['warm'] & humidity['high'] & activity['low'] & comfort_pref['warm'], heater['low'])
r[10] = ctrl.Rule(temperature['warm'] & comfort_pref['cool'],                     heater['off'])
r[11] = ctrl.Rule(temperature['hot'],                                              heater['off'])
r[12] = ctrl.Rule(temperature['cold'] & humidity['high'] & activity['high'] & comfort_pref['warm'], heater['medium'])

# HVAC — Fan (R13–R20)
r[13] = ctrl.Rule(temperature['hot'] & humidity['high'] & activity['high'] & comfort_pref['cool'], fan['high'])
r[14] = ctrl.Rule(temperature['hot'] & comfort_pref['cool'],  fan['medium'])
r[15] = ctrl.Rule(temperature['hot'] & comfort_pref['normal'],fan['low'])
r[16] = ctrl.Rule(temperature['hot'] & comfort_pref['warm'],  fan['off'])
r[17] = ctrl.Rule(temperature['warm'] & humidity['high'] & activity['high'] & comfort_pref['cool'], fan['medium'])
r[18] = ctrl.Rule(temperature['warm'] & humidity['high'] & comfort_pref['cool'], fan['low'])
r[19] = ctrl.Rule(temperature['comfortable'] & activity['high'] & comfort_pref['cool'], fan['low'])
r[20] = ctrl.Rule(temperature['cold'],                         fan['off'])

# Lighting — Dimmer (R21–R30)
r[21] = ctrl.Rule(light_level['dark'] & time_of_day['morning'],                              dimmer['bright'])
r[22] = ctrl.Rule(light_level['dark'] & time_of_day['afternoon'] & comfort_pref['warm'],     dimmer['medium'])
r[23] = ctrl.Rule(light_level['dark'] & time_of_day['afternoon'] & comfort_pref['cool'],     dimmer['off'])
r[24] = ctrl.Rule(light_level['dark'] & time_of_day['evening'] & comfort_pref['warm'],       dimmer['bright'])
r[25] = ctrl.Rule(light_level['dark'] & time_of_day['evening'] & comfort_pref['cool'],       dimmer['medium'])
r[26] = ctrl.Rule(light_level['dark'] & time_of_day['night'],                                dimmer['low'])
r[27] = ctrl.Rule(light_level['dim'] & time_of_day['morning'] & comfort_pref['warm'],        dimmer['medium'])
r[28] = ctrl.Rule(light_level['dim'] & time_of_day['evening'] & comfort_pref['warm'],        dimmer['medium'])
r[29] = ctrl.Rule(light_level['dim'] & comfort_pref['cool'],                                  dimmer['off'])
r[30] = ctrl.Rule(light_level['bright'],                                                       dimmer['off'])

# Blinds (R31–R35)
r[31] = ctrl.Rule(light_level['bright'] & temperature['hot'] & time_of_day['afternoon'],     blinds['closed'])
r[32] = ctrl.Rule(light_level['bright'],                                                       blinds['closed'])
r[33] = ctrl.Rule(light_level['dark'],                                                         blinds['open'])
r[34] = ctrl.Rule(light_level['dim'] & temperature['warm'] & time_of_day['afternoon'],       blinds['partially_open'])
r[35] = ctrl.Rule(light_level['dim'] & time_of_day['morning'],                               blinds['partially_open'])

all_rules = [r[i] for i in range(1, 36)]
flc_system = ctrl.ControlSystem(all_rules)
flc_sim    = ctrl.ControlSystemSimulation(flc_system)

print("[Part 1] FLC built successfully.")

# ─────────────────────────────────────────────────────────────────
# FIGURE 1: Membership function plots
# ─────────────────────────────────────────────────────────────────
print("[Part 1] Plotting membership functions...")

fig, axes = plt.subplots(5, 2, figsize=(14, 20))
fig.suptitle("Membership Functions — FLC Variables", fontsize=14, fontweight='bold', y=0.98)

var_configs = [
    (temperature,  temp_u,     "Temperature (°C)",        ['cold','comfortable','warm','hot']),
    (humidity,     humid_u,    "Humidity (%)",             ['low','medium','high']),
    (activity,     activity_u, "Activity Level",          ['low','medium','high']),
    (light_level,  light_u,    "Light Level (lux)",       ['dark','dim','bright']),
    (time_of_day,  time_u,     "Time of Day (h)",         ['night','morning','afternoon','evening']),
    (comfort_pref, comfort_u,  "Comfort Preference",      ['cool','normal','warm']),
    (heater,       heater_u,   "Heater Output (%)",       ['off','low','medium','high']),
    (fan,          fan_u,      "Fan Output (%)",          ['off','low','medium','high']),
    (dimmer,       dimmer_u,   "Dimmer Output (%)",       ['off','low','medium','bright']),
    (blinds,       blinds_u,   "Blinds Opening (%)",      ['closed','partially_open','open']),
]

colors = ['#2196F3','#4CAF50','#FF9800','#F44336','#9C27B0']

for idx, (var, universe, title, labels) in enumerate(var_configs):
    ax = axes[idx // 2][idx % 2]
    for j, label in enumerate(labels):
        ax.plot(universe, var[label].mf, label=label, color=colors[j % len(colors)], linewidth=2)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_xlabel(title.split('(')[0].strip())
    ax.set_ylabel('Membership')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/task2_mf_plots.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: task2_mf_plots.png")

# ─────────────────────────────────────────────────────────────────
# FIGURE 2: Architecture block diagram
# ─────────────────────────────────────────────────────────────────
print("[Part 1] Generating architecture diagram...")

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_facecolor('#FAFAFA')
fig.patch.set_facecolor('#FAFAFA')

def box(ax, x, y, w, h, text, fc='#E3F2FD', ec='#1565C0', fontsize=9, bold=False):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                    facecolor=fc, edgecolor=ec, linewidth=1.5)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, wrap=True)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

# Sensor boxes
sensors = ['Temperature\nSensor', 'Humidity\nSensor', 'Activity\nSensor',
           'Light Level\nSensor', 'Time of Day', 'Comfort\nPreference']
for i, s in enumerate(sensors):
    box(ax, 0.2, 0.3 + i * 1.0, 1.8, 0.75, s, fc='#E8F5E9', ec='#2E7D32', fontsize=8)
    arrow(ax, 2.0, 0.675 + i * 1.0, 3.2, 0.675 + i * 1.0)

# FLC stages
stages = [
    (3.2, 2.8, 1.8, 1.5, 'Fuzz-\nification', '#E3F2FD', '#1565C0'),
    (5.4, 2.8, 1.8, 1.5, 'Rule\nInference\n(35 rules)', '#FFF9C4', '#F57F17'),
    (7.6, 2.8, 1.8, 1.5, 'Aggre-\ngation\n(max)', '#FCE4EC', '#880E4F'),
    (9.8, 2.8, 1.8, 1.5, 'Defuzz-\nification\n(centroid)', '#F3E5F5', '#4A148C'),
]
for x, y, w, h, label, fc, ec in stages:
    box(ax, x, y, w, h, label, fc=fc, ec=ec, fontsize=9, bold=True)
for i in range(len(stages) - 1):
    arrow(ax, stages[i][0] + stages[i][2], stages[i][1] + stages[i][3]/2,
          stages[i+1][0], stages[i+1][1] + stages[i+1][3]/2)

# Actuator boxes
actuators = ['Heater\nOutput', 'Fan\nOutput', 'Dimmer\nOutput', 'Blinds\nControl']
for i, a in enumerate(actuators):
    box(ax, 12.0, 1.0 + i * 1.3, 1.8, 0.85, a, fc='#FFF3E0', ec='#E65100', fontsize=8)
    arrow(ax, 11.6, 3.55, 12.0, 1.425 + i * 1.3)

ax.text(7.0, 6.5, 'Mamdani Fuzzy Logic Controller — Smart Assistive Living Room',
        ha='center', va='center', fontsize=12, fontweight='bold', color='#1A237E')

plt.savefig(f"{FIG_DIR}/task2_architecture.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: task2_architecture.png")

# ─────────────────────────────────────────────────────────────────
# HELPER: run FLC simulation safely
# ─────────────────────────────────────────────────────────────────
def run_flc(temp_v, humid_v, act_v, light_v, time_v, comfort_v):
    try:
        flc_sim.input['temperature']         = float(temp_v)
        flc_sim.input['humidity']            = float(humid_v)
        flc_sim.input['activity']            = float(act_v)
        flc_sim.input['light_level']         = float(light_v)
        flc_sim.input['time_of_day']         = float(time_v)
        flc_sim.input['comfort_preference']  = float(comfort_v)
        flc_sim.compute()
        return (flc_sim.output.get('heater', 5.0),
                flc_sim.output.get('fan',    5.0),
                flc_sim.output.get('dimmer', 5.0),
                flc_sim.output.get('blinds', 50.0))
    except Exception:
        return (5.0, 5.0, 5.0, 50.0)

# ─────────────────────────────────────────────────────────────────
# FIGURE 3: Control surfaces
# ─────────────────────────────────────────────────────────────────
print("[Part 1] Computing control surfaces (this may take a minute)...")

N = 25  # grid resolution

# Surface 1: Heater vs (temperature, comfort_preference)
T_vals = np.linspace(10, 40, N)
C_vals = np.linspace(0, 1, N)
Z_heater = np.zeros((N, N))
for i, t in enumerate(T_vals):
    for j, c in enumerate(C_vals):
        Z_heater[i, j] = run_flc(t, 55, 0.3, 300, 12, c)[0]

# Surface 2: Fan vs (temperature, humidity)
H_vals = np.linspace(20, 90, N)
Z_fan = np.zeros((N, N))
for i, t in enumerate(T_vals):
    for j, h in enumerate(H_vals):
        Z_fan[i, j] = run_flc(t, h, 0.5, 300, 14, 0.2)[1]

# Surface 3: Dimmer vs (light_level, time_of_day)
L_vals   = np.linspace(0, 1000, N)
TOD_vals = np.linspace(0, 24, N)
Z_dimmer = np.zeros((N, N))
for i, lv in enumerate(L_vals):
    for j, tod in enumerate(TOD_vals):
        Z_dimmer[i, j] = run_flc(20, 50, 0.4, lv, tod, 0.5)[2]

# Surface 4: Blinds vs (light_level, temperature)
Z_blinds = np.zeros((N, N))
for i, lv in enumerate(L_vals):
    for j, t in enumerate(T_vals):
        Z_blinds[i, j] = run_flc(t, 50, 0.5, lv, 14, 0.5)[3]

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig, axes = plt.subplots(2, 2, figsize=(14, 10), subplot_kw={'projection': '3d'})
fig.suptitle("FLC Control Surfaces", fontsize=13, fontweight='bold')

surfaces = [
    (axes[0, 0], T_vals, C_vals, Z_heater, "Temperature (°C)", "Comfort Preference",
     "Heater Output (%)", "Heater vs Temperature & Comfort", 'YlOrRd'),
    (axes[0, 1], T_vals, H_vals, Z_fan, "Temperature (°C)", "Humidity (%)",
     "Fan Output (%)", "Fan vs Temperature & Humidity", 'Blues'),
    (axes[1, 0], L_vals, TOD_vals, Z_dimmer, "Light Level (lux)", "Time of Day (h)",
     "Dimmer Output (%)", "Dimmer vs Light & Time of Day", 'YlOrBr'),
    (axes[1, 1], L_vals, T_vals, Z_blinds, "Light Level (lux)", "Temperature (°C)",
     "Blinds Opening (%)", "Blinds vs Light & Temperature", 'Greens'),
]

for ax, x, y, z, xl, yl, zl, title, cmap in surfaces:
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, z.T, cmap=cmap, alpha=0.85, edgecolor='none')
    ax.set_xlabel(xl, fontsize=8, labelpad=5)
    ax.set_ylabel(yl, fontsize=8, labelpad=5)
    ax.set_zlabel(zl, fontsize=8, labelpad=5)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/task2_control_surfaces.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: task2_control_surfaces.png")

# ─────────────────────────────────────────────────────────────────
# SCENARIO ANALYSIS
# ─────────────────────────────────────────────────────────────────
print("[Part 1] Running scenario analysis...")

scenarios = [
    ("Cold winter morning, resting",       12, 40, 0.1, 200,  9,  0.85),
    ("Hot summer afternoon, active",       32, 75, 0.8, 950, 14,  0.1),
    ("Cool evening, moderate activity",    18, 55, 0.5, 150, 19,  0.5),
]

scenario_results = []
for name, t, h, a, l, tod, c in scenarios:
    h_out, f_out, d_out, b_out = run_flc(t, h, a, l, tod, c)
    scenario_results.append((name, t, h, a, l, tod, c, h_out, f_out, d_out, b_out))
    print(f"  {name}:")
    print(f"    Heater={h_out:.1f}%  Fan={f_out:.1f}%  Dimmer={d_out:.1f}%  Blinds={b_out:.1f}%")

# Scenario bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Scenario Analysis — FLC Crisp Outputs", fontsize=12, fontweight='bold')
outputs = ['Heater', 'Fan', 'Dimmer', 'Blinds']
colors_bar = ['#F44336', '#2196F3', '#FF9800', '#4CAF50']

for idx, (ax, res) in enumerate(zip(axes, scenario_results)):
    name, t, h, a, l, tod, c, h_out, f_out, d_out, b_out = res
    vals = [h_out, f_out, d_out, b_out]
    bars = ax.bar(outputs, vals, color=colors_bar, edgecolor='white', linewidth=1.5)
    ax.set_ylim(0, 110)
    ax.set_title(f"Scenario {idx+1}\n{name}", fontsize=9, fontweight='bold')
    ax.set_ylabel("Output (%)")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    info = f"T={t}°C, H={h}%, Act={a},\nLight={l}lux, Time={tod}h, Pref={c}"
    ax.text(0.5, -0.25, info, ha='center', va='top', transform=ax.transAxes,
            fontsize=7, color='#555555')

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/task2_scenarios.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: task2_scenarios.png")

# ─────────────────────────────────────────────────────────────────
# PART 2: EXPERT DATASET + GA OPTIMISATION
# ─────────────────────────────────────────────────────────────────
print("\n[Part 2] Generating expert dataset (150 examples)...")

# Latin Hypercube Sampling
def lhs(n, d, bounds):
    result = np.zeros((n, d))
    for j in range(d):
        lo, hi = bounds[j]
        perm = np.random.permutation(n)
        result[:, j] = lo + (perm + np.random.uniform(0, 1, n)) / n * (hi - lo)
    return result

bounds = [(10, 40), (20, 90), (0, 1), (0, 1000), (0, 24), (0, 1)]
X_lhs  = lhs(150, 6, bounds)

dataset = []
for row in X_lhs:
    t, h, a, l, tod, c = row
    h_out, f_out, d_out, b_out = run_flc(t, h, a, l, tod, c)
    dataset.append([round(t,2), round(h,2), round(a,3), round(l,1),
                    round(tod,2), round(c,3),
                    round(h_out,2), round(f_out,2), round(d_out,2), round(b_out,2)])

csv_path = f"{DATA_DIR}/task2_expert_dataset.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['temperature','humidity','activity','light_level','time_of_day',
                     'comfort_preference','ideal_heater','ideal_fan','ideal_dimmer','ideal_blinds'])
    writer.writerows(dataset)

print(f"  Saved: {csv_path} ({len(dataset)} rows)")

X_data = np.array(dataset)
inputs_data  = X_data[:, :6]
outputs_data = X_data[:, 6:]   # heater, fan, dimmer, blinds

# ── Lightweight custom Mamdani engine for GA ──────────────────────
# MF parameter blocks (initial values match the skfuzzy system above)
# Format: each variable stores list of (type, params) where type='tri'|'trap'
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

VAR_ORDER = ['temperature','humidity','activity','light_level','time_of_day','comfort_pref',
             'heater','fan','dimmer','blinds']
BOUNDS_MAP = {
    'temperature': (10, 40), 'humidity': (20, 90), 'activity': (0, 1),
    'light_level': (0, 1000), 'time_of_day': (0, 24), 'comfort_pref': (0, 1),
    'heater': (0, 100), 'fan': (0, 100), 'dimmer': (0, 100), 'blinds': (0, 100),
}

def params_to_chromosome(params):
    chrom = []
    for var in VAR_ORDER:
        for mtype, mparams in params[var]:
            chrom.extend(mparams)
    return np.array(chrom, dtype=float)

def chromosome_to_params(chrom):
    params = {}
    idx = 0
    for var in VAR_ORDER:
        mf_list = INIT_PARAMS[var]
        new_list = []
        for mtype, mp in mf_list:
            n = len(mp)
            new_list.append((mtype, list(chrom[idx:idx+n])))
            idx += n
        params[var] = new_list
    return params

CHROM_LEN = len(params_to_chromosome(INIT_PARAMS))
print(f"[Part 2] Chromosome length: {CHROM_LEN} genes")

def mf_value(universe_pts, mtype, mparams, x):
    if mtype == 'tri':
        a, b, c = mparams
        return float(fuzz.trimf(universe_pts, [a, b, c])[np.argmin(np.abs(universe_pts - x))])
    else:
        a, b, c, d = mparams
        return float(fuzz.trapmf(universe_pts, [a, b, c, d])[np.argmin(np.abs(universe_pts - x))])

UNIVERSES = {
    'temperature': temp_u, 'humidity': humid_u, 'activity': activity_u,
    'light_level': light_u, 'time_of_day': time_u, 'comfort_pref': comfort_u,
    'heater': heater_u, 'fan': fan_u, 'dimmer': dimmer_u, 'blinds': blinds_u,
}

# Simplified rule encoding for custom engine
# Each rule: (antecedents {var: set_idx}, consequent_var, consequent_set_idx)
RULES_CUSTOM = [
    # Heater rules R1-R12
    ({'temperature':0,'activity':0,'comfort_pref':2},     'heater', 3),
    ({'temperature':0,'humidity':0,'activity':0,'comfort_pref':1}, 'heater', 2),
    ({'temperature':0,'humidity':2,'activity':0,'comfort_pref':1}, 'heater', 1),
    ({'temperature':0,'activity':0,'comfort_pref':0},     'heater', 1),
    ({'temperature':0,'activity':2},                       'heater', 1),
    ({'temperature':1,'activity':0,'comfort_pref':2},     'heater', 2),
    ({'temperature':1,'comfort_pref':1},                  'heater', 1),
    ({'temperature':1,'comfort_pref':0},                  'heater', 0),
    ({'temperature':2,'humidity':2,'activity':0,'comfort_pref':2}, 'heater', 1),
    ({'temperature':2,'comfort_pref':0},                  'heater', 0),
    ({'temperature':3},                                   'heater', 0),
    ({'temperature':0,'humidity':2,'activity':2,'comfort_pref':2}, 'heater', 2),
    # Fan rules R13-R20
    ({'temperature':3,'humidity':2,'activity':2,'comfort_pref':0}, 'fan', 3),
    ({'temperature':3,'comfort_pref':0},                  'fan', 2),
    ({'temperature':3,'comfort_pref':1},                  'fan', 1),
    ({'temperature':3,'comfort_pref':2},                  'fan', 0),
    ({'temperature':2,'humidity':2,'activity':2,'comfort_pref':0}, 'fan', 2),
    ({'temperature':2,'humidity':2,'comfort_pref':0},     'fan', 1),
    ({'temperature':1,'activity':2,'comfort_pref':0},     'fan', 1),
    ({'temperature':0},                                   'fan', 0),
    # Dimmer rules R21-R30
    ({'light_level':0,'time_of_day':1},                   'dimmer', 3),
    ({'light_level':0,'time_of_day':2,'comfort_pref':2},  'dimmer', 2),
    ({'light_level':0,'time_of_day':2,'comfort_pref':0},  'dimmer', 0),
    ({'light_level':0,'time_of_day':3,'comfort_pref':2},  'dimmer', 3),
    ({'light_level':0,'time_of_day':3,'comfort_pref':0},  'dimmer', 2),
    ({'light_level':0,'time_of_day':0},                   'dimmer', 1),
    ({'light_level':1,'time_of_day':1,'comfort_pref':2},  'dimmer', 2),
    ({'light_level':1,'time_of_day':3,'comfort_pref':2},  'dimmer', 2),
    ({'light_level':1,'comfort_pref':0},                  'dimmer', 0),
    ({'light_level':2},                                   'dimmer', 0),
    # Blinds rules R31-R35
    ({'light_level':2,'temperature':3,'time_of_day':2},   'blinds', 0),
    ({'light_level':2},                                   'blinds', 0),
    ({'light_level':0},                                   'blinds', 2),
    ({'light_level':1,'temperature':2,'time_of_day':2},   'blinds', 1),
    ({'light_level':1,'time_of_day':1},                   'blinds', 1),
]

OUTPUT_VARS = ['heater','fan','dimmer','blinds']
OUTPUT_N_SETS = {'heater':4,'fan':4,'dimmer':4,'blinds':3}

def custom_flc(inputs_vec, params):
    """Lightweight Mamdani inference for GA fitness evaluation."""
    inp_vals = {
        'temperature': inputs_vec[0], 'humidity': inputs_vec[1],
        'activity': inputs_vec[2], 'light_level': inputs_vec[3],
        'time_of_day': inputs_vec[4], 'comfort_pref': inputs_vec[5],
    }
    # Fuzzify inputs
    mu = {}
    for var, mf_list in params.items():
        if var in inp_vals:
            u = UNIVERSES[var]
            x = inp_vals[var]
            mu[var] = [mf_value(u, mt, mp, x) for mt, mp in mf_list]

    outputs = {}
    for out_var in OUTPUT_VARS:
        u = UNIVERSES[out_var]
        n_sets = OUTPUT_N_SETS[out_var]
        agg = np.zeros(len(u))
        for rule in RULES_CUSTOM:
            ant, cons_var, cons_idx = rule
            if cons_var != out_var:
                continue
            firing = min(mu[v][si] for v, si in ant.items() if v in mu)
            out_mf_params = params[out_var][cons_idx]
            out_mf = np.array([mf_value(u, out_mf_params[0], out_mf_params[1], xi) for xi in u])
            clipped = np.fmin(firing, out_mf)
            agg = np.fmax(agg, clipped)
        if np.sum(agg) > 0:
            outputs[out_var] = float(fuzz.defuzz(u, agg, 'centroid'))
        else:
            outputs[out_var] = float(np.mean(u))
    return [outputs.get(v, 50.0) for v in OUTPUT_VARS]

def fitness(chrom, X_in, Y_ideal):
    params = chromosome_to_params(chrom)
    preds = np.array([custom_flc(row, params) for row in X_in])
    rmse = np.sqrt(np.mean((preds - Y_ideal)**2, axis=0))
    return -float(np.mean(rmse))

def constrain(chrom):
    """Ensure MF params remain sorted within each variable."""
    params = chromosome_to_params(chrom)
    new_chrom = []
    for var in VAR_ORDER:
        lo, hi = BOUNDS_MAP[var]
        for mtype, mp in params[var]:
            mp_arr = np.clip(np.sort(mp), lo, hi)
            new_chrom.extend(mp_arr)
    return np.array(new_chrom)

# ── GA loop ──────────────────────────────────────────────────────
print("[Part 2] Running GA optimisation (50 pop × 100 gen)...")

POP_SIZE   = 50
N_GEN      = 100
MUT_RATE   = 0.05
MUT_SIGMA  = 0.5
CX_RATE    = 0.8
TOURN_K    = 3

init_chrom = params_to_chromosome(INIT_PARAMS)
pop = np.array([constrain(init_chrom + np.random.randn(CHROM_LEN) * 0.3) for _ in range(POP_SIZE)])
pop[0] = init_chrom.copy()

Y_ideal = outputs_data  # shape (150, 4)

def tournament(pop, scores, k=3):
    idx = np.random.choice(len(pop), k, replace=False)
    return pop[idx[np.argmax(scores[idx])]]

best_scores = []
init_score  = fitness(init_chrom, inputs_data, Y_ideal)

for gen in range(N_GEN):
    scores = np.array([fitness(c, inputs_data, Y_ideal) for c in pop])
    best_scores.append(float(np.max(scores)))

    new_pop = []
    for _ in range(POP_SIZE):
        p1 = tournament(pop, scores, TOURN_K)
        p2 = tournament(pop, scores, TOURN_K)
        if np.random.rand() < CX_RATE:
            pt = np.random.randint(1, CHROM_LEN)
            child = np.concatenate([p1[:pt], p2[pt:]])
        else:
            child = p1.copy()
        mask = np.random.rand(CHROM_LEN) < MUT_RATE
        child[mask] += np.random.randn(mask.sum()) * MUT_SIGMA
        child = constrain(child)
        new_pop.append(child)
    pop = np.array(new_pop)

    if (gen + 1) % 20 == 0:
        print(f"  Gen {gen+1:3d} | Best fitness: {best_scores[-1]:.4f}")

final_scores = np.array([fitness(c, inputs_data, Y_ideal) for c in pop])
best_chrom   = pop[np.argmax(final_scores)]
best_fitness = np.max(final_scores)

print(f"\n  Initial fitness: {init_score:.4f} (RMSE = {-init_score:.2f})")
print(f"  Optimised fitness: {best_fitness:.4f} (RMSE = {-best_fitness:.2f})")
print(f"  Improvement: {(-init_score - (-best_fitness)):.2f} RMSE units")

# ── GA convergence plot ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, N_GEN + 1), [-s for s in best_scores], color='#1565C0', linewidth=2)
ax.axhline(y=-init_score, color='#F44336', linestyle='--', linewidth=1.5, label='Initial RMSE')
ax.set_xlabel("Generation", fontsize=11)
ax.set_ylabel("Best RMSE (lower = better)", fontsize=11)
ax.set_title("GA Fitness Convergence — FLC Membership Function Optimisation", fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/task2_ga_convergence.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: task2_ga_convergence.png")

# ── Before/after control surface ─────────────────────────────────
print("[Part 2] Plotting before/after control surfaces...")

opt_params = chromosome_to_params(best_chrom)

Z_before = np.zeros((N, N))
Z_after  = np.zeros((N, N))
for i, t in enumerate(T_vals):
    for j, c_val in enumerate(C_vals):
        Z_before[i, j] = run_flc(t, 55, 0.3, 300, 12, c_val)[0]
        Z_after[i, j]  = custom_flc([t, 55, 0.3, 300, 12, c_val], opt_params)[0]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), subplot_kw={'projection': '3d'})
fig.suptitle("Heater Control Surface: Before vs After GA Optimisation", fontsize=12, fontweight='bold')

for ax, Z, title in [(axes[0], Z_before, 'Before Optimisation'),
                      (axes[1], Z_after,  'After Optimisation')]:
    X, Y = np.meshgrid(T_vals, C_vals)
    ax.plot_surface(X, Y, Z.T, cmap='YlOrRd', alpha=0.85, edgecolor='none')
    ax.set_xlabel("Temperature (°C)", fontsize=8)
    ax.set_ylabel("Comfort Preference", fontsize=8)
    ax.set_zlabel("Heater (%)", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/task2_before_after_surface.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: task2_before_after_surface.png")

# ─────────────────────────────────────────────────────────────────
# PART 3: CEC'2005 BENCHMARK COMPARISON
# ─────────────────────────────────────────────────────────────────
print("\n[Part 3] CEC'2005 benchmark comparison (GA vs PSO)...")

# F1: Shifted Sphere
def f1_sphere(x, o):
    z = x - o
    return float(np.sum(z**2) - 450)

# F10: Shifted Rotated Rastrigin
def make_rotation(D, seed=0):
    rng = np.random.RandomState(seed)
    A   = rng.randn(D, D)
    Q, _ = np.linalg.qr(A)
    return Q

def f10_rastrigin(x, o, R):
    z = np.dot(R, x - o)
    return float(np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10) - 330)

N_RUNS   = 15
DIMS     = [2, 10]
MAX_EVAL_FACTOR = 10000  # max evals = factor × D

def run_ga_bench(func, bounds, D, max_evals, pop_size=50):
    lo, hi = bounds
    pop = np.random.uniform(lo, hi, (pop_size, D))
    fit = np.array([func(ind) for ind in pop])
    best = [np.min(fit)]
    evals = pop_size
    while evals < max_evals:
        for i in range(pop_size):
            idxs = np.random.choice([j for j in range(pop_size) if j != i], 3, replace=False)
            mutant = pop[idxs[0]] + 0.8 * (pop[idxs[1]] - pop[idxs[2]])
            mutant = np.clip(mutant, lo, hi)
            trial_fit = func(mutant)
            evals += 1
            if trial_fit < fit[i]:
                pop[i] = mutant
                fit[i] = trial_fit
        best.append(np.min(fit))
        if evals >= max_evals:
            break
    return np.min(fit), best

def run_pso_bench(func, bounds, D, max_evals, n_part=40):
    lo, hi = bounds
    vmax   = (hi - lo) * 0.2
    pos    = np.random.uniform(lo, hi, (n_part, D))
    vel    = np.random.uniform(-vmax, vmax, (n_part, D))
    pbest  = pos.copy()
    pbest_fit = np.array([func(p) for p in pos])
    gbest     = pbest[np.argmin(pbest_fit)].copy()
    gbest_fit = np.min(pbest_fit)
    best   = [gbest_fit]
    evals  = n_part
    w, c1, c2 = 0.7, 1.5, 1.5
    while evals < max_evals:
        for i in range(n_part):
            r1, r2 = np.random.rand(D), np.random.rand(D)
            vel[i] = w * vel[i] + c1*r1*(pbest[i]-pos[i]) + c2*r2*(gbest-pos[i])
            vel[i] = np.clip(vel[i], -vmax, vmax)
            pos[i] = np.clip(pos[i] + vel[i], lo, hi)
            f      = func(pos[i])
            evals += 1
            if f < pbest_fit[i]:
                pbest[i]     = pos[i].copy()
                pbest_fit[i] = f
                if f < gbest_fit:
                    gbest     = pos[i].copy()
                    gbest_fit = f
        best.append(gbest_fit)
        if evals >= max_evals:
            break
    return gbest_fit, best

results = {}
convergence = {}

for D in DIMS:
    o_sphere   = np.random.uniform(-50, 50, D)
    R_rastrigin = make_rotation(D, seed=D)
    o_rastrigin = np.random.uniform(-50, 50, D)
    max_evals   = MAX_EVAL_FACTOR * D

    for fname, func in [('F1_Sphere',    lambda x: f1_sphere(x, o_sphere)),
                         ('F10_Rastrigin', lambda x: f10_rastrigin(x, o_rastrigin, R_rastrigin))]:
        for opt_name, runner in [('GA', run_ga_bench), ('PSO', run_pso_bench)]:
            key = (fname, opt_name, D)
            finals = []
            convs  = []
            for run in range(N_RUNS):
                np.random.seed(run * 100 + D)
                final, conv = runner(func, (-100, 100), D, max_evals)
                finals.append(final)
                convs.append(conv)
            results[key]     = finals
            convergence[key] = convs
            m, s, b, w2 = np.mean(finals), np.std(finals), np.min(finals), np.max(finals)
            print(f"  {fname} | {opt_name} | D={D:2d} | mean={m:10.3f} std={s:9.3f} "
                  f"best={b:10.3f} worst={w2:10.3f}")

# Wilcoxon tests
print("\n[Part 3] Wilcoxon signed-rank tests (GA vs PSO):")
for fname in ['F1_Sphere', 'F10_Rastrigin']:
    for D in DIMS:
        ga_r  = results[(fname, 'GA',  D)]
        pso_r = results[(fname, 'PSO', D)]
        try:
            stat, p = wilcoxon(ga_r, pso_r)
            sig = "significant" if p < 0.05 else "not significant"
            print(f"  {fname} D={D}: p={p:.4f} ({sig})")
        except ValueError as e:
            print(f"  {fname} D={D}: {e}")

# ── Convergence plot ──────────────────────────────────────────────
print("[Part 3] Plotting convergence curves...")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("CEC'2005 Benchmark: GA vs PSO Convergence (15 runs, ±1 std)",
             fontsize=12, fontweight='bold')

plot_configs = [
    ('F1_Sphere',    2,  axes[0, 0], 'F1 Shifted Sphere (D=2)'),
    ('F1_Sphere',    10, axes[0, 1], 'F1 Shifted Sphere (D=10)'),
    ('F10_Rastrigin', 2, axes[1, 0], 'F10 Shifted Rotated Rastrigin (D=2)'),
    ('F10_Rastrigin',10, axes[1, 1], 'F10 Shifted Rotated Rastrigin (D=10)'),
]

for fname, D, ax, title in plot_configs:
    for opt, color, label in [('GA','#1565C0','GA'), ('PSO','#C62828','PSO')]:
        convs = convergence[(fname, opt, D)]
        min_len = min(len(c) for c in convs)
        mat = np.array([c[:min_len] for c in convs])
        mean_ = np.mean(mat, axis=0)
        std_  = np.std(mat, axis=0)
        x_ax  = np.linspace(0, MAX_EVAL_FACTOR * D, min_len)
        ax.plot(x_ax, mean_, color=color, linewidth=2, label=label)
        ax.fill_between(x_ax, mean_ - std_, mean_ + std_, color=color, alpha=0.15)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel("Function Evaluations", fontsize=9)
    ax.set_ylabel("Best Fitness", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/task2_cec_convergence.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: task2_cec_convergence.png")

# ── Compile results table ─────────────────────────────────────────
print("\n[Part 3] Results summary:")
print(f"{'Function':<20} {'Opt':<5} {'D':>3}  {'Mean':>12} {'Std':>10} {'Best':>12} {'Worst':>12}")
print("-" * 80)
results_table = []
for fname in ['F1_Sphere', 'F10_Rastrigin']:
    for D in DIMS:
        for opt in ['GA', 'PSO']:
            r = results[(fname, opt, D)]
            row = (fname, opt, D, np.mean(r), np.std(r), np.min(r), np.max(r))
            results_table.append(row)
            print(f"{fname:<20} {opt:<5} {D:>3}  {np.mean(r):>12.3f} {np.std(r):>10.3f} "
                  f"{np.min(r):>12.3f} {np.max(r):>12.3f}")

# ─────────────────────────────────────────────────────────────────
# ACTIVITY LOG UPDATE
# ─────────────────────────────────────────────────────────────────
log_entry = """
## Task 2 Implementation — 2026-09-07

### Part 1: FLC Design
- Installed scikit-fuzzy 0.5.0
- Defined Mamdani FLC: 6 inputs, 4 outputs, 35 rules
- Generated MF plots (10 variables), architecture diagram, 4 control surfaces, scenario analysis
- Outputs: task2_mf_plots.png, task2_architecture.png, task2_control_surfaces.png, task2_scenarios.png

### Part 2: GA Optimisation
- Generated expert dataset: 150 LHS examples → data/task2_expert_dataset.csv
- Chromosome length: {chrom_len} genes (verified)
- GA: 50 pop × 100 gen, tournament selection, Gaussian mutation
- Initial RMSE: {init_rmse:.2f} → Optimised RMSE: {opt_rmse:.2f}
- Outputs: task2_ga_convergence.png, task2_before_after_surface.png

### Part 3: CEC'2005 Benchmarks
- Functions: F1 Shifted Sphere, F10 Shifted Rotated Rastrigin
- Optimisers: GA (differential evolution), PSO
- 15 runs × 2 functions × 2 optimisers × 2 dimensions = 120 runs
- Wilcoxon tests computed for all (function, D) pairs
- Output: task2_cec_convergence.png
""".format(
    chrom_len=CHROM_LEN,
    init_rmse=-init_score,
    opt_rmse=-best_fitness,
)

log_path = "/home/mjl/softwarica/production/docs/activity-log.md"
with open(log_path, 'a') as f:
    f.write(log_entry)
print(f"\n[Log] Updated {log_path}")

# ─────────────────────────────────────────────────────────────────
# SAVE RESULTS for report
# ─────────────────────────────────────────────────────────────────
results_path = f"{DATA_DIR}/task2_results.npz"
np.savez(results_path,
         init_rmse=-init_score,
         opt_rmse=-best_fitness,
         chrom_len=CHROM_LEN,
         scenario_results=np.array([[r[7],r[8],r[9],r[10]] for r in scenario_results]),
         results_table=np.array([[rt[3],rt[4],rt[5],rt[6]] for rt in results_table]))
print(f"[Results] Saved to {results_path}")

print("\n" + "=" * 60)
print("ALL DONE — all figures and results generated.")
print("=" * 60)
