import json
import numpy as np
from openai import OpenAI
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.decomposition import PCA


# ========== output saver ==============
class Tee:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log      = open(filepath, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Tee("hypothesix_output.txt")

#========= CONFIG =============

os.environ["OPENAI_API_KEY"] = "API_KEY_HERE"
KNOWN_FILE  = "known_conjectures.json"
NEW_FILE    = "primetuples.json"
EMBED_MODEL = "text-embedding-3-large"
REG         = 1e-6
T           = 0.01
LAMBDA      = 0.7
client      = OpenAI()

THETA_MIN = 1
THETA_MAX = 10

CEILING_NAMES = {
    "Riemann Hypothesis",
}


# ======= Examplar set ========

THETA_REFERENCE = {
    "Riemann Hypothesis":                            [10, 10, 10, 10, 10, 10],
    "Twin Prime Conjecture":                         [ 7,  9,  8,  8,  6,  4],
    "Hardy-Littlewood Prime k-tuples Conjecture":    [ 9, 10,  9,  9,  7,  4],
    "Elliott-Halberstam Conjecture":                 [ 8,  8,  7,  5,  6,  6],
    "Generalized Elliott-Halberstam Conjecture":     [ 8,  9,  6,  4,  6,  6],
    "Bateman-Horn Conjecture":                       [10,  9,  7,  9,  6,  5],
    "Dickson's Conjecture":                          [ 8,  7,  8,  8,  6,  4],
    "Polignac's Conjecture":                         [ 6,  7,  8,  9,  6,  4],
    "Firoozbakht's Conjecture":                      [ 5,  7,  4,  9,  3,  2],
    "Granville's Refinement of Cramer's Conjecture": [ 4,  8,  5,  3,  3,  2],
    "Hardy-Littlewood Second Conjecture":            [ 4,  8,  5,  3,  3,  2],
    "Bunyakovsky Conjecture":                        [ 7,  7,  5,  9,  4,  5],
    "Chowla's Conjecture on Prime Correlations":     [ 8,  9,  8,  6,  7,  6],
    "Shanks Conjecture on Prime Gaps":               [ 4,  5,  4,  8,  4,  3],
    "Schinzel's Hypothesis H":                       [ 9,  9,  6,  7,  7,  6],
    "Legendre's Conjecture":                         [ 5,  5,  6,  9,  6,  4],
    "Landau's Fourth Problem (Near-Square Primes)":  [ 6,  6,  7,  9,  5,  5],
    "Parity Problem (Selberg's Parity Barrier)":     [ 7, 10,  8,  5,  7,  7],
}


# ========= VALIDATION =============

def validate_theta(name: str, vec: list) -> np.ndarray:
    for i, val in enumerate(vec, 1):
        if not (THETA_MIN <= val <= THETA_MAX):
            raise ValueError(
                f"'{name}': theta_{i} = {val} out of [{THETA_MIN}, {THETA_MAX}]"
            )
    return np.array(vec, dtype=float)



# ========== HELPERS ============

def l2_norm(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def hybrid_rep(e_norm: np.ndarray, theta_norm: np.ndarray,
               lam: float = LAMBDA) -> np.ndarray:
    """ 
    Hybrid representation combining text embedding and theta vector.
    h = l2_norm(concat[(1-lambda)*e_norm, lambda*theta_norm])
    """
    h = np.concatenate([(1.0 - lam) * e_norm, lam * theta_norm])
    return l2_norm(h)



# ============ LOAD REFERENCE SET ================

with open(KNOWN_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

known     = data["conjectures"]
ref_items = [(c["name"], c["statement"])
             for c in known if c["name"] in THETA_REFERENCE]
ref_names = [r[0] for r in ref_items]
ref_texts = [r[1] for r in ref_items]
n         = len(ref_names)

X_R = np.array([validate_theta(name, THETA_REFERENCE[name])
                for name in ref_names])

X_R_theta_norm = np.array([l2_norm(x) for x in X_R])

print(f"Reference set R  (n = {n})")
print(f"theta bounds: [{THETA_MIN}, {THETA_MAX}] for all features\n")
print(f"  {'Name':<48} θ₁  θ₂  θ₃  θ₄  θ₅  θ₆")
print("  " + "-"*70)
for name, vec in zip(ref_names, X_R):
    tag  = " <- ceiling" if name in CEILING_NAMES else ""
    vals = "  ".join(f"{int(v):>2}" for v in vec)
    print(f"  {name:<48} {vals}{tag}")



# ========= CLUSTER IN THETA-SPACE ==============

mu  = X_R.mean(axis=0)
cov = np.cov(X_R, rowvar=False)
cov += REG * np.eye(6)
try:
    cov_inv = np.linalg.inv(cov)
except np.linalg.LinAlgError:
    print("Singular covariance — using pseudoinverse.")
    cov_inv = np.linalg.pinv(cov)


def mahal_sq(x: np.ndarray) -> float:
    d = x - mu
    return float(d @ cov_inv @ d)


ref_d2      = np.array([mahal_sq(x) for x in X_R])
ceiling_idx = {i for i, name in enumerate(ref_names) if name in CEILING_NAMES}



# ========= NON-TRIVIALITY SCORE  Υ(C) ==========

def upsilon(x: np.ndarray, idx: int = -1) -> float:
    d2_c = mahal_sq(x)
    if idx in ceiling_idx:
        mask = np.array([i not in ceiling_idx for i in range(n)])
    elif 0 <= idx < n:
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
    else:
        mask = np.ones(n, dtype=bool)
    pool = ref_d2[mask]
    if len(pool) == 0:
        return 1.0
    return float(np.sum(pool <= d2_c)) / len(pool)


# ======== EXAMPLAR SET REPORT ============

print(f"\n\n=== REFERENCE RANKING (lowest d^2 = closest to cluster centre) ===\n")
print(f"  {'Rank':<5} {'d^2':>9}  {'Upsilon':>8}  Name")
print("  " + "-"*65)
for rank, (i, name) in enumerate(
        sorted(enumerate(ref_names), key=lambda t: ref_d2[t[0]]), 1):
    score = upsilon(X_R[i], idx=i)
    tag   = " <- ceiling" if i in ceiling_idx else ""
    print(f"  {rank:<5} {ref_d2[i]:>9.4f}  {score:>8.4f}  {name}{tag}")

print("\n--- Calibration ---")
for i in ceiling_idx:
    s      = upsilon(X_R[i], idx=i)
    status = "OK" if abs(s - 1.0) < 1e-9 else "WARN"
    print(f"  [{status}] Upsilon({ref_names[i]}) = {s:.4f}")



# ========== EMBED REFERENCE ===========

print("\nEmbedding reference conjectures...")
resp     = client.embeddings.create(model=EMBED_MODEL, input=ref_texts)
E_R      = np.array([e.embedding for e in resp.data])
E_R_norm = E_R / np.linalg.norm(E_R, axis=1, keepdims=True)

# Build hybrid representations for reference conjectures
H_R = np.array([
    hybrid_rep(E_R_norm[i], X_R_theta_norm[i])
    for i in range(n)
])



# ========= LOAD AND EMBED THEOREMS ==========

theorems   = data["related_results"]
thm_names  = [t["name"]      for t in theorems]
thm_texts  = [t["statement"] for t in theorems]

print(f"Embedding {len(thm_names)} theorems as visual landmarks...")
resp_thm   = client.embeddings.create(model=EMBED_MODEL, input=thm_texts)
E_thm      = np.array([e.embedding for e in resp_thm.data])
E_thm_norm = E_thm / np.linalg.norm(E_thm, axis=1, keepdims=True)

thm_theta_hats = []
for e_norm in E_thm_norm:
    sims    = E_R_norm @ e_norm
    exp_s   = np.exp((sims - sims.max()) / T)
    weights = exp_s / exp_s.sum()
    thm_theta_hats.append(weights @ X_R)
thm_theta_hats = np.array(thm_theta_hats)


# =============================
# LOAD NEW CONJECTURES
# =============================
with open(NEW_FILE, "r", encoding="utf-8") as f:
    data_new = json.load(f)

new_conjectures = []
for conversation in data_new.get("conversations", []):
    for c in conversation.get("conjectures", []):
        if c.get("type", "").lower() == "conjecture" and "statement" in c:
            new_conjectures.append({"name": c["id"], "text": c["statement"]})

print(f"Loaded {len(new_conjectures)} new conjectures.")


# =============================
# EMBED NEW CONJECTURES
# =============================
print("Embedding new conjectures...")
new_texts  = [c["text"] for c in new_conjectures]
resp_new   = client.embeddings.create(model=EMBED_MODEL, input=new_texts)
E_new      = np.array([e.embedding for e in resp_new.data])
E_new_norm = E_new / np.linalg.norm(E_new, axis=1, keepdims=True)


# =============================
# SCORE NEW CONJECTURES
# =============================
print("\n\n=== NEW CONJECTURE SCORES ===\n")
results = []
for c, e_norm in zip(new_conjectures, E_new_norm):

    # Pass 1: cold-start theta estimate via text-only similarity
    sims_text      = E_R_norm @ e_norm
    exp_text       = np.exp((sims_text - sims_text.max()) / T)
    w_text         = exp_text / exp_text.sum()
    theta_hat_init = w_text @ X_R

    # Pass 2: build hybrid and recompute weights
    theta_hat_norm = l2_norm(theta_hat_init)
    h_new          = hybrid_rep(e_norm, theta_hat_norm)
    sims_hybrid    = H_R @ h_new
    exp_h          = np.exp((sims_hybrid - sims_hybrid.max()) / T)
    weights        = exp_h / exp_h.sum()

    # Final estimates
    theta_hat = weights @ X_R
    d2        = mahal_sq(theta_hat)
    score     = float(np.sum(ref_d2 <= d2)) / n
    results.append((c["name"], theta_hat, d2, score))

    top3 = np.argsort(sims_hybrid)[::-1][:3]
    print(f"  {c['name']}")
    print(f"    theta_hat = {np.round(theta_hat, 2).tolist()}")
    print(f"    d^2       = {d2:.4f}   Upsilon = {score:.4f}")
    print(f"    Closest (hybrid): {ref_names[top3[0]]} (sim={sims_hybrid[top3[0]]:.3f})")
    print()


# =============================
# FINAL RANKED SUMMARY
# =============================
print("=== FINAL RANKING (highest Upsilon = most non-trivial) ===\n")
print(f"  {'Rank':<6} {'Upsilon':>8}  {'d^2':>10}  Name")
print("  " + "-"*58)
for rank, (name, _, d2, score) in enumerate(
        sorted(results, key=lambda x: -x[3]), 1):
    print(f"  {rank:<6} {score:>8.4f}  {d2:>10.4f}  {name}")

print(f"\n  Ceiling = 1.0  ({', '.join(CEILING_NAMES)})")


# =============================
# VISUALISATION
# =============================
pca        = PCA(n_components=2, random_state=42)
X_R_2d     = pca.fit_transform(X_R)
X_new_2d   = pca.transform(np.array([r[1] for r in results]))
X_thm_2d   = pca.transform(thm_theta_hats)
new_scores = [r[3] for r in results]

BG        = "#FFFFFF"
BLUE_CEIL = "#5B9BD5"
BLUE_REF  = "#1E3F7A"
GREEN     = "#2E8B57"

fig, ax = plt.subplots(figsize=(5, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_aspect("equal")
ax.axis("off")

all_x = np.concatenate([X_R_2d[:,0], X_new_2d[:,0], X_thm_2d[:,0]])
all_y = np.concatenate([X_R_2d[:,1], X_new_2d[:,1], X_thm_2d[:,1]])
cx    = (all_x.max() + all_x.min()) / 2
cy    = (all_y.max() + all_y.min()) / 2
rad   = max(all_x.max() - cx, all_y.max() - cy) * 1.15

ax.add_patch(Circle((cx, cy), rad, facecolor="#F7F7F7",
                     edgecolor="black", linewidth=2.0, zorder=0))
clip_circle = Circle((cx, cy), rad, transform=ax.transData)

for pt, score in zip(X_new_2d, new_scores):
    t  = float(score)
    sc = ax.scatter(pt[0], pt[1], s=72,
                    color=(0.60 + 0.38*t, 0.22 + 0.18*t, 0.04),
                    edgecolors="white", linewidths=0.5,
                    alpha=0.88, zorder=3)
    sc.set_clip_path(clip_circle)

for pt in X_thm_2d:
    sc = ax.scatter(pt[0], pt[1], s=72, color=GREEN,
                    edgecolors="white", linewidths=1.0,
                    alpha=0.90, zorder=4)
    sc.set_clip_path(clip_circle)

for name, pt in zip(ref_names, X_R_2d):
    is_ceil = name in CEILING_NAMES
    color   = BLUE_CEIL if is_ceil else BLUE_REF
    glow    = "#A8D4FF" if is_ceil else "#4A7ABF"
    for s, a, z, col, lw in [(300, 0, 5, glow, 0),
                              (72,  1.0,  6, color, 1.2)]:
        sc = ax.scatter(pt[0], pt[1], s=s, color=col,
                        edgecolors="none" if lw == 0 else "white",
                        linewidths=lw, alpha=a, zorder=z)
        sc.set_clip_path(clip_circle)

mu_2d = pca.transform(mu.reshape(1, -1))[0]
ax.scatter(mu_2d[0], mu_2d[1], s=72, color="#333333",
           marker="+", linewidths=1.8, zorder=7, alpha=0.7)

ax.set_xlim(cx - rad*1.08, cx + rad*1.08)
ax.set_ylim(cy - rad*1.08, cy + rad*1.08)
plt.tight_layout(pad=0.1)
plt.savefig("upsilon_cluster.png", dpi=200,
            bbox_inches="tight", facecolor=BG)
plt.show()