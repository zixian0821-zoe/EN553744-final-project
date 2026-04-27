\
\
\
\
\
\
\
\
\
\
\
\
   

import json
import time
import numpy as np
from scipy.sparse import load_npz, csr_matrix, diags

BASE = "/sessions/modest-kind-goldberg/mnt/large_scale_graph_final_project/Experiment2"

adj_path = f"{BASE}/results/recommendation_learned_alpha/source_graph_adjacency.npz"
music_adj = load_npz(adj_path)
N = music_adj.shape[0]
assert music_adj.shape == (N, N)

print("="*65)
print("Exp 1 re-run on Exp 2's source graph (top-30 L1 music profile)")
print("="*65)
print(f"Nodes: {N}")
print(f"Symmetric nnz: {music_adj.nnz}")
print(f"Undirected edges: {music_adj.nnz // 2}")
print(f"Mean degree (directed entries / N): {music_adj.nnz / N:.2f}")

book_prof = load_npz(f"{BASE}/data/book_user_genre_matrix.npz").toarray()
print(f"\nBook profile: shape={book_prof.shape}")

top20 = np.argsort(book_prof.sum(axis=0))[-20:][::-1]
Y_raw = book_prof[:, top20].copy().astype(np.float64)

row_sums = Y_raw.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
Y = Y_raw / row_sums
print(f"Signal: shape={Y.shape}, top-20 book genres, L1-normalized per user")

def compute_dirichlet_energy_multi(adj, signals):
    degrees = np.asarray(adj.sum(axis=1)).ravel()
    D = diags(degrees)
    L = D - adj
    per_dim = []
    for d in range(signals.shape[1]):
        x = signals[:, d].copy()
        std = x.std()
        if std < 1e-10:
            continue
        x = (x - x.mean()) / std
        energy = float(x @ L @ x)
        per_dim.append(energy)
    return float(np.mean(per_dim)), per_dim

N_PERMS = 200
np.random.seed(42)

print(f"\nComputing energy on ACTUAL music graph (top-30 profile)...")
t0 = time.time()
music_energy, music_per_dim = compute_dirichlet_energy_multi(music_adj, Y)
print(f"  Music graph avg Dirichlet energy: {music_energy:.2f}  (took {time.time()-t0:.1f}s)")

print(f"\nRunning {N_PERMS} permutations (node-ID shuffle on signal)...")
t0 = time.time()
perm_energies = []
perm_per_dim_all = []
for p in range(N_PERMS):
    perm = np.random.permutation(N)
    Y_perm = Y[perm]
    e, per_dim = compute_dirichlet_energy_multi(music_adj, Y_perm)
    perm_energies.append(e)
    perm_per_dim_all.append(per_dim)
    if (p+1) % 50 == 0:
        print(f"  [{p+1}/{N_PERMS}] elapsed {time.time()-t0:.1f}s")
perm_energies = np.array(perm_energies)
perm_per_dim_all = np.array(perm_per_dim_all)

perm_mean = perm_energies.mean()
perm_std = perm_energies.std()
z_score = (music_energy - perm_mean) / perm_std
n_below = int((perm_energies < music_energy).sum())
p_value_lower_bound = (n_below + 1) / (N_PERMS + 1)
energy_ratio = music_energy / perm_mean
smoothness_pct = (1.0 - energy_ratio) * 100.0

per_genre_ratios = []
for d in range(Y.shape[1]):
    real_d = music_per_dim[d] if d < len(music_per_dim) else np.nan
    perm_d_mean = perm_per_dim_all[:, d].mean()
    per_genre_ratios.append(real_d / perm_d_mean if perm_d_mean > 0 else np.nan)
per_genre_ratios = np.array(per_genre_ratios)

per_genre_zscores = []
for d in range(Y.shape[1]):
    real_d = music_per_dim[d] if d < len(music_per_dim) else np.nan
    perm_d = perm_per_dim_all[:, d]
    z = (real_d - perm_d.mean()) / perm_d.std() if perm_d.std() > 0 else 0.0
    per_genre_zscores.append(z)
per_genre_zscores = np.array(per_genre_zscores)

sig_ratio_05 = int((per_genre_ratios < 0.95).sum())
sig_z2 = int((per_genre_zscores < -2).sum())
sig_z3 = int((per_genre_zscores < -3).sum())
sig_smoother = sig_z3
similar = Y.shape[1] - sig_smoother
sig_rougher = int(((per_genre_ratios > 1.05) & (per_genre_zscores > 2)).sum())

print("\n" + "="*65)
print("RESULTS")
print("="*65)
print(f"Music graph Dirichlet energy       : {music_energy:.2f}")
print(f"Random permutation mean ± std       : {perm_mean:.2f} ± {perm_std:.2f}")
print(f"Random range                        : [{perm_energies.min():.2f}, {perm_energies.max():.2f}]")
print(f"Z-score                             : {z_score:.2f}")
print(f"Energy ratio (music / random)       : {energy_ratio:.4f}")
print(f"Smoothness improvement              : {smoothness_pct:.1f}%")
print(f"Music energy below {n_below}/{N_PERMS} permutations")
print(f"P-value (lower bound)               : <= {p_value_lower_bound:.4f}")
print(f"Per-genre: {sig_smoother} smoother / {similar} similar / {sig_rougher} rougher (of 20)")

out = {
    "experiment": "signal_smoothness_validation_RERUN_on_exp2_graph",
    "graph": "Exp 2 source graph (top-30 L1 music profile, k=10 cosine kNN)",
    "graph_adjacency_path": adj_path,
    "signal": "20-dim book genre distribution (L1-normalized)",
    "n_users": int(N),
    "n_permutations": N_PERMS,
    "n_edges_symmetric_nnz": int(music_adj.nnz),
    "n_edges_undirected": int(music_adj.nnz // 2),
    "avg_degree_directed_entries": float(music_adj.nnz / N),
    "music_graph_energy": float(music_energy),
    "random_mean": float(perm_mean),
    "random_std": float(perm_std),
    "random_min": float(perm_energies.min()),
    "random_max": float(perm_energies.max()),
    "z_score": float(z_score),
    "p_value_lower_bound": float(p_value_lower_bound),
    "n_below_permutations": n_below,
    "energy_ratio": float(energy_ratio),
    "smoothness_percent": float(smoothness_pct),
    "per_genre_smoother_by_ratio_lt_0.95": sig_ratio_05,
    "per_genre_smoother_by_z_lt_neg2": sig_z2,
    "per_genre_smoother_by_z_lt_neg3": sig_z3,
    "per_genre_smoother_significant": sig_smoother,
    "per_genre_similar": similar,
    "per_genre_rougher_significant": sig_rougher,
    "per_genre_total": int(Y.shape[1]),
    "per_genre_ratios": per_genre_ratios.tolist(),
    "per_genre_zscores": per_genre_zscores.tolist(),
}

out_path = "/sessions/modest-kind-goldberg/mnt/large_scale_graph_final_project/experiments/exp1_signal_smoothness/results/permutation_test_exp2_graph.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")
