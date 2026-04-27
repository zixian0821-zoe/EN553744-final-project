\
\
\
\
\
\
\
\
\
   

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix, diags
from sklearn.neighbors import NearestNeighbors
import time

BASE = "/sessions/amazing-loving-ritchie/mnt/large_scale_graph_final_project/New project/output/amazon-genre-graph"

music_prof = load_npz(f"{BASE}/profiles/music_user_genre_matrix.npz").toarray()
book_prof  = load_npz(f"{BASE}/profiles/book_user_genre_matrix.npz").toarray()

n_full = len(music_prof)
print(f"Full dataset: {n_full} users")

np.random.seed(42)
N = 3000
sub_idx = np.sort(np.random.choice(n_full, N, replace=False))
music_sub = music_prof[sub_idx]
book_sub  = book_prof[sub_idx]

top20 = np.argsort(book_sub.sum(axis=0))[-20:][::-1]
Y_raw = book_sub[:, top20].copy().astype(np.float64)
row_sums = Y_raw.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
Y = Y_raw / row_sums

n_genres = Y.shape[1]
valid_mask = Y_raw.sum(axis=1) > 0
print(f"Subsample: {N} users, {valid_mask.sum()} with book data, {n_genres} genre dims")

K = 5
print(f"\nBuilding music k-NN graph (k={K})...")
nn = NearestNeighbors(n_neighbors=K+1, metric="cosine", algorithm="brute")
nn.fit(music_sub)
dist, idx = nn.kneighbors(music_sub)
sim = 1.0 - dist

rows, cols, vals = [], [], []
for i in range(N):
    for j in range(1, K+1):
        s = max(sim[i, j], 0.0)
        if s > 0:
            rows.append(i)
            cols.append(idx[i, j])
            vals.append(s)

music_adj = csr_matrix((vals, (rows, cols)), shape=(N, N))
music_adj = music_adj + music_adj.T
music_adj.data = np.minimum(music_adj.data, 1.0)

print(f"Music graph: {N} nodes, {music_adj.nnz} edges, avg degree = {music_adj.nnz/N:.1f}")
print(f"Edge weight stats: mean={music_adj.data.mean():.4f}, "
      f"std={music_adj.data.std():.4f}, "
      f"min={music_adj.data.min():.4f}, max={music_adj.data.max():.4f}")

def compute_dirichlet_energy_multi(adj, signals):
\
\
\
\
\
\
\
\
       
    degrees = np.array(adj.sum(axis=1)).flatten()
    D = diags(degrees)
    L = D - adj

    energies = []
    for d in range(signals.shape[1]):
        x = signals[:, d].copy()
        std = x.std()
        if std < 1e-10:
            continue
        x = (x - x.mean()) / std
        energy = float(x @ L @ x)
        energies.append(energy)

    return np.mean(energies), energies

def permute_graph(adj):
\
\
\
       
    perm = np.random.permutation(adj.shape[0])
    adj_coo = adj.tocoo()
    new_rows = perm[adj_coo.row]
    new_cols = perm[adj_coo.col]
    perm_adj = csr_matrix((adj_coo.data, (new_rows, new_cols)), shape=adj.shape)
    return perm_adj

print("\n" + "=" * 65)
print("DIRICHLET ENERGY — PERMUTATION TEST")
print("=" * 65)

print("\nComputing Dirichlet energy on MUSIC graph...")
music_energy, music_per_dim = compute_dirichlet_energy_multi(music_adj, Y)
print(f"  Music graph average energy: {music_energy:.4f}")

N_PERMS = 100
print(f"\nRunning {N_PERMS} permutations...")
t0 = time.time()
perm_energies = []
for i in range(N_PERMS):
    perm_adj = permute_graph(music_adj)
    perm_energy, _ = compute_dirichlet_energy_multi(perm_adj, Y)
    perm_energies.append(perm_energy)
    if (i + 1) % 20 == 0:
        elapsed = time.time() - t0
        print(f"  {i+1}/{N_PERMS} done ({elapsed:.1f}s), "
              f"current mean={np.mean(perm_energies):.4f}")

perm_energies = np.array(perm_energies)
elapsed = time.time() - t0

print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)

print(f"\nMusic graph Dirichlet energy:   {music_energy:.4f}")
print(f"Random permutations (n={N_PERMS}):")
print(f"  Mean:   {perm_energies.mean():.4f}")
print(f"  Std:    {perm_energies.std():.4f}")
print(f"  Min:    {perm_energies.min():.4f}")
print(f"  Max:    {perm_energies.max():.4f}")

p_value = np.mean(perm_energies <= music_energy)
print(f"\np-value (one-sided): {p_value:.4f}")
print(f"  (fraction of random graphs with energy <= music graph)")

z_score = (music_energy - perm_energies.mean()) / perm_energies.std()
print(f"z-score: {z_score:.2f}")

if music_energy < perm_energies.min():
    print(f"\n✅ Music graph energy ({music_energy:.4f}) is LOWER than ALL "
          f"{N_PERMS} random permutations!")
    print(f"   → p < {1/N_PERMS:.4f}")
    print(f"   → Book genre signal is significantly smoother on music topology")
    print(f"      than on random topology with identical edge weights.")
elif p_value < 0.05:
    print(f"\n✅ Music graph energy is significantly lower (p={p_value:.4f})")
    print(f"   → Book signal is smoother on music graph than random.")
else:
    print(f"\n⚠️ Not significant (p={p_value:.4f})")
    print(f"   → Music topology may not provide smoothness advantage.")

print("\n" + "=" * 65)
print("PER-GENRE DIRICHLET ENERGY (Music graph)")
print("=" * 65)

genre_indices = top20
print(f"\n{'Genre Idx':<12} {'Music E':<12} {'Random E (mean)':<18} {'Ratio':<10} {'Smoother?'}")
print("-" * 62)

perm_adj_ref = permute_graph(music_adj)
_, perm_per_dim_ref = compute_dirichlet_energy_multi(perm_adj_ref, Y)

for d in range(min(n_genres, 20)):
    if d < len(music_per_dim) and d < len(perm_per_dim_ref):
        m_e = music_per_dim[d]
        r_e = perm_per_dim_ref[d]
        ratio = m_e / max(r_e, 1e-8)
        smoother = "✅ Yes" if ratio < 0.95 else "❌ No" if ratio > 1.05 else "≈ Same"
        print(f"Genre {d:<5} {m_e:<12.2f} {r_e:<18.2f} {ratio:<10.3f} {smoother}")

ratio_overall = music_energy / perm_energies.mean()
print(f"\nOverall energy ratio (music / random): {ratio_overall:.4f}")
print(f"  → Music graph uses {(1-ratio_overall)*100:.1f}% less energy than random")
print(f"  → Book signal is {(1-ratio_overall)*100:.1f}% smoother on music topology")

print(f"\nTime elapsed: {elapsed:.1f}s")
