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
\
\
\
\
   

BASE = "/content/output/amazon-genre-graph"

import numpy as np
from scipy.sparse import load_npz, csr_matrix, diags
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import time

music_prof = load_npz(f"{BASE}/profiles/music_user_genre_matrix.npz").toarray()
book_prof  = load_npz(f"{BASE}/profiles/book_user_genre_matrix.npz").toarray()

N = len(music_prof)
print(f"Loaded {N} users")
print(f"Music profile: {music_prof.shape}, density={( music_prof > 0).mean():.3%}")
print(f"Book  profile: {book_prof.shape}, density={(book_prof > 0).mean():.3%}")

top20 = np.argsort(book_prof.sum(axis=0))[-20:][::-1]
Y_raw = book_prof[:, top20].copy().astype(np.float64)

row_sums = Y_raw.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
Y = Y_raw / row_sums

n_genres = Y.shape[1]
valid_mask = Y_raw.sum(axis=1) > 0
print(f"Signal: {n_genres}-dim book genre distribution")
print(f"Users with book data: {valid_mask.sum()} / {N}")
print(f"Signal mean per dim: {Y[valid_mask].mean(axis=0)[:5].round(4)} ...")
print(f"Signal std  per dim: {Y[valid_mask].std(axis=0)[:5].round(4)} ...")

K = 10
print(f"Building music k-NN graph (k={K}) for {N} users...")
t0 = time.time()

nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine", algorithm="brute",
                      n_jobs=-1)
nn.fit(music_prof)
dist, idx = nn.kneighbors(music_prof)
sim = 1.0 - dist

rows, cols, vals = [], [], []
for i in range(N):
    for j in range(1, K + 1):
        s = max(sim[i, j], 0.0)
        if s > 0:
            rows.append(i)
            cols.append(idx[i, j])
            vals.append(s)

music_adj = csr_matrix((vals, (rows, cols)), shape=(N, N))
music_adj = music_adj + music_adj.T
music_adj.data = np.minimum(music_adj.data, 1.0)

elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s")
print(f"Music graph: {N} nodes, {music_adj.nnz} edges, "
      f"avg degree = {music_adj.nnz / N:.1f}")
print(f"Edge weights: mean={music_adj.data.mean():.4f}, "
      f"std={music_adj.data.std():.4f}, "
      f"range=[{music_adj.data.min():.4f}, {music_adj.data.max():.4f}]")

def compute_dirichlet_energy_multi(adj, signals):
\
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

    per_dim = []
    for d in range(signals.shape[1]):
        x = signals[:, d].copy()
        std = x.std()
        if std < 1e-10:
            continue
        x = (x - x.mean()) / std
        energy = float(x @ L @ x)
        per_dim.append(energy)

    return np.mean(per_dim), per_dim

def permute_graph(adj):
\
\
\
       
    perm = np.random.permutation(adj.shape[0])
    adj_coo = adj.tocoo()
    new_rows = perm[adj_coo.row]
    new_cols = perm[adj_coo.col]
    return csr_matrix((adj_coo.data, (new_rows, new_cols)), shape=adj.shape)

print("Functions defined ✓")

N_PERMS = 200
np.random.seed(42)

print("=" * 65)
print(f"DIRICHLET ENERGY PERMUTATION TEST  (N={N}, k={K}, perms={N_PERMS})")
print("=" * 65)

print("\n1. Computing energy on MUSIC graph...")
music_energy, music_per_dim = compute_dirichlet_energy_multi(music_adj, Y)
print(f"   Music graph average Dirichlet energy: {music_energy:.2f}")

print(f"\n2. Running {N_PERMS} random permutations...")
t0 = time.time()
perm_energies = []
perm_per_dim_all = []

for i in range(N_PERMS):
    perm_adj = permute_graph(music_adj)
    pe, ppd = compute_dirichlet_energy_multi(perm_adj, Y)
    perm_energies.append(pe)
    perm_per_dim_all.append(ppd)
    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"   {i+1}/{N_PERMS} done ({elapsed:.1f}s), "
              f"running mean = {np.mean(perm_energies):.2f}")

perm_energies = np.array(perm_energies)
perm_per_dim_all = np.array(perm_per_dim_all)
elapsed = time.time() - t0
print(f"   Completed in {elapsed:.1f}s")

print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)

print(f"\nMusic graph energy:          {music_energy:.2f}")
print(f"Permuted graphs (n={N_PERMS}):")
print(f"  Mean:  {perm_energies.mean():.2f}")
print(f"  Std:   {perm_energies.std():.2f}")
print(f"  Range: [{perm_energies.min():.2f}, {perm_energies.max():.2f}]")

p_value = np.mean(perm_energies <= music_energy)
z_score = (music_energy - perm_energies.mean()) / perm_energies.std()
ratio   = music_energy / perm_energies.mean()

print(f"\np-value (one-sided): {p_value:.4f}")
if p_value == 0:
    print(f"  → Music energy < ALL {N_PERMS} permutations → p < {1/N_PERMS:.4f}")
print(f"z-score: {z_score:.2f}")
print(f"Energy ratio (music / random): {ratio:.4f}")
print(f"  → Book signal is {(1 - ratio) * 100:.1f}% smoother on music topology")

if music_energy < perm_energies.min():
    print(f"\n✅ HIGHLY SIGNIFICANT: Music graph energy ({music_energy:.2f}) is "
          f"lower than ALL {N_PERMS} random permutations.")
    print(f"   Book genre distribution is significantly smoother on music")
    print(f"   similarity topology than on random topology.")
elif p_value < 0.01:
    print(f"\n✅ SIGNIFICANT (p < 0.01): Book signal is smoother on music graph.")
elif p_value < 0.05:
    print(f"\n✅ Significant (p < 0.05): Book signal is smoother on music graph.")
else:
    print(f"\n⚠️ Not significant (p = {p_value:.4f}).")

print("\n" + "=" * 65)
print("PER-GENRE ANALYSIS")
print("=" * 65)

n_valid_dims = len(music_per_dim)
perm_per_dim_mean = perm_per_dim_all.mean(axis=0)
perm_per_dim_std  = perm_per_dim_all.std(axis=0)

print(f"\n{'Genre':<8} {'Music E':>10} {'Rand E (μ)':>12} {'Rand E (σ)':>12} "
      f"{'Ratio':>8} {'z-score':>9} {'Status'}")
print("-" * 75)

genre_ratios = []
genre_zscores = []
for d in range(n_valid_dims):
    m_e = music_per_dim[d]
    r_mean = perm_per_dim_mean[d]
    r_std  = perm_per_dim_std[d]
    r = m_e / max(r_mean, 1e-8)
    z = (m_e - r_mean) / max(r_std, 1e-8)
    genre_ratios.append(r)
    genre_zscores.append(z)

    if r < 0.95:
        status = "✅ Smoother"
    elif r > 1.05:
        status = "❌ Rougher"
    else:
        status = "≈ Same"
    print(f"G{d:<6} {m_e:>10.1f} {r_mean:>12.1f} {r_std:>12.1f} "
          f"{r:>8.3f} {z:>9.2f} {status}")

n_smoother = sum(1 for r in genre_ratios if r < 0.95)
n_same     = sum(1 for r in genre_ratios if 0.95 <= r <= 1.05)
n_rougher  = sum(1 for r in genre_ratios if r > 1.05)
print(f"\nSummary: {n_smoother} smoother, {n_same} similar, {n_rougher} rougher "
      f"(out of {n_valid_dims} genres)")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.hist(perm_energies, bins=30, color="lightcoral", edgecolor="darkred",
        alpha=0.7, label=f"Random permutations (n={N_PERMS})")
ax.axvline(music_energy, color="steelblue", linewidth=2.5, linestyle="--",
           label=f"Music graph (E={music_energy:.1f})")
ax.set_xlabel("Average Dirichlet Energy", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"Permutation Test (p{'<' if p_value==0 else '='}"
             f"{max(p_value, 1/N_PERMS):.4f}, z={z_score:.2f})", fontsize=12)
ax.legend(fontsize=10)

ax = axes[1]
colors = ["steelblue" if r < 0.95 else "lightcoral" if r > 1.05 else "gray"
          for r in genre_ratios]
ax.bar(range(n_valid_dims), genre_ratios, color=colors, edgecolor="black",
       linewidth=0.3)
ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
ax.axhline(0.95, color="steelblue", linewidth=0.8, linestyle=":", alpha=0.7)
ax.axhline(1.05, color="lightcoral", linewidth=0.8, linestyle=":", alpha=0.7)
ax.set_xlabel("Genre Index", fontsize=12)
ax.set_ylabel("Energy Ratio (Music / Random)", fontsize=12)
ax.set_title("Per-Genre Smoothness", fontsize=12)
ax.set_ylim(0.75, 1.15)

ax = axes[2]
colors_z = ["steelblue" if z < -2 else "lightcoral" if z > 2 else "gray"
            for z in genre_zscores]
ax.bar(range(n_valid_dims), genre_zscores, color=colors_z, edgecolor="black",
       linewidth=0.3)
ax.axhline(0, color="black", linewidth=1)
ax.axhline(-2, color="steelblue", linewidth=0.8, linestyle=":", alpha=0.7,
           label="z = ±2")
ax.axhline(2, color="lightcoral", linewidth=0.8, linestyle=":", alpha=0.7)
ax.set_xlabel("Genre Index", fontsize=12)
ax.set_ylabel("z-score (negative = smoother)", fontsize=12)
ax.set_title("Per-Genre Statistical Significance", fontsize=12)
ax.legend(fontsize=10)

plt.suptitle(f"Dirichlet Energy Analysis — Book Genre Signal on Music Graph\n"
             f"(N={N}, k={K}, {N_PERMS} permutations)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("dirichlet_energy_permutation_test.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nFigure saved: dirichlet_energy_permutation_test.png")

print("\n" + "=" * 65)
print("SUMMARY FOR REPORT")
print("=" * 65)

print(f"""
Experiment: Dirichlet Energy Permutation Test
  Dataset: Amazon Reviews 2023, CDs & Vinyl → Books
  Users: {N}
  Graph: k-NN (k={K}), cosine similarity on music genre profiles
  Signal: 20-dim book genre distribution (normalized)
  Permutations: {N_PERMS} (node-ID shuffle, identical edge weights)

Results:
  Music graph energy:    {music_energy:.2f}
  Random mean energy:    {perm_energies.mean():.2f} ± {perm_energies.std():.2f}
  Energy ratio:          {ratio:.4f} ({(1-ratio)*100:.1f}% smoother)
  z-score:               {z_score:.2f}
  p-value:               {'< ' + str(round(1/N_PERMS, 4)) if p_value == 0 else str(round(p_value, 4))}

Per-genre breakdown:
  Smoother on music graph: {n_smoother}/{n_valid_dims} genres
  Similar:                 {n_same}/{n_valid_dims} genres
  Rougher on music graph:  {n_rougher}/{n_valid_dims} genres

Conclusion:
  The book genre distribution signal has significantly lower Dirichlet
  energy on the music similarity graph compared to random permutations
  with identical edge weights (p {'< ' + str(round(1/N_PERMS, 4)) if p_value == 0 else '= ' + str(round(p_value, 4))}, z = {z_score:.2f}).
  This confirms that users with similar music preferences also tend
  to have similar book genre preferences, validating the use of
  graph-based methods (GCN) for cross-domain prediction.
""")
