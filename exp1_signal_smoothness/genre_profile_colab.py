from huggingface_hub import hf_hub_download
import os

META_DIR = "/content/amazon_metadata"
os.makedirs(META_DIR, exist_ok=True)

REV = "6a62d7bf8b3f3a90943ac9c6ea800ae7736959ad"

print("Downloading CDs metadata...")
cd_meta_path = hf_hub_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023",
    repo_type="dataset",
    filename="raw_meta_CDs_and_Vinyl/meta_CDs_and_Vinyl.jsonl",
    revision=REV,
    local_dir=META_DIR,
)
print(f"Done: {cd_meta_path} ({os.path.getsize(cd_meta_path) / 1e6:.1f} MB)")

print("Downloading Books metadata...")
book_meta_path = hf_hub_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023",
    repo_type="dataset",
    filename="raw_meta_Books/meta_Books.jsonl",
    revision=REV,
    local_dir=META_DIR,
)
print(f"Done: {book_meta_path} ({os.path.getsize(book_meta_path) / 1e6:.1f} MB)")

import json
import pandas as pd

def load_metadata(jsonl_path):
    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                records.append({
                    "parent_asin": obj.get("parent_asin"),
                    "main_category": obj.get("main_category"),
                    "categories": obj.get("categories"),
                })
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records)

print("Loading CDs metadata...")
cd_meta = load_metadata(cd_meta_path)
print(f"Shape: {cd_meta.shape}")
print(f"main_category distribution:\n{cd_meta['main_category'].value_counts().head(10)}\n")
print(f"Sample categories field:\n{cd_meta['categories'].head(3).tolist()}\n")

print("=" * 60)
print("Loading Books metadata...")
book_meta = load_metadata(book_meta_path)
print(f"Shape: {book_meta.shape}")
print(f"main_category distribution:\n{book_meta['main_category'].value_counts().head(10)}\n")
print(f"Sample categories field:\n{book_meta['categories'].head(3).tolist()}")

def extract_genres(cat_field, max_depth=3):
    if cat_field is None:
        return []
    if isinstance(cat_field, str):
        return [cat_field]
    if isinstance(cat_field, list):
        genres = []
        for item in cat_field:
            if isinstance(item, str):
                genres.append(item)
            elif isinstance(item, list):
                genres.extend(item[:max_depth])
        return list(set(genres))
    return []

cd_meta["genres"] = cd_meta["categories"].apply(extract_genres)
book_meta["genres"] = book_meta["categories"].apply(extract_genres)

cd_coverage = (cd_meta["genres"].apply(len) > 0).mean()
book_coverage = (book_meta["genres"].apply(len) > 0).mean()
print(f"CDs with genre labels:  {cd_coverage:.1%}")
print(f"Books with genre labels: {book_coverage:.1%}")

from collections import Counter

cd_genre_counter = Counter()
for g in cd_meta["genres"]:
    cd_genre_counter.update(g)
print(f"\nUnique CD genres: {len(cd_genre_counter)}")
print(f"Top 15 CD genres: {cd_genre_counter.most_common(15)}")

book_genre_counter = Counter()
for g in book_meta["genres"]:
    book_genre_counter.update(g)
print(f"\nUnique Book genres: {len(book_genre_counter)}")
print(f"Top 15 Book genres: {book_genre_counter.most_common(15)}")

import numpy as np

DATA_DIR = "/content/amazon_cd_book_thr10_targetItem2"

source_df = pd.read_csv(os.path.join(DATA_DIR, "source_interactions.csv"))
target_df = pd.read_csv(os.path.join(DATA_DIR, "target_interactions.csv"))

print(f"Source interactions: {len(source_df)}")
print(f"Target interactions: {len(target_df)}")

cd_genre_map = cd_meta[["parent_asin", "genres"]].copy()
book_genre_map = book_meta[["parent_asin", "genres"]].copy()

source_g = source_df.merge(cd_genre_map, on="parent_asin", how="left")
target_g = target_df.merge(book_genre_map, on="parent_asin", how="left")

source_g["genres"] = source_g["genres"].apply(lambda x: x if isinstance(x, list) else [])
target_g["genres"] = target_g["genres"].apply(lambda x: x if isinstance(x, list) else [])

src_with = (source_g["genres"].apply(len) > 0).sum()
tgt_with = (target_g["genres"].apply(len) > 0).sum()
print(f"\nSource interactions with genres: {src_with}/{len(source_g)} ({src_with/len(source_g):.1%})")
print(f"Target interactions with genres: {tgt_with}/{len(target_g)} ({tgt_with/len(target_g):.1%})")

def build_genre_profiles(interaction_df, min_genre_freq=50):
\
\
\
       
    genre_counter = Counter()
    for genres in interaction_df["genres"]:
        genre_counter.update(genres)

    valid_genres = sorted([g for g, c in genre_counter.items() if c >= min_genre_freq])
    genre2idx = {g: i for i, g in enumerate(valid_genres)}
    print(f"  Valid genres (freq >= {min_genre_freq}): {len(valid_genres)}")

    user_ids = sorted(interaction_df["user_idx"].unique())
    user2row = {u: i for i, u in enumerate(user_ids)}

    mat = np.zeros((len(user_ids), len(valid_genres)), dtype=np.float32)
    for _, row in interaction_df.iterrows():
        u = user2row.get(row["user_idx"])
        if u is None:
            continue
        for g in row["genres"]:
            if g in genre2idx:
                mat[u, genre2idx[g]] += 1.0

    sums = mat.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    mat = mat / sums

    return mat, valid_genres, user_ids

print("Building MUSIC genre profiles...")
music_prof, music_genres, user_ids = build_genre_profiles(source_g, min_genre_freq=50)
print(f"  Matrix shape: {music_prof.shape}")
print(f"  Density: {(music_prof > 0).mean():.1%}")
print(f"  Sample genres: {music_genres[:10]}")

print("\nBuilding BOOK genre profiles...")
book_prof, book_genres, user_ids_b = build_genre_profiles(target_g, min_genre_freq=50)
print(f"  Matrix shape: {book_prof.shape}")
print(f"  Density: {(book_prof > 0).mean():.1%}")
print(f"  Sample genres: {book_genres[:10]}")

assert user_ids == user_ids_b, "User mismatch!"
print(f"\nTotal aligned users: {len(user_ids)}")

from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

K = 10

print(f"Building MUSIC-BASED k-NN graph (k={K})...")
nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine", algorithm="brute")
nn.fit(music_prof)
distances, indices = nn.kneighbors(music_prof)
similarities = 1 - distances

n = len(user_ids)
rows, cols, vals = [], [], []
for i in range(n):
    for j_pos in range(1, K + 1):
        j = indices[i, j_pos]
        s = similarities[i, j_pos]
        if s > 0:
            rows.append(i)
            cols.append(j)
            vals.append(s)

music_adj = csr_matrix((vals, (rows, cols)), shape=(n, n))
music_adj = music_adj + music_adj.T
music_adj.data = np.minimum(music_adj.data, 1.0)
print(f"  Nodes: {n}, Edges: {music_adj.nnz}, Avg degree: {music_adj.nnz / n:.1f}")

print(f"\nBuilding RANDOM graph (baseline, same edge count)...")
np.random.seed(42)
n_edges = music_adj.nnz
r_rows = np.random.randint(0, n, size=n_edges)
r_cols = np.random.randint(0, n, size=n_edges)
r_vals = np.random.rand(n_edges).astype(np.float32)
random_adj = csr_matrix((r_vals, (r_rows, r_cols)), shape=(n, n))
random_adj = random_adj + random_adj.T
print(f"  Nodes: {n}, Edges: {random_adj.nnz}")

print(f"\nBuilding BOOK-BASED k-NN graph (oracle upper bound, k={K})...")
nn_book = NearestNeighbors(n_neighbors=K + 1, metric="cosine", algorithm="brute")
nn_book.fit(book_prof)
distances_b, indices_b = nn_book.kneighbors(book_prof)
similarities_b = 1 - distances_b

rows_b, cols_b, vals_b = [], [], []
for i in range(n):
    for j_pos in range(1, K + 1):
        j = indices_b[i, j_pos]
        s = similarities_b[i, j_pos]
        if s > 0:
            rows_b.append(i)
            cols_b.append(j)
            vals_b.append(s)

book_adj = csr_matrix((vals_b, (rows_b, cols_b)), shape=(n, n))
book_adj = book_adj + book_adj.T
book_adj.data = np.minimum(book_adj.data, 1.0)
print(f"  Nodes: {n}, Edges: {book_adj.nnz}, Avg degree: {book_adj.nnz / n:.1f}")

from scipy.sparse import diags

def dirichlet_energy(adj, signal):
                                      
    sig = (signal - signal.mean()) / (signal.std() + 1e-8)
    deg = np.array(adj.sum(axis=1)).flatten()
    L = diags(deg) - adj
    E = float(sig @ L @ sig)
    E_norm = E / max(adj.nnz / 2, 1)
    return E, E_norm

avg_book_rating = target_df.groupby("user_idx")["rating"].mean()
book_signal = np.array([avg_book_rating.get(u, 3.0) for u in user_ids], dtype=np.float32)

print("=" * 65)
print("SANITY CHECK 1: Dirichlet Energy of Book Signal on Each Graph")
print("=" * 65)
print(f"Signal: avg book rating per user (mean={book_signal.mean():.3f}, std={book_signal.std():.3f})\n")

E_music, En_music = dirichlet_energy(music_adj, book_signal)
E_random, En_random = dirichlet_energy(random_adj, book_signal)
E_book, En_book = dirichlet_energy(book_adj, book_signal)

print(f"{'Graph':<25} {'Raw Energy':<18} {'Normalized':<15}")
print("-" * 58)
print(f"{'Book graph (oracle)':<25} {E_book:<18.2f} {En_book:<15.6f}")
print(f"{'Music graph':<25} {E_music:<18.2f} {En_music:<15.6f}")
print(f"{'Random graph':<25} {E_random:<18.2f} {En_random:<15.6f}")

ratio = En_random / max(En_music, 1e-8)
print(f"\nRandom / Music ratio = {ratio:.2f}x")

if En_book < En_music < En_random:
    print("\n✅ PERFECT ordering: Book < Music < Random")
    print("   Book signal is smoothest on book graph (expected),")
    print("   smoother on music graph than random → cross-domain signal EXISTS!")
elif En_music < En_random:
    print("\n✅ Music < Random → cross-domain signal exists!")
else:
    print("\n⚠️ Music graph does not show smoothness advantage.")

from sklearn.cluster import KMeans
from scipy.stats import f_oneway

print("=" * 65)
print("SANITY CHECK 2: Do Music Clusters Differ in Book Behavior?")
print("=" * 65)

N_CLUSTERS = 8
print(f"\nK-Means clustering on music profiles (k={N_CLUSTERS})...")
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels = km.fit_predict(music_prof)

print(f"\n{'Cluster':<9} {'Size':<7} {'Avg Book Rating':<17} {'#Books/User':<13} {'Top Book Genres'}")
print("-" * 85)

cluster_ratings = []
for c in range(N_CLUSTERS):
    mask = labels == c
    c_uids = [user_ids[i] for i in range(n) if mask[i]]
    c_books = target_df[target_df["user_idx"].isin(c_uids)]
    avg_r = c_books["rating"].mean() if len(c_books) > 0 else 0
    avg_n = len(c_books) / max(mask.sum(), 1)

    c_book_genres = target_g[target_g["user_idx"].isin(c_uids)]["genres"]
    gc = Counter()
    for gl in c_book_genres:
        if isinstance(gl, list):
            gc.update(gl)
    top3 = [g for g, _ in gc.most_common(3)]

    cluster_ratings.append(c_books["rating"].values)
    print(f"{c:<9} {mask.sum():<7} {avg_r:<17.3f} {avg_n:<13.1f} {top3}")

F, p = f_oneway(*[r for r in cluster_ratings if len(r) > 0])
print(f"\nANOVA: F = {F:.4f}, p = {p:.2e}")
if p < 0.01:
    print("✅ Highly significant (p < 0.01)! Music clusters predict book behavior!")
elif p < 0.05:
    print("✅ Significant (p < 0.05). Some cross-domain signal detected.")
else:
    print("⚠️ Not significant. Cross-domain signal may be weak.")

print("\n--- Detailed Book Genre Profile per Music Cluster ---")
for c in range(min(4, N_CLUSTERS)):
    mask = labels == c
    avg_book = book_prof[mask].mean(axis=0)
    top5 = np.argsort(avg_book)[-5:][::-1]
    top5_info = [(book_genres[i], f"{avg_book[i]:.3f}") for i in top5]
    print(f"  Music Cluster {c} (n={mask.sum()}): {top5_info}")

import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

def compute_gft(adj, signal, n_eig=50, name="Graph"):
    deg = np.array(adj.sum(axis=1)).flatten()
    L = diags(deg) - adj
    print(f"  Computing {n_eig} smallest eigenvalues for {name}...")
    eigvals, eigvecs = eigsh(L.tocsc(), k=n_eig, which="SM")
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    sig = (signal - signal.mean()) / (signal.std() + 1e-8)
    coeffs = eigvecs.T @ sig
    energy = coeffs ** 2
    return eigvals, energy

print("=" * 65)
print("Spectral Analysis: GFT of Book Signal")
print("=" * 65)

N_EIG = min(50, n - 2)

ev_music, se_music = compute_gft(music_adj, book_signal, N_EIG, "Music graph")
ev_random, se_random = compute_gft(random_adj, book_signal, N_EIG, "Random graph")
ev_book, se_book = compute_gft(book_adj, book_signal, N_EIG, "Book graph")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

x = np.arange(N_EIG)
w = 0.28
axes[0].bar(x - w, se_book, w, alpha=0.8, label="Book graph (oracle)", color="#2ecc71")
axes[0].bar(x, se_music, w, alpha=0.8, label="Music graph", color="#3498db")
axes[0].bar(x + w, se_random, w, alpha=0.8, label="Random graph", color="#e74c3c")
axes[0].set_xlabel("Frequency Index (eigenvalue order)", fontsize=12)
axes[0].set_ylabel("Spectral Energy |x̂(λ)|²", fontsize=12)
axes[0].set_title("Book Rating Signal — Spectral Energy Distribution", fontsize=13)
axes[0].legend(fontsize=10)
axes[0].set_xlim(-1, min(30, N_EIG))

cum_book = np.cumsum(se_book) / se_book.sum()
cum_music = np.cumsum(se_music) / se_music.sum()
cum_random = np.cumsum(se_random) / se_random.sum()
axes[1].plot(cum_book, "o-", label="Book graph (oracle)", color="#2ecc71", markersize=4)
axes[1].plot(cum_music, "s-", label="Music graph", color="#3498db", markersize=4)
axes[1].plot(cum_random, "^-", label="Random graph", color="#e74c3c", markersize=4)
axes[1].axhline(y=0.9, color="gray", ls="--", alpha=0.5)
axes[1].text(N_EIG * 0.7, 0.91, "90% energy", color="gray", fontsize=10)
axes[1].set_xlabel("Number of Eigenvectors", fontsize=12)
axes[1].set_ylabel("Cumulative Energy Fraction", fontsize=12)
axes[1].set_title("Cumulative Spectral Energy", fontsize=13)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig("/content/spectral_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

lf_book = se_book[:10].sum() / se_book.sum()
lf_music = se_music[:10].sum() / se_music.sum()
lf_random = se_random[:10].sum() / se_random.sum()
print(f"\nLow-freq energy (first 10 eigenvectors):")
print(f"  Book graph (oracle): {lf_book:.1%}")
print(f"  Music graph:         {lf_music:.1%}")
print(f"  Random graph:        {lf_random:.1%}")

print("\n" + "=" * 65)
print("FINAL VERDICT")
print("=" * 65)

print(f"""
DATA SUMMARY
  Users:              {len(user_ids):,}
  Music genres:       {len(music_genres)}
  Book genres:        {len(book_genres)}
  Music profile density: {(music_prof > 0).mean():.1%}
  Book profile density:  {(book_prof > 0).mean():.1%}

GRAPH SUMMARY
  Music k-NN graph:   {music_adj.nnz:,} edges, avg degree {music_adj.nnz/n:.1f}
  Book k-NN graph:    {book_adj.nnz:,} edges, avg degree {book_adj.nnz/n:.1f}
  Random graph:       {random_adj.nnz:,} edges

CHECK 1 — DIRICHLET ENERGY (lower = smoother)
  Book graph:   {En_book:.6f}
  Music graph:  {En_music:.6f}
  Random graph: {En_random:.6f}
  Random/Music ratio: {ratio:.2f}x

CHECK 2 — ANOVA
  F = {F:.4f}, p = {p:.2e}

CHECK 3 — LOW-FREQ SPECTRAL ENERGY
  Book graph:   {lf_book:.1%}
  Music graph:  {lf_music:.1%}
  Random graph: {lf_random:.1%}
""")

score = 0
if En_music < En_random:
    score += 1
    print("✅ Check 1 passed: Music graph is smoother than random")
else:
    print("❌ Check 1 failed")

if p < 0.05:
    score += 1
    print("✅ Check 2 passed: Music clusters differ in book behavior")
else:
    print("❌ Check 2 failed")

if lf_music > lf_random:
    score += 1
    print("✅ Check 3 passed: More low-freq energy on music graph")
else:
    print("❌ Check 3 failed")

print(f"\nScore: {score}/3")
if score >= 2:
    print("\n🎉 GREEN LIGHT — Cross-domain signal is strong enough!")
    print("   Proceed with GNN experiments confidently.")
elif score == 1:
    print("\n⚠️ YELLOW — Signal exists but weak. Consider:")
    print("   1. Try SVD embedding instead of genre profiles")
    print("   2. Switch source to Movies (may correlate better with Books)")
else:
    print("\n🔴 RED — Signal too weak. Recommend pivoting:")
    print("   1. Use Movies as source domain")
    print("   2. Or switch to a different project direction")

print("\n📌 NEXT STEPS if green light:")
print("   1. Save music_adj as edge_index for PyG")
print("   2. Implement GCN / GAT / ChebNet / GRNN")
print("   3. Run the full experiment matrix")
