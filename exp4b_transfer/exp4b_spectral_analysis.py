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
   
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_PIPELINE_DIR = _Path(__file__).resolve().parent.parent / "pipeline"
if str(_PIPELINE_DIR) not in _sys.path:
    _sys.path.insert(0, str(_PIPELINE_DIR))

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import wasserstein_distance

from recommendation_config import RecommendationConfig
from recommendation_graphs import build_fixed_fused_operator

GRAPH_NAMES = ["source", "target", "fused"]
COLORS = {"source": "#e74c3c", "target": "#3498db", "fused": "#2ecc71"}

def load_operators(config: RecommendationConfig) -> dict[str, sparse.csr_matrix]:
    root = config.results_dir
    source_op = sparse.load_npz(root / config.artifact_names["source_operator"]).tocsr().astype(np.float32)
    target_op = sparse.load_npz(root / config.artifact_names["target_operator"]).tocsr().astype(np.float32)
    fused_op = build_fixed_fused_operator(source_op, target_op, alpha=0.5)
    return {"source": source_op, "target": target_op, "fused": fused_op}

def compute_laplacian_eigenvalues(
    operator: sparse.csr_matrix,
    n_eigs: int,
    graph_name: str,
) -> np.ndarray:
\
\
\
\
\
       
    n = operator.shape[0]
    n_eigs = min(n_eigs, n - 2)

    identity = sparse.eye(n, format="csr", dtype=np.float32)
    laplacian = identity - operator

    laplacian = (laplacian + laplacian.T) / 2.0
    laplacian = laplacian.tocsr()

    print(f"  [{graph_name}] Computing {n_eigs} smallest eigenvalues of L ({n}×{n}) ...", flush=True)
    t0 = time.perf_counter()

    eigenvalues, _ = eigsh(laplacian.astype(np.float64), k=n_eigs, which="SM")
    eigenvalues = np.sort(np.real(eigenvalues))

    elapsed = time.perf_counter() - t0
    print(f"  [{graph_name}] Done in {elapsed:.1f}s — λ range: [{eigenvalues[0]:.6f}, {eigenvalues[-1]:.6f}]")

    return eigenvalues.astype(np.float64)

def compute_spectral_distances(
    eigenvalues: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
                                                                              
    results = {}
    names = list(eigenvalues.keys())
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                ev_a, ev_b = eigenvalues[a], eigenvalues[b]
                min_len = min(len(ev_a), len(ev_b))
                ea, eb = ev_a[:min_len], ev_b[:min_len]

                w1 = float(wasserstein_distance(ea, eb))
                l2_rmse = float(np.sqrt(np.mean((ea - eb) ** 2)))
                l2_total = float(np.linalg.norm(ea - eb))

                key = f"{a}-{b}"
                results[key] = {
                    "wasserstein_1": w1,
                    "l2_rmse": l2_rmse,
                    "l2_total": l2_total,
                    "n_eigenvalues_compared": min_len,
                }
    return results

def compute_operator_distances(
    operators: dict[str, sparse.csr_matrix],
) -> dict[str, float]:
    results = {}
    names = list(operators.keys())
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                diff = operators[a] - operators[b]
                dist = float(sparse.linalg.norm(diff, ord="fro"))
                results[f"{a}-{b}"] = dist
    return results

def plot_eigenvalue_distribution(
    eigenvalues: dict[str, np.ndarray],
    save_path: Path,
) -> None:
                                                      
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in GRAPH_NAMES:
        ev = eigenvalues[name]
        ax.plot(range(len(ev)), ev, color=COLORS[name], label=f"{name} ({len(ev)} eigs)",
                linewidth=1.5, alpha=0.85)

    ax.set_xlabel("Eigenvalue Index (sorted)", fontsize=12)
    ax.set_ylabel("Eigenvalue λ", fontsize=12)
    ax.set_title("Normalized Laplacian Eigenvalue Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

def plot_eigenvalue_zoom(
    eigenvalues: dict[str, np.ndarray],
    n_zoom: int,
    save_path: Path,
) -> None:
                                                                                 
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in GRAPH_NAMES:
        ev = eigenvalues[name][:n_zoom]
        ax.plot(range(len(ev)), ev, color=COLORS[name], label=name,
                linewidth=2, alpha=0.85, marker=".", markersize=3)

    ax.set_xlabel("Eigenvalue Index (sorted)", fontsize=12)
    ax.set_ylabel("Eigenvalue λ", fontsize=12)
    ax.set_title(f"Low-Frequency Eigenvalues (first {n_zoom})", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

def plot_eigenvalue_cdf(
    eigenvalues: dict[str, np.ndarray],
    save_path: Path,
) -> None:
                                                         
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in GRAPH_NAMES:
        ev = eigenvalues[name]
        cdf_y = np.arange(1, len(ev) + 1) / len(ev)
        ax.step(ev, cdf_y, color=COLORS[name], label=name, linewidth=1.5, where="post")

    ax.set_xlabel("Eigenvalue λ", fontsize=12)
    ax.set_ylabel("Cumulative Fraction", fontsize=12)
    ax.set_title("Eigenvalue CDF Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

def plot_spectral_energy(
    eigenvalues: dict[str, np.ndarray],
    save_path: Path,
) -> None:
\
                                              
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in GRAPH_NAMES:
        ev = eigenvalues[name]
        cum_energy = np.cumsum(ev)
        total_energy = cum_energy[-1] if cum_energy[-1] > 0 else 1.0
        frac = cum_energy / total_energy
        ax.plot(range(len(frac)), frac, color=COLORS[name], label=name, linewidth=1.5)

    ax.set_xlabel("Number of Eigenvalues", fontsize=12)
    ax.set_ylabel("Cumulative Energy Fraction", fontsize=12)
    ax.set_title("Spectral Energy Concentration", fontsize=14, fontweight="bold")
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="90% energy")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

def plot_spectral_distance_heatmap(
    spectral_distances: dict[str, dict[str, float]],
    metric_key: str,
    save_path: Path,
) -> None:
                                            
    mat = np.zeros((3, 3), dtype=np.float64)
    for key, dists in spectral_distances.items():
        a, b = key.split("-")
        i, j = GRAPH_NAMES.index(a), GRAPH_NAMES.index(b)
        mat[i, j] = dists[metric_key]
        mat[j, i] = dists[metric_key]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(mat, cmap="Blues", aspect="equal")
    ax.set_xticks(range(3))
    ax.set_xticklabels(GRAPH_NAMES, fontsize=11)
    ax.set_yticks(range(3))
    ax.set_yticklabels(GRAPH_NAMES, fontsize=11)
    ax.set_title(f"Spectral Distance ({metric_key})", fontsize=13, fontweight="bold")

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center", fontsize=10)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Exp4B: Spectral Analysis")
    parser.add_argument("--n-eigs", type=int, default=200, help="Number of eigenvalues to compute (default 200)")
    args = parser.parse_args()

    config = RecommendationConfig()
    out_dir = config.results_dir / "exp4b_transferability"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading graph operators ...")
    operators = load_operators(config)
    n = operators["source"].shape[0]
    print(f"  Graph size: {n} nodes")

    print(f"\n[2/4] Computing Laplacian eigenvalues (k={args.n_eigs}) ...")
    eigenvalues: dict[str, np.ndarray] = {}
    for name in GRAPH_NAMES:
        eigenvalues[name] = compute_laplacian_eigenvalues(
            operators[name], n_eigs=args.n_eigs, graph_name=name,
        )

    print("\n[3/4] Computing spectral and operator distances ...")
    spectral_dists = compute_spectral_distances(eigenvalues)
    operator_dists = compute_operator_distances(operators)

    print("  Spectral distances (Wasserstein-1):")
    for key, dists in sorted(spectral_dists.items()):
        print(f"    {key}: W1={dists['wasserstein_1']:.6f}  L2_RMSE={dists['l2_rmse']:.6f}  "
              f"L2_total={dists['l2_total']:.6f}")

    print("  Operator Frobenius distances:")
    for key, dist in sorted(operator_dists.items()):
        print(f"    {key}: ‖ΔS‖_F = {dist:.4f}")

    print("\n[4/4] Saving results and plots ...")

    json_results = {
        "n_eigenvalues": args.n_eigs,
        "n_nodes": n,
        "eigenvalue_ranges": {
            name: {"min": float(ev.min()), "max": float(ev.max()), "mean": float(ev.mean())}
            for name, ev in eigenvalues.items()
        },
        "spectral_distances": spectral_dists,
        "operator_frobenius_distances": operator_dists,
    }
    json_path = out_dir / "spectral_analysis_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  Saved: {json_path}")

    np.savez(
        out_dir / "eigenvalues.npz",
        source=eigenvalues["source"],
        target=eigenvalues["target"],
        fused=eigenvalues["fused"],
    )
    print(f"  Saved: {out_dir / 'eigenvalues.npz'}")

    plot_eigenvalue_distribution(eigenvalues, out_dir / "eigenvalue_distribution.png")
    plot_eigenvalue_zoom(eigenvalues, n_zoom=50, save_path=out_dir / "eigenvalue_zoom_50.png")
    plot_eigenvalue_cdf(eigenvalues, out_dir / "eigenvalue_cdf.png")
    plot_spectral_energy(eigenvalues, out_dir / "spectral_energy.png")
    plot_spectral_distance_heatmap(spectral_dists, "wasserstein_1", out_dir / "spectral_distance_wasserstein.png")
    plot_spectral_distance_heatmap(spectral_dists, "l2_rmse", out_dir / "spectral_distance_l2.png")

    print("\n✓ Spectral analysis complete!")

if __name__ == "__main__":
    main()
