"""
mine_subgraphs.py
-----------------
Iterative multi-subgraph mining using FRAUDAR with edge removal.
Extracts top-K (default K=20) suspicious dense subgraphs from the cached bipartite matrix
and exports structured metadata, rater lists, note lists, and edge lists for each.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy.sparse as sp
import polars as pl

from src.fraudar import fraudar_extract, remove_subgraph_edges


def parse_args():
    parser = argparse.ArgumentParser(description="Mine suspicious subgraphs using FRAUDAR.")
    parser.add_argument("--cache-dir", type=str, default="data_cache", help="Path to cache directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Path to output directory")
    parser.add_argument("--k-iterations", type=int, default=20, help="Number of subgraphs to mine (default: 20)")
    parser.add_argument("--min-raters", type=int, default=5, help="Minimum number of raters for a valid subgraph")
    parser.add_argument("--min-notes", type=int, default=3, help="Minimum number of notes for a valid subgraph")
    parser.add_argument("--col-weight-c", type=float, default=5.0, help="FRAUDAR camouflage smoothing constant c")
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = cache_dir / "bipartite_matrix.npz"
    rater_index_path = cache_dir / "rater_index.parquet"
    note_index_path = cache_dir / "note_index.parquet"

    if not matrix_path.exists() or not rater_index_path.exists() or not note_index_path.exists():
        print(f"Error: Missing cached artifacts in {cache_dir}. Please run src/build_graph.py first.")
        return

    print("--- Loading Cached Bipartite Graph ---")
    t0 = time.time()
    csr = sp.load_npz(matrix_path).astype(np.uint8)
    print(f"Loaded CSR matrix: {csr.shape[0]:,} raters x {csr.shape[1]:,} notes with {csr.nnz:,} edges ({time.time()-t0:.2f}s)")

    rater_df = pl.read_parquet(rater_index_path)
    note_df = pl.read_parquet(note_index_path)
    rater_ids = rater_df["raterParticipantId"].to_numpy()
    note_ids = note_df["noteId"].to_numpy()
    print(f"Loaded {len(rater_ids):,} rater IDs and {len(note_ids):,} note IDs.")

    manifest = []
    print(f"\n--- Starting FRAUDAR Mining (Target: K = {args.k_iterations}) ---")

    for k in range(1, args.k_iterations + 1):
        iter_t0 = time.time()
        print(f"\n[Iteration {k}/{args.k_iterations}] Running FRAUDAR peeling on active graph ({csr.nnz:,} edges remaining)...", flush=True)

        sub_r, sub_n, w_density, u_density, edge_count = fraudar_extract(csr, col_weight_c=args.col_weight_c)
        iter_time = time.time() - iter_t0

        num_r = len(sub_r)
        num_n = len(sub_n)

        print(f"Detected core: {num_r:,} raters, {num_n:,} notes, {edge_count:,} edges (peeling took {iter_time:.2f}s)")
        print(f"Weighted density: {w_density:.4f} | Avg-deg density: {u_density:.4f}")

        # Check stopping criteria
        if num_r < args.min_raters or num_n < args.min_notes or edge_count == 0:
            print(f"Stopping early at iteration {k}: Subgraph size ({num_r} raters, {num_n} notes) below thresholds ({args.min_raters} raters, {args.min_notes} notes).")
            break

        bipartite_density = edge_count / (num_r * num_n) if num_r > 0 and num_n > 0 else 0.0

        # Subgraph directory
        sub_dir = output_dir / f"subgraph_{k:02d}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        # Map to original string IDs
        sub_rater_ids = rater_ids[sub_r]
        sub_note_ids = note_ids[sub_n]

        # Extract local edges
        local_sub = csr[sub_r, :][:, sub_n].tocoo()
        edge_r_ids = sub_rater_ids[local_sub.row]
        edge_n_ids = sub_note_ids[local_sub.col]

        # Save raters.csv
        pl.DataFrame({"raterParticipantId": sub_rater_ids}).write_csv(sub_dir / "raters.csv")

        # Save notes.csv
        pl.DataFrame({"noteId": sub_note_ids}).write_csv(sub_dir / "notes.csv")

        # Save edges.csv
        pl.DataFrame({
            "raterParticipantId": edge_r_ids,
            "noteId": edge_n_ids,
        }).write_csv(sub_dir / "edges.csv")

        # Metadata
        meta = {
            "subgraph_id": k,
            "num_raters": num_r,
            "num_notes": num_n,
            "num_edges": edge_count,
            "bipartite_density": round(bipartite_density, 6),
            "unweighted_avg_degree_density": round(u_density, 6),
            "camouflage_weighted_density": round(w_density, 6),
            "peeling_time_seconds": round(iter_time, 2),
        }
        with open(sub_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        manifest.append(meta)
        print(f"Saved artifacts to {sub_dir}/ (Bipartite Density: {bipartite_density*100:.1f}%)")

        # Zero out edges for subsequent iterations
        remove_subgraph_edges(csr, sub_r, sub_n)

    # Save summary manifest
    manifest_df = pl.DataFrame(manifest)
    manifest_df.write_csv(output_dir / "subgraphs_manifest.csv")
    with open(output_dir / "subgraphs_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nMining complete! Extracted {len(manifest)} subgraphs to {output_dir}/")
    print(f"Summary manifest saved to {output_dir / 'subgraphs_manifest.csv'}")


if __name__ == "__main__":
    main()
