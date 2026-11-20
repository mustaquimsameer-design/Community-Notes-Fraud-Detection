"""
mine_additional_subgraphs.py
----------------------------
Continues FRAUDAR mining to extract Subgraphs 21 through 100.
Loads the bipartite matrix, fast-forwards through iterations 1..20 by peeling
and removing edges, and exports subgraphs 21..100 with full metadata and edge lists.
Updates outputs/subgraphs_manifest.csv and outputs/subgraphs_manifest.json.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy.sparse as sp
import polars as pl

from src.fraudar import fraudar_extract, remove_subgraph_edges


def main():
    parser = argparse.ArgumentParser(description="Mine additional subgraphs (21 to 100) using FRAUDAR.")
    parser.add_argument("--cache-dir", type=str, default="data_cache", help="Path to cache directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Path to output directory")
    parser.add_argument("--total-k", type=int, default=100, help="Total subgraphs to mine (default: 100)")
    parser.add_argument("--min-raters", type=int, default=5, help="Minimum number of raters")
    parser.add_argument("--min-notes", type=int, default=3, help="Minimum number of notes")
    parser.add_argument("--col-weight-c", type=float, default=5.0, help="Camouflage smoothing constant c")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = cache_dir / "bipartite_matrix.npz"
    rater_index_path = cache_dir / "rater_index.parquet"
    note_index_path = cache_dir / "note_index.parquet"

    print("--- Loading Cached Bipartite Graph ---")
    t0 = time.time()
    csr = sp.load_npz(matrix_path).astype(np.uint8)
    print(f"Loaded CSR matrix: {csr.shape[0]:,} raters x {csr.shape[1]:,} notes with {csr.nnz:,} edges ({time.time()-t0:.2f}s)")

    rater_df = pl.read_parquet(rater_index_path)
    note_df = pl.read_parquet(note_index_path)
    rater_ids = rater_df["raterParticipantId"].to_numpy()
    note_ids = note_df["noteId"].to_numpy()
    print(f"Loaded {len(rater_ids):,} rater IDs and {len(note_ids):,} note IDs.")

    # Load existing manifest for subgraphs 1..20 if present
    manifest_path = output_dir / "subgraphs_manifest.json"
    manifest = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            existing_manifest = json.load(f)
            # Filter to subgraphs <= 20
            manifest = [m for m in existing_manifest if m.get("subgraph_id", 0) <= 20]
        print(f"Loaded existing manifest with {len(manifest)} subgraphs.")

    print(f"\n--- Starting Iterative Peeling up to K = {args.total_k} ---")
    total_start = time.time()

    for k in range(1, args.total_k + 1):
        iter_t0 = time.time()
        active_nnz = csr.nnz
        sub_r, sub_n, w_density, u_density, edge_count = fraudar_extract(csr, col_weight_c=args.col_weight_c)
        iter_time = time.time() - iter_t0

        num_r = len(sub_r)
        num_n = len(sub_n)

        # Check stopping criteria
        if num_r < args.min_raters or num_n < args.min_notes or edge_count == 0:
            print(f"\n[Iteration {k}] Stopping early: core ({num_r} raters, {num_n} notes, {edge_count} edges) below thresholds.")
            break

        bipartite_density = edge_count / (num_r * num_n) if num_r > 0 and num_n > 0 else 0.0

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

        sub_dir = output_dir / f"subgraph_{k:02d}"
        already_exported = (
            (sub_dir / "edges.csv").exists()
            and (sub_dir / "raters.csv").exists()
            and (sub_dir / "notes.csv").exists()
        )

        if already_exported:
            print(f"[Iter {k:02d}/{args.total_k}] Reusing existing exported subgraph {k} ({num_r:,} raters, {num_n:,} notes, {edge_count:,} edges in {iter_time:.2f}s)")
            if not (sub_dir / "metadata.json").exists():
                with open(sub_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
        else:
            sub_dir.mkdir(parents=True, exist_ok=True)
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

            # Save metadata.json
            with open(sub_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            print(f"[Iter {k:02d}/{args.total_k}] Mined & Exported Subgraph {k:02d}: {num_r:,} raters, {num_n:,} notes, {edge_count:,} edges (Density: {bipartite_density*100:.2f}%, Peeling: {iter_time:.2f}s)")

        # Update or append in manifest
        manifest = [m for m in manifest if m["subgraph_id"] != k]
        manifest.append(meta)

        # Zero out edges between sub_r and sub_n in CSR matrix
        remove_subgraph_edges(csr, sub_r, sub_n)

    # Sort manifest by subgraph_id
    manifest.sort(key=lambda x: x["subgraph_id"])

    # Save manifest files
    manifest_df = pl.DataFrame(manifest)
    manifest_df.write_csv(output_dir / "subgraphs_manifest.csv")
    with open(output_dir / "subgraphs_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_time = time.time() - total_start
    print(f"\n========================================================")
    print(f"Mining completed in {total_time/60:.2f} minutes.")
    print(f"Total subgraphs in manifest: {len(manifest)}")
    print(f"Manifest written to: {output_dir / 'subgraphs_manifest.csv'}")
    print(f"========================================================")


if __name__ == "__main__":
    main()
