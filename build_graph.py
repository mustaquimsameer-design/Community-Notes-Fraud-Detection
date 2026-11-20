"""
build_graph.py
--------------
Streaming ingestion and degree-filtering pipeline for Community Notes ratings data.
Constructs a bipartite interaction graph (raters x notes) and prunes it to a k-core
(default min-degree = 5), saving a memory-efficient sparse CSR matrix and index tables.
"""

import os
import gc
import sys
import time
import json
import glob
import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import polars as pl


def parse_args():
    parser = argparse.ArgumentParser(description="Build and filter Community Notes bipartite graph.")
    parser.add_argument("--data-dir", type=str, default="Data", help="Path to Data directory containing ratings TSVs")
    parser.add_argument("--output-dir", type=str, default="data_cache", help="Path to output cache directory")
    parser.add_argument("--min-degree", type=int, default=5, help="Minimum degree for raters and notes (5-core filter)")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process (for testing)")
    return parser.parse_args()


def get_process_memory_mb():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tsv_files = sorted(glob.glob(str(data_dir / "ratings-*.tsv")))
    if not tsv_files:
        print(f"Error: No ratings-*.tsv files found in {data_dir}")
        sys.exit(1)

    if args.max_files:
        tsv_files = tsv_files[:args.max_files]
        print(f"Testing mode enabled: Processing first {len(tsv_files)} file(s).")
    else:
        print(f"Found {len(tsv_files)} ratings file(s) to process.")

    start_time = time.time()
    total_raw_rows = 0

    rater_to_id = {}
    note_to_id = {}
    rater_list = []
    note_list = []

    row_chunks = []
    col_chunks = []

    print("\n--- Step 1: Streaming Ingestion & ID Mapping ---")
    for f_idx, file_path in enumerate(tsv_files, 1):
        f_start = time.time()
        file_size_gb = os.path.getsize(file_path) / (1024 ** 3)
        print(f"[{f_idx}/{len(tsv_files)}] Reading {os.path.basename(file_path)} ({file_size_gb:.2f} GB)...", end="", flush=True)

        df = pl.read_csv(
            file_path,
            separator="\t",
            columns=["raterParticipantId", "noteId"],
            schema_overrides={"raterParticipantId": pl.Utf8, "noteId": pl.Utf8},
        )
        raw_count = len(df)
        total_raw_rows += raw_count

        # Deduplicate within chunk
        df = df.unique()
        chunk_raters = df["raterParticipantId"].to_list()
        chunk_notes = df["noteId"].to_list()
        del df
        gc.collect()

        num_edges = len(chunk_raters)
        r_chunk = np.empty(num_edges, dtype=np.int32)
        n_chunk = np.empty(num_edges, dtype=np.int32)

        for i in range(num_edges):
            r = chunk_raters[i]
            n = chunk_notes[i]

            ri = rater_to_id.get(r)
            if ri is None:
                ri = len(rater_list)
                rater_to_id[r] = ri
                rater_list.append(r)

            ni = note_to_id.get(n)
            if ni is None:
                ni = len(note_list)
                note_to_id[n] = ni
                note_list.append(n)

            r_chunk[i] = ri
            n_chunk[i] = ni

        row_chunks.append(r_chunk)
        col_chunks.append(n_chunk)

        f_elapsed = time.time() - f_start
        print(f" Done ({raw_count:,} rows in {f_elapsed:.1f}s | Unique raters: {len(rater_list):,}, notes: {len(note_list):,})")

    # Free dictionary lookups to conserve memory
    del rater_to_id
    del note_to_id
    gc.collect()

    print(f"\nCompleted streaming. Total raw rows scanned: {total_raw_rows:,}")
    print(f"Discovered {len(rater_list):,} unique raters and {len(note_list):,} unique notes.")

    print("\n--- Step 2: Assembling Global Sparse Matrix ---")
    cat_start = time.time()
    rows = np.concatenate(row_chunks)
    cols = np.concatenate(col_chunks)
    del row_chunks
    del col_chunks
    gc.collect()

    num_raters_raw = len(rater_list)
    num_notes_raw = len(note_list)
    raw_edges = len(rows)

    print(f"Building initial CSR matrix ({num_raters_raw:,} x {num_notes_raw:,} with {raw_edges:,} entries)...")
    mat = sp.csr_matrix(
        (np.ones(raw_edges, dtype=np.uint8), (rows, cols)),
        shape=(num_raters_raw, num_notes_raw),
        dtype=np.uint8,
    )
    del rows
    del cols
    gc.collect()

    # Binarize matrix (in case cross-file duplicate ratings exist)
    mat.data.fill(1)
    unique_edges_raw = mat.nnz
    print(f"CSR matrix assembled in {time.time() - cat_start:.1f}s. Unique edges: {unique_edges_raw:,}")

    print(f"\n--- Step 3: Bipartite {args.min_degree}-Core Pruning ---")
    prune_start = time.time()
    active_raters = np.ones(num_raters_raw, dtype=bool)
    active_notes = np.ones(num_notes_raw, dtype=bool)

    iteration = 0
    while True:
        iteration += 1
        r_deg = np.diff(mat.indptr)
        c_deg = np.bincount(mat.indices, minlength=mat.shape[1])

        valid_r = r_deg >= args.min_degree
        valid_c = c_deg >= args.min_degree

        dropped_r = int(np.sum(~valid_r))
        dropped_c = int(np.sum(~valid_c))

        print(f"Iteration {iteration}: dropped {dropped_r:,} raters, {dropped_c:,} notes | Remaining: {mat.shape[0] - dropped_r:,} x {mat.shape[1] - dropped_c:,}")

        if dropped_r == 0 and dropped_c == 0:
            print(f"Core pruning converged in {iteration} iteration(s) ({time.time() - prune_start:.1f}s).")
            break

        # Slicing CSR matrix
        mat = mat[valid_r, :][:, valid_c]

        # Update boolean tracking arrays
        active_raters[active_raters] = valid_r
        active_notes[active_notes] = valid_c

    # Ensure format and clean up
    mat.eliminate_zeros()

    surviving_rater_indices = np.where(active_raters)[0]
    surviving_note_indices = np.where(active_notes)[0]

    surviving_raters = [rater_list[i] for i in surviving_rater_indices]
    surviving_notes = [note_list[j] for j in surviving_note_indices]

    del rater_list
    del note_list
    del active_raters
    del active_notes
    gc.collect()

    num_raters_filtered = mat.shape[0]
    num_notes_filtered = mat.shape[1]
    filtered_edges = mat.nnz

    density_raw = unique_edges_raw / (num_raters_raw * num_notes_raw) if num_raters_raw and num_notes_raw else 0.0
    density_filtered = filtered_edges / (num_raters_filtered * num_notes_filtered) if num_raters_filtered and num_notes_filtered else 0.0

    print("\n--- Summary Statistics ---")
    print(f"Raters: {num_raters_raw:,} -> {num_raters_filtered:,} ({num_raters_filtered/num_raters_raw*100:.1f}%)")
    print(f"Notes:  {num_notes_raw:,} -> {num_notes_filtered:,} ({num_notes_filtered/num_notes_raw*100:.1f}%)")
    print(f"Edges:  {unique_edges_raw:,} -> {filtered_edges:,} ({filtered_edges/unique_edges_raw*100:.1f}%)")
    print(f"Filtered graph density: {density_filtered:.6e} (vs raw {density_raw:.6e})")

    print("\n--- Step 4: Saving Artifacts to Cache ---")
    save_start = time.time()

    matrix_path = output_dir / "bipartite_matrix.npz"
    print(f"Saving sparse matrix to {matrix_path}...", end="", flush=True)
    sp.save_npz(matrix_path, mat)
    matrix_size_mb = os.path.getsize(matrix_path) / (1024 * 1024)
    print(f" Done ({matrix_size_mb:.1f} MB)")

    rater_index_path = output_dir / "rater_index.parquet"
    print(f"Saving rater index to {rater_index_path}...", end="", flush=True)
    pl.DataFrame({
        "rater_idx": np.arange(num_raters_filtered, dtype=np.int32),
        "raterParticipantId": surviving_raters,
    }).write_parquet(rater_index_path)
    print(" Done")

    note_index_path = output_dir / "note_index.parquet"
    print(f"Saving note index to {note_index_path}...", end="", flush=True)
    pl.DataFrame({
        "note_idx": np.arange(num_notes_filtered, dtype=np.int32),
        "noteId": surviving_notes,
    }).write_parquet(note_index_path)
    print(" Done")

    total_elapsed = time.time() - start_time
    stats = {
        "raw_rows_scanned": total_raw_rows,
        "raw_unique_raters": num_raters_raw,
        "raw_unique_notes": num_notes_raw,
        "raw_edges_deduped": unique_edges_raw,
        "min_degree": args.min_degree,
        "filtered_raters": num_raters_filtered,
        "filtered_notes": num_notes_filtered,
        "filtered_edges": filtered_edges,
        "density_raw": density_raw,
        "density_filtered": density_filtered,
        "total_elapsed_seconds": round(total_elapsed, 2),
    }

    stats_path = output_dir / "graph_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")

    print(f"\nPipeline finished successfully in {total_elapsed:.1f}s ({total_elapsed/60:.2f} min).")


if __name__ == "__main__":
    main()
