"""
build_inspection_datasets.py
----------------------------
Builds two specialized CSV files for manual inspection of detected suspicious subgraphs,
excluding macro/background components (3, 8, 13, 16, 20):
1. outputs/manual_inspection_notes.csv:
   - subgraph_id, writer_user_id, status (classification), timestamp, noteId, tweet_url,
     subgraph_ratings_count, subgraph_helpful_count, pct_of_ratings_helpful,
     pct_of_subgraph_raters_helpful, summary
2. outputs/manual_inspection_users.csv:
   - subgraph_id, raterParticipantId, enrollment_status, subgraph_notes_rated_count,
     subgraph_notes_helpful_count, pct_of_ratings_helpful, pct_of_subgraph_notes_rated_helpful,
     successfulRatingNeededToEarnIn, numberOfTimesEarnedOut
"""

import os
import sys
import glob
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl


def parse_args():
    parser = argparse.ArgumentParser(description="Build manual inspection CSV files.")
    parser.add_argument("--data-dir", type=str, default="Data", help="Path to Data directory")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Path to outputs directory")
    parser.add_argument(
        "--exclude-subgraphs",
        type=int,
        nargs="+",
        default=[3, 8, 13, 16, 20],
        help="Subgraphs to exclude (default: 3 8 13 16 20)",
    )
    return parser.parse_args()


def format_timestamp(ms):
    if ms is None or ms <= 0:
        return ""
    try:
        dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return ""


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    outputs_dir = Path(args.outputs_dir)

    manifest_path = outputs_dir / "subgraphs_manifest.csv"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found. Run src/mine_subgraphs.py first.")
        return

    manifest = pl.read_csv(manifest_path)
    all_k = manifest["subgraph_id"].to_list()
    target_k = [k for k in all_k if k not in args.exclude_subgraphs]

    print(f"Targeting {len(target_k)} subgraphs for manual inspection: {target_k}")
    print(f"Excluded macro components: {args.exclude_subgraphs}\n")

    # Load total rater and note counts per subgraph from manifest for normalization
    subgraph_meta = {}
    for row in manifest.iter_rows(named=True):
        subgraph_meta[row["subgraph_id"]] = {
            "num_raters": row["num_raters"],
            "num_notes": row["num_notes"],
        }

    # =========================================================================
    # STEP 1: LOAD SUBGRAPH EDGES
    # =========================================================================
    print("--- Step 1: Loading Subgraph Edges ---")
    t0 = time.time()
    edge_dfs = []
    for k in target_k:
        edge_file = outputs_dir / f"subgraph_{k:02d}" / "edges.csv"
        if not edge_file.exists():
            print(f"Warning: Missing {edge_file}")
            continue
        df_k = pl.read_csv(edge_file, schema_overrides={"noteId": pl.Utf8, "raterParticipantId": pl.Utf8})
        df_k = df_k.with_columns(pl.lit(k).cast(pl.Int16).alias("subgraph_id"))
        edge_dfs.append(df_k)

    all_target_edges = pl.concat(edge_dfs)
    print(f"Loaded {len(all_target_edges):,} target edges across {len(target_k)} subgraphs in {time.time()-t0:.2f}s.\n")

    # =========================================================================
    # STEP 2: STREAM RATINGS TSVs & ACCUMULATE HELPFULNESS COUNTS
    # =========================================================================
    print("--- Step 2: Streaming Ratings TSVs to Match Helpfulness Signals ---")
    rating_files = sorted(glob.glob(str(data_dir / "ratings-*.tsv")))
    t_stream_start = time.time()

    note_agg_chunks = []
    user_agg_chunks = []

    for f_idx, f_path in enumerate(rating_files, 1):
        f_start = time.time()
        print(f"[{f_idx}/{len(rating_files)}] Processing {os.path.basename(f_path)}...", end="", flush=True)

        r_chunk = pl.read_csv(
            f_path,
            separator="\t",
            columns=["raterParticipantId", "noteId", "helpful", "notHelpful", "helpfulnessLevel"],
            schema_overrides={"raterParticipantId": pl.Utf8, "noteId": pl.Utf8},
        )
        r_chunk = r_chunk.with_columns(
            ((pl.col("helpfulnessLevel") == "HELPFUL") | (pl.col("helpful") == 1)).cast(pl.Int32).alias("is_helpful")
        ).select(["raterParticipantId", "noteId", "is_helpful"])

        # Inner join to find edges belonging to our target subgraphs
        matched = all_target_edges.join(r_chunk, on=["raterParticipantId", "noteId"], how="inner")
        del r_chunk

        # Aggregate per note
        n_agg = matched.group_by(["subgraph_id", "noteId"]).agg([
            pl.len().alias("ratings_count"),
            pl.col("is_helpful").sum().alias("helpful_count"),
        ])
        note_agg_chunks.append(n_agg)

        # Aggregate per user
        u_agg = matched.group_by(["subgraph_id", "raterParticipantId"]).agg([
            pl.len().alias("ratings_count"),
            pl.col("is_helpful").sum().alias("helpful_count"),
        ])
        user_agg_chunks.append(u_agg)

        del matched
        f_time = time.time() - f_start
        print(f" Done ({n_agg.height:,} note hits, {u_agg.height:,} user hits in {f_time:.1f}s)")

    print(f"\nCompleted ratings scan across all files in {time.time()-t_stream_start:.1f}s.")

    # Combine chunk aggregations
    print("Consolidating note and user ratings stats...")
    final_note_counts = (
        pl.concat(note_agg_chunks)
        .group_by(["subgraph_id", "noteId"])
        .agg([
            pl.col("ratings_count").sum(),
            pl.col("helpful_count").sum(),
        ])
    )
    final_user_counts = (
        pl.concat(user_agg_chunks)
        .group_by(["subgraph_id", "raterParticipantId"])
        .agg([
            pl.col("ratings_count").sum(),
            pl.col("helpful_count").sum(),
        ])
    )
    del note_agg_chunks
    del user_agg_chunks
    del all_target_edges

    # =========================================================================
    # STEP 3: BUILD MANUAL INSPECTION - NOTES DATASET
    # =========================================================================
    print("\n--- Step 3: Enriching Note Metadata & Building manual_inspection_notes.csv ---")
    note_files = sorted(glob.glob(str(data_dir / "notes-*.tsv")))
    print(f"Loading note metadata from {len(note_files)} notes file(s)...")
    notes_meta_df = pl.concat([
        pl.read_csv(
            f, separator="\t",
            columns=["noteId", "noteAuthorParticipantId", "createdAtMillis", "tweetId", "classification", "summary"],
            schema_overrides={
                "noteId": pl.Utf8,
                "noteAuthorParticipantId": pl.Utf8,
                "tweetId": pl.Utf8,
                "classification": pl.Utf8,
                "summary": pl.Utf8,
            },
        )
        for f in note_files
    ]).unique(subset=["noteId"])
    print(f"Loaded metadata for {len(notes_meta_df):,} unique notes.")

    # Load all unique notes in target subgraphs to ensure complete coverage
    sub_notes_list = []
    for k in target_k:
        nf = outputs_dir / f"subgraph_{k:02d}" / "notes.csv"
        if nf.exists():
            df_n = pl.read_csv(nf, schema_overrides={"noteId": pl.Utf8})
            df_n = df_n.with_columns(pl.lit(k).cast(pl.Int16).alias("subgraph_id"))
            sub_notes_list.append(df_n)
    all_sub_notes = pl.concat(sub_notes_list)

    # Join with ratings counts
    notes_joined = all_sub_notes.join(final_note_counts, on=["subgraph_id", "noteId"], how="left")
    notes_joined = notes_joined.with_columns([
        pl.col("ratings_count").fill_null(0),
        pl.col("helpful_count").fill_null(0),
    ])

    # Join with notes metadata
    notes_enriched = notes_joined.join(notes_meta_df, on="noteId", how="left")
    del notes_meta_df
    del notes_joined

    # Add total raters per subgraph for normalization
    raters_map = {k: subgraph_meta[k]["num_raters"] for k in target_k}
    notes_enriched = notes_enriched.with_columns(
        pl.col("subgraph_id").replace_strict(raters_map, default=1).alias("total_subgraph_raters")
    )

    # Calculate percentages and URLs
    notes_enriched = notes_enriched.with_columns([
        # Percent of ratings on this note that were helpful
        pl.when(pl.col("ratings_count") > 0)
        .then((pl.col("helpful_count") / pl.col("ratings_count") * 100).round(2))
        .otherwise(0.0)
        .alias("pct_of_ratings_helpful"),

        # Percent of the entire subgraph rater population that rated this note helpful
        pl.when(pl.col("total_subgraph_raters") > 0)
        .then((pl.col("helpful_count") / pl.col("total_subgraph_raters") * 100).round(2))
        .otherwise(0.0)
        .alias("pct_of_subgraph_raters_helpful"),

        # Tweet URL
        pl.when(pl.col("tweetId").is_not_null())
        .then(pl.concat_str([pl.lit("https://x.com/i/web/status/"), pl.col("tweetId")]))
        .otherwise(pl.lit(""))
        .alias("tweet_url"),

        # Formatted timestamp
        pl.col("createdAtMillis").map_elements(format_timestamp, return_dtype=pl.Utf8).alias("timestamp_utc"),
    ])

    # Select and order columns cleanly
    final_notes_df = notes_enriched.select([
        pl.col("subgraph_id"),
        pl.col("noteId"),
        pl.col("noteAuthorParticipantId").fill_null("UNKNOWN").alias("writer_user_id"),
        pl.col("classification").fill_null("UNKNOWN").alias("status"),
        pl.col("timestamp_utc"),
        pl.col("createdAtMillis").alias("timestamp_epoch_ms"),
        pl.col("tweetId").fill_null("").alias("tweet_id"),
        pl.col("tweet_url"),
        pl.col("ratings_count").alias("subgraph_ratings_count"),
        pl.col("helpful_count").alias("subgraph_helpful_count"),
        pl.col("pct_of_ratings_helpful"),
        pl.col("total_subgraph_raters"),
        pl.col("pct_of_subgraph_raters_helpful"),
        pl.col("summary").fill_null("").alias("note_summary"),
    ]).sort(["subgraph_id", "subgraph_helpful_count"], descending=[False, True])

    out_notes_csv = outputs_dir / "manual_inspection_notes.csv"
    final_notes_df.write_csv(out_notes_csv)
    print(f"Saved {final_notes_df.height:,} rows to {out_notes_csv}")

    # =========================================================================
    # STEP 4: BUILD MANUAL INSPECTION - USERS DATASET
    # =========================================================================
    print("\n--- Step 4: Enriching User Enrollment & Building manual_inspection_users.csv ---")
    enroll_path = data_dir / "userEnrollment-00000.tsv"
    if enroll_path.exists():
        print(f"Loading user enrollment from {enroll_path}...")
        enroll_df = pl.read_csv(
            enroll_path, separator="\t",
            columns=["participantId", "enrollmentState", "successfulRatingNeededToEarnIn", "numberOfTimesEarnedOut", "modelingGroup"],
            schema_overrides={"participantId": pl.Utf8},
        ).unique(subset=["participantId"])
    else:
        enroll_df = None

    # Load all unique raters in target subgraphs
    sub_raters_list = []
    for k in target_k:
        rf = outputs_dir / f"subgraph_{k:02d}" / "raters.csv"
        if rf.exists():
            df_r = pl.read_csv(rf, schema_overrides={"raterParticipantId": pl.Utf8})
            df_r = df_r.with_columns(pl.lit(k).cast(pl.Int16).alias("subgraph_id"))
            sub_raters_list.append(df_r)
    all_sub_raters = pl.concat(sub_raters_list)

    # Join with ratings counts
    users_joined = all_sub_raters.join(final_user_counts, on=["subgraph_id", "raterParticipantId"], how="left")
    users_joined = users_joined.with_columns([
        pl.col("ratings_count").fill_null(0),
        pl.col("helpful_count").fill_null(0),
    ])

    # Join with enrollment
    if enroll_df is not None:
        users_enriched = users_joined.join(enroll_df, left_on="raterParticipantId", right_on="participantId", how="left")
        del enroll_df
    else:
        users_enriched = users_joined.with_columns([
            pl.lit("UNKNOWN").alias("enrollmentState"),
            pl.lit(None).cast(pl.Int32).alias("successfulRatingNeededToEarnIn"),
            pl.lit(None).cast(pl.Int32).alias("numberOfTimesEarnedOut"),
            pl.lit(None).cast(pl.Float64).alias("modelingGroup"),
        ])
    del users_joined

    # Add total notes per subgraph for normalization
    notes_map = {k: subgraph_meta[k]["num_notes"] for k in target_k}
    users_enriched = users_enriched.with_columns(
        pl.col("subgraph_id").replace_strict(notes_map, default=1).alias("total_subgraph_notes")
    )

    # Calculate percentages
    users_enriched = users_enriched.with_columns([
        # Percent of this rater's ratings in the subgraph that were helpful
        pl.when(pl.col("ratings_count") > 0)
        .then((pl.col("helpful_count") / pl.col("ratings_count") * 100).round(2))
        .otherwise(0.0)
        .alias("pct_of_ratings_helpful"),

        # Percent of ALL notes in the subgraph that this user rated helpful
        pl.when(pl.col("total_subgraph_notes") > 0)
        .then((pl.col("helpful_count") / pl.col("total_subgraph_notes") * 100).round(2))
        .otherwise(0.0)
        .alias("pct_of_subgraph_notes_rated_helpful"),
    ])

    final_users_df = users_enriched.select([
        pl.col("subgraph_id"),
        pl.col("raterParticipantId"),
        pl.col("enrollmentState").fill_null("UNKNOWN").alias("enrollment_status"),
        pl.col("ratings_count").alias("subgraph_notes_rated_count"),
        pl.col("helpful_count").alias("subgraph_notes_helpful_count"),
        pl.col("pct_of_ratings_helpful"),
        pl.col("total_subgraph_notes"),
        pl.col("pct_of_subgraph_notes_rated_helpful"),
        pl.col("successfulRatingNeededToEarnIn"),
        pl.col("numberOfTimesEarnedOut"),
        pl.col("modelingGroup"),
    ]).sort(["subgraph_id", "subgraph_notes_helpful_count"], descending=[False, True])

    out_users_csv = outputs_dir / "manual_inspection_users.csv"
    final_users_df.write_csv(out_users_csv)
    print(f"Saved {final_users_df.height:,} rows to {out_users_csv}")

    print("\n--- Manual Inspection Datasets Generated Successfully ---")
    print(f"1. Notes CSV: {out_notes_csv} ({os.path.getsize(out_notes_csv)/(1024**2):.1f} MB)")
    print(f"2. Users CSV: {out_users_csv} ({os.path.getsize(out_users_csv)/(1024**2):.1f} MB)")


if __name__ == "__main__":
    main()
