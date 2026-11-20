"""
build_inspection_subgraphs_21_to_100.py
---------------------------------------
Builds manual inspection CSV files for Subgraphs 21 through 100:
1. Calculates note-level helpfulness, classifications, timestamps, tweet URLs, and summaries.
2. Calculates user-level enrollment status, helpfulness, and majority concordance (pct_matched_subgraph_majority).
3. Outputs:
   - outputs/subgraph_{k:02d}/manual_inspection_notes.csv
   - outputs/subgraph_{k:02d}/manual_inspection_users.csv
   - outputs/manual_inspection_subgraphs_21_to_100/subgraph_{k:02d}_notes.csv
   - outputs/manual_inspection_subgraphs_21_to_100/subgraph_{k:02d}_users.csv
   - outputs/manual_inspection_notes_subgraphs_21_to_100.csv
   - outputs/manual_inspection_users_subgraphs_21_to_100.csv
"""

import os
import sys
import glob
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl


def format_timestamp(ms):
    if ms is None or ms <= 0:
        return ""
    try:
        dt = datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(description="Build manual inspection CSVs for subgraphs 21 to 100.")
    parser.add_argument("--data-dir", type=str, default="Data", help="Path to Data directory")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Path to outputs directory")
    parser.add_argument("--start-k", type=int, default=21, help="Start subgraph ID")
    parser.add_argument("--end-k", type=int, default=100, help="End subgraph ID")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    outputs_dir = Path(args.outputs_dir)

    manifest_path = outputs_dir / "subgraphs_manifest.csv"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found.")
        return

    manifest = pl.read_csv(manifest_path)
    all_k = manifest["subgraph_id"].to_list()
    target_k = [k for k in all_k if args.start_k <= k <= args.end_k]

    if not target_k:
        print(f"No subgraphs found in range [{args.start_k}, {args.end_k}].")
        return

    print(f"Targeting {len(target_k)} subgraphs for manual inspection: Subgraphs {target_k[0]} to {target_k[-1]}")

    subgraph_meta = {}
    for row in manifest.iter_rows(named=True):
        if row["subgraph_id"] in target_k:
            subgraph_meta[row["subgraph_id"]] = {
                "num_raters": row["num_raters"],
                "num_notes": row["num_notes"],
            }

    # =========================================================================
    # STEP 1: LOAD SUBGRAPH EDGES
    # =========================================================================
    print("\n--- Step 1: Loading Subgraph Edges ---")
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

    if not edge_dfs:
        print("Error: No edge files found for target subgraphs.")
        return

    all_target_edges = pl.concat(edge_dfs)
    print(f"Loaded {len(all_target_edges):,} target edges across {len(target_k)} subgraphs in {time.time()-t0:.2f}s.")

    # =========================================================================
    # STEP 2: STREAM RATINGS TSVs TO COLLECT MATCHED RATINGS
    # =========================================================================
    print("\n--- Step 2: Streaming Ratings TSVs to Match Ratings ---")
    rating_files = sorted(glob.glob(str(data_dir / "ratings-*.tsv")))
    t_stream_start = time.time()
    matched_chunks = []

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
            ((pl.col("helpfulnessLevel") == "HELPFUL") | (pl.col("helpful") == 1)).cast(pl.Int8).alias("is_helpful")
        ).select(["raterParticipantId", "noteId", "is_helpful"])

        matched = all_target_edges.join(r_chunk, on=["raterParticipantId", "noteId"], how="inner")
        del r_chunk

        matched_chunks.append(matched)
        f_time = time.time() - f_start
        print(f" Done ({len(matched):,} edge matches in {f_time:.1f}s)")

    all_matched_ratings = pl.concat(matched_chunks)
    del matched_chunks
    del all_target_edges
    print(f"Total matched rating interactions: {len(all_matched_ratings):,} in {time.time()-t_stream_start:.1f}s.")

    # =========================================================================
    # STEP 3: COMPUTE NOTE MAJORITY & USER MAJORITY CONCORDANCE
    # =========================================================================
    print("\n--- Step 3: Computing Note Helpfulness & User Majority Concordance ---")
    # Note level aggregations
    note_ratings_summary = (
        all_matched_ratings.group_by(["subgraph_id", "noteId"])
        .agg([
            pl.len().alias("ratings_count"),
            pl.col("is_helpful").sum().alias("helpful_count"),
        ])
        .with_columns(
            pl.when(pl.col("ratings_count") > 0)
            .then((pl.col("helpful_count") / pl.col("ratings_count") * 100).round(2))
            .otherwise(0.0)
            .alias("pct_of_ratings_helpful")
        )
        .with_columns(
            pl.when(pl.col("pct_of_ratings_helpful") > 50.0).then(1)
            .when(pl.col("pct_of_ratings_helpful") < 50.0).then(0)
            .otherwise(-1).cast(pl.Int8).alias("note_majority")
        )
    )

    # Join note_majority back to compute concordance
    ratings_with_majority = all_matched_ratings.join(
        note_ratings_summary.select(["subgraph_id", "noteId", "note_majority"]),
        on=["subgraph_id", "noteId"],
        how="inner",
    ).with_columns(
        (
            ((pl.col("is_helpful") == 1) & (pl.col("note_majority") == 1))
            | ((pl.col("is_helpful") == 0) & (pl.col("note_majority") == 0))
        ).cast(pl.Int32).alias("is_majority_match")
    )

    # User level aggregations
    user_ratings_summary = (
        ratings_with_majority.group_by(["subgraph_id", "raterParticipantId"])
        .agg([
            pl.len().alias("ratings_count"),
            pl.col("is_helpful").sum().alias("helpful_count"),
            pl.col("is_majority_match").sum().alias("majority_matches_count"),
        ])
        .with_columns([
            pl.when(pl.col("ratings_count") > 0)
            .then((pl.col("helpful_count") / pl.col("ratings_count") * 100).round(2))
            .otherwise(0.0)
            .alias("pct_of_ratings_helpful"),

            pl.when(pl.col("ratings_count") > 0)
            .then((pl.col("majority_matches_count") / pl.col("ratings_count") * 100).round(2))
            .otherwise(0.0)
            .alias("pct_matched_subgraph_majority"),
        ])
    )
    del ratings_with_majority
    del all_matched_ratings

    print(f"Computed stats for {len(note_ratings_summary):,} notes and {len(user_ratings_summary):,} users.")

    # =========================================================================
    # STEP 4: ENRICH NOTE METADATA & BUILD NOTES DATASETS
    # =========================================================================
    print("\n--- Step 4: Enriching Note Metadata ---")
    note_files = sorted(glob.glob(str(data_dir / "notes-*.tsv")))
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

    # Load all notes in target subgraphs
    sub_notes_list = []
    for k in target_k:
        nf = outputs_dir / f"subgraph_{k:02d}" / "notes.csv"
        if nf.exists():
            df_n = pl.read_csv(nf, schema_overrides={"noteId": pl.Utf8})
            df_n = df_n.with_columns(pl.lit(k).cast(pl.Int16).alias("subgraph_id"))
            sub_notes_list.append(df_n)
    all_sub_notes = pl.concat(sub_notes_list)

    notes_enriched = all_sub_notes.join(note_ratings_summary, on=["subgraph_id", "noteId"], how="left")
    notes_enriched = notes_enriched.with_columns([
        pl.col("ratings_count").fill_null(0),
        pl.col("helpful_count").fill_null(0),
        pl.col("pct_of_ratings_helpful").fill_null(0.0),
    ])
    notes_enriched = notes_enriched.join(notes_meta_df, on="noteId", how="left")
    del notes_meta_df

    raters_map = {k: subgraph_meta[k]["num_raters"] for k in target_k}
    notes_enriched = notes_enriched.with_columns(
        pl.col("subgraph_id").replace_strict(raters_map, default=1).alias("total_subgraph_raters")
    )

    notes_enriched = notes_enriched.with_columns([
        pl.when(pl.col("total_subgraph_raters") > 0)
        .then((pl.col("helpful_count") / pl.col("total_subgraph_raters") * 100).round(2))
        .otherwise(0.0)
        .alias("pct_of_subgraph_raters_helpful"),

        pl.when(pl.col("tweetId").is_not_null())
        .then(pl.concat_str([pl.lit("https://x.com/i/web/status/"), pl.col("tweetId")]))
        .otherwise(pl.lit(""))
        .alias("tweet_url"),

        pl.col("createdAtMillis").map_elements(format_timestamp, return_dtype=pl.Utf8).alias("timestamp_utc"),
    ])

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

    # =========================================================================
    # STEP 5: ENRICH USER ENROLLMENT & BUILD USERS DATASETS
    # =========================================================================
    print("\n--- Step 5: Enriching User Enrollment ---")
    enroll_path = data_dir / "userEnrollment-00000.tsv"
    if enroll_path.exists():
        enroll_df = pl.read_csv(
            enroll_path, separator="\t",
            columns=["participantId", "enrollmentState", "successfulRatingNeededToEarnIn", "numberOfTimesEarnedOut", "modelingGroup"],
            schema_overrides={"participantId": pl.Utf8},
        ).unique(subset=["participantId"])
    else:
        enroll_df = None

    sub_raters_list = []
    for k in target_k:
        rf = outputs_dir / f"subgraph_{k:02d}" / "raters.csv"
        if rf.exists():
            df_r = pl.read_csv(rf, schema_overrides={"raterParticipantId": pl.Utf8})
            df_r = df_r.with_columns(pl.lit(k).cast(pl.Int16).alias("subgraph_id"))
            sub_raters_list.append(df_r)
    all_sub_raters = pl.concat(sub_raters_list)

    users_enriched = all_sub_raters.join(user_ratings_summary, on=["subgraph_id", "raterParticipantId"], how="left")
    users_enriched = users_enriched.with_columns([
        pl.col("ratings_count").fill_null(0),
        pl.col("helpful_count").fill_null(0),
        pl.col("pct_of_ratings_helpful").fill_null(0.0),
        pl.col("pct_matched_subgraph_majority").fill_null(0.0),
    ])

    if enroll_df is not None:
        users_enriched = users_enriched.join(enroll_df, left_on="raterParticipantId", right_on="participantId", how="left")
        del enroll_df
    else:
        users_enriched = users_enriched.with_columns([
            pl.lit("UNKNOWN").alias("enrollmentState"),
            pl.lit(None).cast(pl.Int32).alias("successfulRatingNeededToEarnIn"),
            pl.lit(None).cast(pl.Int32).alias("numberOfTimesEarnedOut"),
            pl.lit(None).cast(pl.Float64).alias("modelingGroup"),
        ])

    notes_map = {k: subgraph_meta[k]["num_notes"] for k in target_k}
    users_enriched = users_enriched.with_columns(
        pl.col("subgraph_id").replace_strict(notes_map, default=1).alias("total_subgraph_notes")
    )

    users_enriched = users_enriched.with_columns([
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
        pl.col("pct_matched_subgraph_majority"),
        pl.col("total_subgraph_notes"),
        pl.col("pct_of_subgraph_notes_rated_helpful"),
        pl.col("successfulRatingNeededToEarnIn"),
        pl.col("numberOfTimesEarnedOut"),
        pl.col("modelingGroup"),
    ]).sort(["subgraph_id", "subgraph_notes_helpful_count"], descending=[False, True])

    # =========================================================================
    # STEP 6: EXPORT INSPECTION FILES
    # =========================================================================
    print("\n--- Step 6: Exporting Inspection Files ---")
    separate_dir = outputs_dir / "manual_inspection_subgraphs_21_to_100"
    separate_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-subgraph CSVs
    for k in target_k:
        sub_n = final_notes_df.filter(pl.col("subgraph_id") == k)
        sub_u = final_users_df.filter(pl.col("subgraph_id") == k)

        # In outputs/subgraph_{k:02d}/
        k_dir = outputs_dir / f"subgraph_{k:02d}"
        k_dir.mkdir(parents=True, exist_ok=True)
        sub_n.write_csv(k_dir / "manual_inspection_notes.csv")
        sub_u.write_csv(k_dir / "manual_inspection_users.csv")

        # In dedicated outputs/manual_inspection_subgraphs_21_to_100/
        sub_n.write_csv(separate_dir / f"subgraph_{k:02d}_notes.csv")
        sub_u.write_csv(separate_dir / f"subgraph_{k:02d}_users.csv")

    print(f"Exported separate CSVs for each of the {len(target_k)} subgraphs to:")
    print(f"  - outputs/subgraph_XX/manual_inspection_notes.csv & manual_inspection_users.csv")
    print(f"  - outputs/manual_inspection_subgraphs_21_to_100/subgraph_XX_notes.csv & subgraph_XX_users.csv")

    # 2. Combined master files for subgraphs 21 to 100
    master_notes_path = outputs_dir / "manual_inspection_notes_subgraphs_21_to_100.csv"
    master_users_path = outputs_dir / "manual_inspection_users_subgraphs_21_to_100.csv"
    final_notes_df.write_csv(master_notes_path)
    final_users_df.write_csv(master_users_path)
    print(f"\nExported combined batch files:")
    print(f"  - {master_notes_path} ({len(final_notes_df):,} rows)")
    print(f"  - {master_users_path} ({len(final_users_df):,} rows)")


if __name__ == "__main__":
    main()
