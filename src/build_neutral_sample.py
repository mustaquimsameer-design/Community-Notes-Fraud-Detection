"""
build_neutral_sample.py
-----------------------
Generates a neutral control baseline of 10,000 randomly sampled notes and 10,000 randomly
sampled users from the Community Notes dataset with identical column schemas to:
- outputs/manual_inspection_notes.csv -> outputs/neutral_sample_notes.csv
- outputs/manual_inspection_users.csv -> outputs/neutral_sample_users.csv
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
    parser = argparse.ArgumentParser(description="Build neutral control baseline datasets.")
    parser.add_argument("--data-dir", type=str, default="Data", help="Path to Data directory")
    parser.add_argument("--cache-dir", type=str, default="data_cache", help="Path to cache directory")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Path to outputs directory")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of notes and users to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
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
    cache_dir = Path(args.cache_dir)
    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {args.sample_size:,} neutral notes and {args.sample_size:,} neutral users (seed={args.seed})...")

    # Step 1: Draw Random Samples from the Filtered Graph Population (>= 5 ratings)
    note_index_path = cache_dir / "note_index.parquet"
    rater_index_path = cache_dir / "rater_index.parquet"

    if not note_index_path.exists() or not rater_index_path.exists():
        print("Error: Missing note_index.parquet or rater_index.parquet in data_cache/")
        return

    n_df = pl.read_parquet(note_index_path)
    r_df = pl.read_parquet(rater_index_path)

    sample_notes_df = n_df.sample(n=args.sample_size, seed=args.seed).select(["noteId"])
    sample_raters_df = r_df.sample(n=args.sample_size, seed=args.seed).select(["raterParticipantId"])

    sample_notes_set = set(sample_notes_df["noteId"])
    sample_raters_set = set(sample_raters_df["raterParticipantId"])

    print(f"Selected {len(sample_notes_set):,} neutral notes and {len(sample_raters_set):,} neutral raters.")

    # Step 2: Stream Ratings TSVs to Accumulate Rating Counts and Helpful Counts
    print("\n--- Streaming Ratings TSVs to Gather Baseline Platform Ratings ---")
    rating_files = sorted(glob.glob(str(data_dir / "ratings-*.tsv")))
    t_stream_start = time.time()

    note_chunks = []
    user_chunks = []

    for f_idx, f_path in enumerate(rating_files, 1):
        f_start = time.time()
        print(f"[{f_idx}/{len(rating_files)}] Scanning {os.path.basename(f_path)}...", end="", flush=True)

        r_df = pl.read_csv(
            f_path,
            separator="\t",
            columns=["raterParticipantId", "noteId", "helpful", "notHelpful", "helpfulnessLevel"],
            schema_overrides={"raterParticipantId": pl.Utf8, "noteId": pl.Utf8},
        )
        r_df = r_df.with_columns(
            ((pl.col("helpfulnessLevel") == "HELPFUL") | (pl.col("helpful") == 1)).cast(pl.Int32).alias("is_helpful")
        )

        # Filter for sampled notes
        matched_notes = r_df.filter(pl.col("noteId").is_in(sample_notes_set))
        if len(matched_notes) > 0:
            n_agg = matched_notes.group_by("noteId").agg([
                pl.len().alias("ratings_count"),
                pl.col("is_helpful").sum().alias("helpful_count"),
            ])
            note_chunks.append(n_agg)

        # Filter for sampled raters
        matched_raters = r_df.filter(pl.col("raterParticipantId").is_in(sample_raters_set))
        if len(matched_raters) > 0:
            u_agg = matched_raters.group_by("raterParticipantId").agg([
                pl.len().alias("ratings_count"),
                pl.col("is_helpful").sum().alias("helpful_count"),
            ])
            user_chunks.append(u_agg)

        del r_df
        print(f" Done ({len(matched_notes):,} note ratings, {len(matched_raters):,} user ratings in {time.time()-f_start:.1f}s)")

    print(f"\nRatings streaming scan completed in {time.time()-t_stream_start:.1f}s.")

    # Consolidate counts
    final_note_counts = (
        pl.concat(note_chunks)
        .group_by("noteId")
        .agg([
            pl.col("ratings_count").sum(),
            pl.col("helpful_count").sum(),
        ])
    )
    final_user_counts = (
        pl.concat(user_chunks)
        .group_by("raterParticipantId")
        .agg([
            pl.col("ratings_count").sum(),
            pl.col("helpful_count").sum(),
        ])
    )
    del note_chunks
    del user_chunks

    # Step 3: Build neutral_sample_notes.csv
    print("\n--- Enriching Neutral Notes Dataset ---")
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

    notes_joined = sample_notes_df.join(final_note_counts, on="noteId", how="left").with_columns([
        pl.col("ratings_count").fill_null(0),
        pl.col("helpful_count").fill_null(0),
    ])
    notes_enriched = notes_joined.join(notes_meta_df, on="noteId", how="left")
    del notes_meta_df
    del notes_joined

    notes_enriched = notes_enriched.with_columns([
        pl.lit(0).cast(pl.Int64).alias("subgraph_id"),  # 0 indicates neutral control sample
        pl.lit(args.sample_size).cast(pl.Int64).alias("total_subgraph_raters"),

        # Percent of ratings on this note that were helpful
        pl.when(pl.col("ratings_count") > 0)
        .then((pl.col("helpful_count") / pl.col("ratings_count") * 100).round(2))
        .otherwise(0.0)
        .alias("pct_of_ratings_helpful"),

        # Percent of the baseline sample size that rated this note helpful
        pl.when(pl.col("ratings_count") > 0)
        .then((pl.col("helpful_count") / args.sample_size * 100).round(2))
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
    ]).sort("subgraph_helpful_count", descending=True)

    out_notes_csv = outputs_dir / "neutral_sample_notes.csv"
    final_notes_df.write_csv(out_notes_csv)
    print(f"Saved {final_notes_df.height:,} neutral notes to {out_notes_csv}")

    # Step 4: Build neutral_sample_users.csv
    print("\n--- Enriching Neutral Users Dataset ---")
    enroll_path = data_dir / "userEnrollment-00000.tsv"
    if enroll_path.exists():
        enroll_df = pl.read_csv(
            enroll_path, separator="\t",
            columns=["participantId", "enrollmentState", "successfulRatingNeededToEarnIn", "numberOfTimesEarnedOut", "modelingGroup"],
            schema_overrides={"participantId": pl.Utf8},
        ).unique(subset=["participantId"])
    else:
        enroll_df = None

    users_joined = sample_raters_df.join(final_user_counts, on="raterParticipantId", how="left").with_columns([
        pl.col("ratings_count").fill_null(0),
        pl.col("helpful_count").fill_null(0),
    ])

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

    users_enriched = users_enriched.with_columns([
        pl.lit(0).cast(pl.Int64).alias("subgraph_id"),
        pl.lit(args.sample_size).cast(pl.Int64).alias("total_subgraph_notes"),

        # Percent of this rater's ratings that were helpful
        pl.when(pl.col("ratings_count") > 0)
        .then((pl.col("helpful_count") / pl.col("ratings_count") * 100).round(2))
        .otherwise(0.0)
        .alias("pct_of_ratings_helpful"),

        # Percent of the baseline note size rated helpful by this user
        pl.when(pl.col("ratings_count") > 0)
        .then((pl.col("helpful_count") / args.sample_size * 100).round(2))
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
    ]).sort("subgraph_notes_helpful_count", descending=True)

    out_users_csv = outputs_dir / "neutral_sample_users.csv"
    final_users_df.write_csv(out_users_csv)
    print(f"Saved {final_users_df.height:,} neutral users to {out_users_csv}")

    print("\n--- Neutral Control Datasets Created Successfully ---")
    print(f"1. Notes CSV: {out_notes_csv} ({os.path.getsize(out_notes_csv)/(1024**2):.1f} MB)")
    print(f"2. Users CSV: {out_users_csv} ({os.path.getsize(out_users_csv)/(1024**2):.1f} MB)")


if __name__ == "__main__":
    main()
