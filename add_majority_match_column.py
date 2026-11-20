"""
add_majority_match_column.py
----------------------------
Calculates the majority agreement rate for each user within their subgraph:
What percent of notes within the subgraph did this user rate helpful that have
>50% pct_of_ratings_helpful, and didn't rate helpful that have <50%.

Adds the new column `pct_matched_subgraph_majority` to:
- outputs/manual_inspection_users.csv
- outputs/combined_inspection_users.csv
"""

import os
import sys
import glob
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl


def main():
    outputs_dir = Path("outputs")
    data_dir = Path("Data")

    notes_file = outputs_dir / "manual_inspection_notes.csv"
    users_file = outputs_dir / "manual_inspection_users.csv"
    combined_users_file = outputs_dir / "combined_inspection_users.csv"

    if not notes_file.exists() or not users_file.exists():
        print("Error: Missing manual inspection CSVs in outputs/")
        return

    print("--- Step 1: Loading Notes Majority Direction ---")
    notes_df = pl.read_csv(notes_file, schema_overrides={"noteId": pl.Utf8})
    notes_majority = notes_df.with_columns(
        pl.when(pl.col("pct_of_ratings_helpful") > 50.0).then(1)
        .when(pl.col("pct_of_ratings_helpful") < 50.0).then(0)
        .otherwise(-1).cast(pl.Int8).alias("note_majority")
    ).select(["subgraph_id", "noteId", "note_majority"])

    target_subgraphs = notes_df["subgraph_id"].unique().to_list()
    target_subgraphs.sort()
    print(f"Loaded note majority labels for {len(notes_majority):,} notes across {len(target_subgraphs)} subgraphs.")

    print("\n--- Step 2: Loading Subgraph Edges ---")
    edge_dfs = []
    for k in target_subgraphs:
        edge_path = outputs_dir / f"subgraph_{k:02d}" / "edges.csv"
        if edge_path.exists():
            df_k = pl.read_csv(edge_path, schema_overrides={"noteId": pl.Utf8, "raterParticipantId": pl.Utf8})
            df_k = df_k.with_columns(pl.lit(k).cast(pl.Int16).alias("subgraph_id"))
            edge_dfs.append(df_k)

    all_target_edges = pl.concat(edge_dfs)
    print(f"Loaded {len(all_target_edges):,} edges across {len(target_subgraphs)} subgraphs.")

    # Join edges with note majority
    all_target_edges = all_target_edges.join(notes_majority, on=["subgraph_id", "noteId"], how="inner")
    print(f"Annotated edges with note majority direction ({len(all_target_edges):,} edges).")

    print("\n--- Step 3: Streaming Ratings TSVs to Measure Majority Concordance ---")
    rating_files = sorted(glob.glob(str(data_dir / "ratings-*.tsv")))
    t_start = time.time()
    chunk_aggs = []

    for f_idx, f_path in enumerate(rating_files, 1):
        f_t0 = time.time()
        print(f"[{f_idx}/{len(rating_files)}] Scanning {os.path.basename(f_path)}...", end="", flush=True)

        r_df = pl.read_csv(
            f_path,
            separator="\t",
            columns=["raterParticipantId", "noteId", "helpful", "notHelpful", "helpfulnessLevel"],
            schema_overrides={"raterParticipantId": pl.Utf8, "noteId": pl.Utf8},
        )
        r_df = r_df.with_columns(
            ((pl.col("helpfulnessLevel") == "HELPFUL") | (pl.col("helpful") == 1)).cast(pl.Int8).alias("is_helpful")
        ).select(["raterParticipantId", "noteId", "is_helpful"])

        matched = all_target_edges.join(r_df, on=["raterParticipantId", "noteId"], how="inner")
        del r_df

        # Match condition:
        # (user helpful == 1 AND note_majority == 1) OR (user helpful == 0 AND note_majority == 0)
        matched = matched.with_columns(
            (
                ((pl.col("is_helpful") == 1) & (pl.col("note_majority") == 1))
                | ((pl.col("is_helpful") == 0) & (pl.col("note_majority") == 0))
            ).cast(pl.Int32).alias("is_match")
        )

        agg = matched.group_by(["subgraph_id", "raterParticipantId"]).agg([
            pl.len().alias("ratings_count"),
            pl.col("is_match").sum().alias("matches_count"),
        ])
        chunk_aggs.append(agg)
        del matched

        print(f" Done in {time.time()-f_t0:.1f}s ({agg.height:,} user hits)")

    print(f"\nRatings scan completed in {time.time()-t_start:.1f}s.")

    print("\n--- Step 4: Consolidating Majority Concordance Metrics ---")
    final_user_majority = (
        pl.concat(chunk_aggs)
        .group_by(["subgraph_id", "raterParticipantId"])
        .agg([
            pl.col("ratings_count").sum(),
            pl.col("matches_count").sum(),
        ])
        .with_columns(
            pl.when(pl.col("ratings_count") > 0)
            .then((pl.col("matches_count") / pl.col("ratings_count") * 100).round(2))
            .otherwise(0.0)
            .alias("pct_matched_subgraph_majority")
        )
        .select(["subgraph_id", "raterParticipantId", "pct_matched_subgraph_majority"])
    )

    print(f"Computed majority concordance for {len(final_user_majority):,} users.")

    # Save cache so this calculation never needs to be recomputed
    cache_path = Path("data_cache") / "user_majority_concordance.parquet"
    final_user_majority.write_parquet(cache_path)
    print(f"Saved concordance cache to {cache_path}")

    # =========================================================================
    # STEP 5: UPDATE manual_inspection_users.csv
    # =========================================================================
    print(f"\n--- Step 5: Updating {users_file} ---")
    orig_users_df = pl.read_csv(users_file)
    updated_users_df = orig_users_df.join(
        final_user_majority, on=["subgraph_id", "raterParticipantId"], how="left"
    )

    # Fill any null with 0.0
    updated_users_df = updated_users_df.with_columns(
        pl.col("pct_matched_subgraph_majority").fill_null(0.0)
    )

    # Reorder columns so pct_matched_subgraph_majority is right after pct_of_ratings_helpful
    cols = updated_users_df.columns
    idx = cols.index("pct_of_ratings_helpful")
    new_cols_order = (
        cols[: idx + 1]
        + ["pct_matched_subgraph_majority"]
        + [c for c in cols[idx + 1 :] if c != "pct_matched_subgraph_majority"]
    )
    updated_users_df = updated_users_df.select(new_cols_order)

    # Handle Windows file-lock if Excel has the file open
    try:
        updated_users_df.write_csv(users_file)
        print(f"Successfully updated {users_file} ({len(updated_users_df):,} rows).")
    except (PermissionError, OSError) as e:
        alt_users_file = outputs_dir / "manual_inspection_users_with_majority.csv"
        print(f"Notice: {users_file} is locked by Excel/another program. Saving to {alt_users_file} instead.")
        updated_users_df.write_csv(alt_users_file)
        print(f"Successfully saved to {alt_users_file} ({len(updated_users_df):,} rows).")

    # =========================================================================
    # STEP 6: UPDATE combined_inspection_users.csv IF IT EXISTS
    # =========================================================================
    if combined_users_file.exists():
        print(f"\n--- Step 6: Updating {combined_users_file} ---")
        orig_combined = pl.read_csv(combined_users_file)
        updated_combined = orig_combined.join(
            final_user_majority, on=["subgraph_id", "raterParticipantId"], how="left"
        ).with_columns(pl.col("pct_matched_subgraph_majority").fill_null(0.0))
        updated_combined = updated_combined.select(new_cols_order)
        try:
            updated_combined.write_csv(combined_users_file)
            print(f"Successfully updated {combined_users_file} ({len(updated_combined):,} rows).")
        except (PermissionError, OSError) as e:
            alt_combined_file = outputs_dir / "combined_inspection_users_with_majority.csv"
            print(f"Notice: {combined_users_file} is locked by Excel. Saving to {alt_combined_file} instead.")
            updated_combined.write_csv(alt_combined_file)
            print(f"Successfully saved to {alt_combined_file} ({len(updated_combined):,} rows).")

    # =========================================================================
    # STEP 7: PRINT SUBGRAPH 01 SUMMARY STATS
    # =========================================================================
    print("\n" + "=" * 64)
    print("      SUBGRAPH 01: MAJORITY CONCORDANCE ANALYSIS")
    print("=" * 64)
    sg1_users = updated_users_df.filter(pl.col("subgraph_id") == 1)
    sg1_match = sg1_users["pct_matched_subgraph_majority"]

    print(f"Total SG-01 Raters: {len(sg1_users):,}")
    print(f"Mean Majority Match Rate:   {sg1_match.mean():.2f}%")
    print(f"Median Majority Match Rate: {sg1_match.median():.2f}%")
    print(f"25th Percentile:            {sg1_match.quantile(0.25):.2f}%")
    print(f"75th Percentile:            {sg1_match.quantile(0.75):.2f}%")
    print(f"90th Percentile:            {sg1_match.quantile(0.90):.2f}%")
    print(f"Min:                        {sg1_match.min():.2f}%")
    print(f"Max:                        {sg1_match.max():.2f}%")

    ge_90 = (sg1_match >= 90.0).sum()
    ge_80 = (sg1_match >= 80.0).sum()
    ge_70 = (sg1_match >= 70.0).sum()
    print(f"\nRaters with >= 90% Majority Match: {ge_90:,} ({ge_90 / len(sg1_users) * 100:.1f}%)")
    print(f"Raters with >= 80% Majority Match: {ge_80:,} ({ge_80 / len(sg1_users) * 100:.1f}%)")
    print(f"Raters with >= 70% Majority Match: {ge_70:,} ({ge_70 / len(sg1_users) * 100:.1f}%)")

    # Compare with other subgraphs
    print("\n" + "=" * 64)
    print("   MAJORITY MATCH RATE ACROSS OTHER KEY SUBGRAPHS")
    print("=" * 64)
    for sg_k in [14, 12, 17, 15, 2, 4, 10]:
        sub_k = updated_users_df.filter(pl.col("subgraph_id") == sg_k)
        if len(sub_k) > 0:
            m_rate = sub_k["pct_matched_subgraph_majority"].mean()
            med_rate = sub_k["pct_matched_subgraph_majority"].median()
            print(f"Subgraph {sg_k:02d} ({len(sub_k):,} raters): Mean = {m_rate:.2f}% | Median = {med_rate:.2f}%")


if __name__ == "__main__":
    main()
