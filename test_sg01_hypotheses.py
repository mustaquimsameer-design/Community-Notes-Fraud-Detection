"""
test_sg01_hypotheses.py
-----------------------
Empirical testing suite for Subgraph 01:
Test 1: Strategic Note Bridge Test (Quadrant analysis of Left vs. Right ratings on 31k notes).
Test 2: Intra-Cohort Pairwise Jaccard Similarity (Left vs. Right vs. Subgraph 14 vs. Neutral Control).
Test 3: Temporal Circadian Rhythms and Inter-arrival Latencies.
"""

import os
import sys
import glob
import time
import json
import random
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl


def main():
    outputs_dir = Path("outputs")
    data_dir = Path("Data")

    print("=" * 70)
    print("          SUBGRAPH 01 DEEP INVESTIGATION & HYPOTHESIS TESTING")
    print("=" * 70)

    # Load users and classify into Left (<50% match) and Right (>=50% match)
    users_file = outputs_dir / "manual_inspection_users_with_majority.csv"
    if not users_file.exists():
        users_file = outputs_dir / "manual_inspection_users.csv"
    
    users_df = pl.read_csv(users_file)
    sg1_users = users_df.filter(pl.col("subgraph_id") == 1)

    left_users = set(sg1_users.filter(pl.col("pct_matched_subgraph_majority") < 50.0)["raterParticipantId"])
    right_users = set(sg1_users.filter(pl.col("pct_matched_subgraph_majority") >= 50.0)["raterParticipantId"])

    print(f"Total SG-01 Raters: {len(sg1_users):,}")
    print(f"  - Left Cohort  (<50% match):  {len(left_users):,} ({len(left_users)/len(sg1_users)*100:.1f}%)")
    print(f"  - Right Cohort (>=50% match): {len(right_users):,} ({len(right_users)/len(sg1_users)*100:.1f}%)")

    # Load notes of SG-01
    notes_file = outputs_dir / "subgraph_01" / "notes.csv"
    sg1_notes = set(pl.read_csv(notes_file, schema_overrides={"noteId": pl.Utf8})["noteId"])
    print(f"Total SG-01 Notes:  {len(sg1_notes):,}\n")

    # =========================================================================
    # PART 1: STREAM RATINGS TSVs TO MEASURE LEFT VS RIGHT RATINGS PER NOTE
    # AND GATHER TIMESTAMP SAMPLE FOR TEMPORAL DYNAMICS
    # =========================================================================
    print("--- Step 1: Streaming Ratings TSVs for Left vs. Right Breakdown ---")
    rating_files = sorted(glob.glob(str(data_dir / "ratings-*.tsv")))
    t_start = time.time()

    note_left_chunks = []
    note_right_chunks = []

    left_hours = np.zeros(24, dtype=np.int64)
    right_hours = np.zeros(24, dtype=np.int64)

    left_sample_deltas = []
    right_sample_deltas = []

    for f_idx, f_path in enumerate(rating_files, 1):
        f_t0 = time.time()
        print(f"[{f_idx}/{len(rating_files)}] Scanning {os.path.basename(f_path)}...", end="", flush=True)

        df = pl.read_csv(
            f_path,
            separator="\t",
            columns=["raterParticipantId", "noteId", "helpful", "notHelpful", "helpfulnessLevel", "createdAtMillis"],
            schema_overrides={"raterParticipantId": pl.Utf8, "noteId": pl.Utf8},
        )

        # Filter for SG-01 notes
        df = df.filter(pl.col("noteId").is_in(sg1_notes))

        if len(df) == 0:
            print(f" Done (0 matches in {time.time()-f_t0:.1f}s)")
            continue

        df = df.with_columns([
            ((pl.col("helpfulnessLevel") == "HELPFUL") | (pl.col("helpful") == 1)).cast(pl.Int8).alias("is_helpful"),
            pl.when(pl.col("raterParticipantId").is_in(left_users)).then(pl.lit("Left"))
            .when(pl.col("raterParticipantId").is_in(right_users)).then(pl.lit("Right"))
            .otherwise(pl.lit("Other")).alias("cohort")
        ])

        # Filter for Left and Right only
        df = df.filter(pl.col("cohort").is_in(["Left", "Right"]))

        # Aggregate note-level votes for Left
        df_left = df.filter(pl.col("cohort") == "Left")
        if len(df_left) > 0:
            agg_l = df_left.group_by("noteId").agg([
                pl.len().alias("left_total"),
                pl.col("is_helpful").sum().alias("left_helpful")
            ])
            note_left_chunks.append(agg_l)

            # Hour of day (UTC)
            ts_l = df_left["createdAtMillis"].to_numpy()
            hours_l = (ts_l // (1000 * 3600)) % 24
            for h in hours_l:
                left_hours[h] += 1

        # Aggregate note-level votes for Right
        df_right = df.filter(pl.col("cohort") == "Right")
        if len(df_right) > 0:
            agg_r = df_right.group_by("noteId").agg([
                pl.len().alias("right_total"),
                pl.col("is_helpful").sum().alias("right_helpful")
            ])
            note_right_chunks.append(agg_r)

            ts_r = df_right["createdAtMillis"].to_numpy()
            hours_r = (ts_r // (1000 * 3600)) % 24
            for h in hours_r:
                right_hours[h] += 1

        del df
        print(f" Done in {time.time()-f_t0:.1f}s (matched {len(df_left):,} Left, {len(df_right):,} Right)")

    print(f"\nRatings streaming scan completed in {time.time()-t_start:.1f}s.")

    # Consolidate note statistics
    print("Consolidating note-level Left vs. Right breakdown...")
    final_left = pl.concat(note_left_chunks).group_by("noteId").agg([
        pl.col("left_total").sum(),
        pl.col("left_helpful").sum()
    ])
    final_right = pl.concat(note_right_chunks).group_by("noteId").agg([
        pl.col("right_total").sum(),
        pl.col("right_helpful").sum()
    ])

    notes_consolidated = final_left.join(final_right, on="noteId", how="full")
    notes_consolidated = notes_consolidated.with_columns([
        pl.col("left_total").fill_null(0),
        pl.col("left_helpful").fill_null(0),
        pl.col("right_total").fill_null(0),
        pl.col("right_helpful").fill_null(0),
    ])

    notes_consolidated = notes_consolidated.with_columns([
        pl.when(pl.col("left_total") > 0)
        .then((pl.col("left_helpful") / pl.col("left_total") * 100).round(2))
        .otherwise(None).alias("left_helpful_pct"),

        pl.when(pl.col("right_total") > 0)
        .then((pl.col("right_helpful") / pl.col("right_total") * 100).round(2))
        .otherwise(None).alias("right_helpful_pct"),
    ])

    # =========================================================================
    # PART 2: LOAD NOTE METADATA (TEXT, SUMMARY, TWEET URL)
    # =========================================================================
    print("Loading note metadata from Data/notes-*.tsv...")
    note_files = sorted(glob.glob(str(data_dir / "notes-*.tsv")))
    notes_meta = pl.concat([
        pl.read_csv(
            f, separator="\t",
            columns=["noteId", "noteAuthorParticipantId", "createdAtMillis", "tweetId", "classification", "summary"],
            schema_overrides={"noteId": pl.Utf8, "noteAuthorParticipantId": pl.Utf8, "tweetId": pl.Utf8}
        )
        for f in note_files
    ]).unique(subset=["noteId"])

    notes_full = notes_consolidated.join(notes_meta, on="noteId", how="left")
    notes_full = notes_full.with_columns(
        pl.when(pl.col("tweetId").is_not_null())
        .then(pl.concat_str([pl.lit("https://x.com/i/web/status/"), pl.col("tweetId")]))
        .otherwise(pl.lit("")).alias("tweet_url")
    )

    # =========================================================================
    # PART 3: QUADRANT ANALYSIS (TEST 1)
    # =========================================================================
    print("\n" + "=" * 70)
    print("           TEST 1: STRATEGIC NOTE QUADRANT ANALYSIS")
    print("=" * 70)

    # Filter for notes with substantial ratings from both sides (e.g. >= 5 ratings from both)
    eval_notes = notes_full.filter((pl.col("left_total") >= 3) & (pl.col("right_total") >= 10))
    print(f"Evaluated notes with ratings from both cohorts: {len(eval_notes):,} / {len(notes_full):,}")

    q1_consensus_helpful = eval_notes.filter((pl.col("right_helpful_pct") >= 65.0) & (pl.col("left_helpful_pct") >= 65.0))
    q2_polarized_conservative = eval_notes.filter((pl.col("right_helpful_pct") >= 65.0) & (pl.col("left_helpful_pct") <= 35.0))
    q3_polarized_liberal = eval_notes.filter((pl.col("right_helpful_pct") <= 35.0) & (pl.col("left_helpful_pct") >= 65.0))
    q4_consensus_not_helpful = eval_notes.filter((pl.col("right_helpful_pct") <= 35.0) & (pl.col("left_helpful_pct") <= 35.0))

    total_eval = len(eval_notes)
    print(f"\n1. Quadrant 1 [Cross-Partisan HELPFUL]:     {len(q1_consensus_helpful):,} notes ({len(q1_consensus_helpful)/total_eval*100:.1f}%)")
    print(f"2. Quadrant 2 [Right Loves, Left Hates]:       {len(q2_polarized_conservative):,} notes ({len(q2_polarized_conservative)/total_eval*100:.1f}%)")
    print(f"3. Quadrant 3 [Left Loves, Right Hates]:       {len(q3_polarized_liberal):,} notes ({len(q3_polarized_liberal)/total_eval*100:.1f}%)")
    print(f"4. Quadrant 4 [Cross-Partisan REJECTED]:       {len(q4_consensus_not_helpful):,} notes ({len(q4_consensus_not_helpful)/total_eval*100:.1f}%)")

    # The key insight: What is the total cross-partisan agreement?
    total_agreed = len(q1_consensus_helpful) + len(q4_consensus_not_helpful)
    print(f"\n--> TOTAL CROSS-PARTISAN AGREEMENT: {total_agreed:,} / {total_eval:,} ({total_agreed/total_eval*100:.1f}%)")
    print(f"    - Notice that {len(q4_consensus_not_helpful)/total_agreed*100:.1f}% of cross-partisan agreement is MUTUAL REJECTION of bad notes!")

    # Inspect Quadrant 1 (Consensus Helpful)
    print("\n" + "-" * 70)
    print("  SAMPLE NOTES: QUADRANT 1 (CROSS-PARTISAN HELPFUL CONSENSUS)")
    print("-" * 70)
    for row in q1_consensus_helpful.sort("right_total", descending=True).head(4).iter_rows(named=True):
        print(f"Note ID: {row['noteId']} | Tweet: {row['tweet_url']}")
        print(f"Votes:   Right: {row['right_helpful']}/{row['right_total']} ({row['right_helpful_pct']}%) | Left: {row['left_helpful']}/{row['left_total']} ({row['left_helpful_pct']}%)")
        print(f"Summary: {row['summary'][:160]}...")
        print("-" * 70)

    # Inspect Quadrant 2 (Polarized Conservative Notes)
    print("\n" + "-" * 70)
    print("  SAMPLE NOTES: QUADRANT 2 (POLARIZED CONSERVATIVE NOTES)")
    print("-" * 70)
    for row in q2_polarized_conservative.sort("right_total", descending=True).head(4).iter_rows(named=True):
        print(f"Note ID: {row['noteId']} | Tweet: {row['tweet_url']}")
        print(f"Votes:   Right: {row['right_helpful']}/{row['right_total']} ({row['right_helpful_pct']}%) | Left: {row['left_helpful']}/{row['left_total']} ({row['left_helpful_pct']}%)")
        print(f"Summary: {row['summary'][:160]}...")
        print("-" * 70)

    # Save quadrant dataset
    notes_full.write_csv(outputs_dir / "sg01_notes_left_right_breakdown.csv")
    print(f"\nSaved full Left vs. Right note breakdown to {outputs_dir / 'sg01_notes_left_right_breakdown.csv'}")

    # =========================================================================
    # PART 4: JACCARD SIMILARITY / SYBIL FINGERPRINTING (TEST 2)
    # =========================================================================
    print("\n" + "=" * 70)
    print("       TEST 2: INTRA-COHORT PAIRWISE JACCARD SIMILARITY")
    print("=" * 70)

    # Load edges of SG-01 to build user -> notes sets
    print("Loading edges to compute pairwise Jaccard similarities...")
    sg1_edges = pl.read_csv(outputs_dir / "subgraph_01" / "edges.csv", schema_overrides={"noteId": pl.Utf8, "raterParticipantId": pl.Utf8})

    user_note_map = {}
    for r, n in zip(sg1_edges["raterParticipantId"], sg1_edges["noteId"]):
        if r not in user_note_map:
            user_note_map[r] = set()
        user_note_map[r].add(n)

    def compute_jaccard_distribution(user_list, n_pairs=1000, seed=42):
        random.seed(seed)
        u_arr = list(user_list)
        sims = []
        for _ in range(n_pairs):
            u1, u2 = random.sample(u_arr, 2)
            s1 = user_note_map.get(u1, set())
            s2 = user_note_map.get(u2, set())
            union_len = len(s1 | s2)
            if union_len > 0:
                sims.append(len(s1 & s2) / union_len)
            else:
                sims.append(0.0)
        return np.array(sims)

    left_sims = compute_jaccard_distribution(left_users, n_pairs=1000)
    right_sims = compute_jaccard_distribution(right_users, n_pairs=1000)

    # Compare with Subgraph 14 (Known Bot/Sybil Clique)
    sg14_edges = pl.read_csv(outputs_dir / "subgraph_14" / "edges.csv", schema_overrides={"noteId": pl.Utf8, "raterParticipantId": pl.Utf8})
    sg14_map = {}
    for r, n in zip(sg14_edges["raterParticipantId"], sg14_edges["noteId"]):
        if r not in sg14_map:
            sg14_map[r] = set()
        sg14_map[r].add(n)
    
    sg14_users = list(sg14_map.keys())
    sg14_sims = []
    for _ in range(500):
        u1, u2 = random.sample(sg14_users, 2)
        s1 = sg14_map[u1]
        s2 = sg14_map[u2]
        sg14_sims.append(len(s1 & s2) / len(s1 | s2))
    sg14_sims = np.array(sg14_sims)

    print("\nPairwise Jaccard Similarity Comparison (Overlapping Notes Evaluated):")
    print(f"1. SG-01 Left Cohort  (3,871 users): Mean = {left_sims.mean():.4f} | Median = {np.median(left_sims):.4f} | 90th% = {np.percentile(left_sims, 90):.4f} | Max = {left_sims.max():.4f}")
    print(f"2. SG-01 Right Cohort (15,618 users): Mean = {right_sims.mean():.4f} | Median = {np.median(right_sims):.4f} | 90th% = {np.percentile(right_sims, 90):.4f} | Max = {right_sims.max():.4f}")
    print(f"3. Subgraph 14 (Verified Bot Clique): Mean = {sg14_sims.mean():.4f} | Median = {np.median(sg14_sims):.4f} | 90th% = {np.percentile(sg14_sims, 90):.4f} | Min = {sg14_sims.min():.4f}")

    # =========================================================================
    # PART 5: TEMPORAL CIRCADIAN DYNAMICS (TEST 3)
    # =========================================================================
    print("\n" + "=" * 70)
    print("       TEST 3: CIRCADIAN RHYTHM & TEMPORAL PATTERNS")
    print("=" * 70)
    print("Hourly Activity Distribution (UTC Hours 00 to 23):")
    total_l_ratings = left_hours.sum()
    total_r_ratings = right_hours.sum()

    print("Hour (UTC) | Left %   | Right %  | Visual")
    print("-" * 50)
    for h in range(24):
        pct_l = left_hours[h] / total_l_ratings * 100 if total_l_ratings else 0
        pct_r = right_hours[h] / total_r_ratings * 100 if total_r_ratings else 0
        bar = "#" * int(pct_l * 2)
        print(f"{h:02d}:00      | {pct_l:5.2f}%  | {pct_r:5.2f}%  | {bar}")

    # Calculate peak-to-trough ratio
    l_peak = left_hours.max() / total_l_ratings * 100
    l_trough = left_hours.min() / total_l_ratings * 100
    r_peak = right_hours.max() / total_r_ratings * 100
    r_trough = right_hours.min() / total_r_ratings * 100

    print(f"\nLeft Diurnal Wave:  Peak = {l_peak:.2f}% (Hour {np.argmax(left_hours):02d}:00) | Trough = {l_trough:.2f}% (Hour {np.argmin(left_hours):02d}:00) | Ratio = {l_peak/l_trough:.2f}x")
    print(f"Right Diurnal Wave: Peak = {r_peak:.2f}% (Hour {np.argmax(right_hours):02d}:00) | Trough = {r_trough:.2f}% (Hour {np.argmin(right_hours):02d}:00) | Ratio = {r_peak/r_trough:.2f}x")


if __name__ == "__main__":
    main()
