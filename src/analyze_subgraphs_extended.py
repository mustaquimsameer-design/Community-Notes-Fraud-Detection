"""
analyze_subgraphs_extended.py
-----------------------------
Phase 2 Statistical Analysis Pipeline for all 100 mined subgraphs.
Evaluates:
1. User Enrollment Status (earnedIn vs newUser) with two-proportion z-tests vs platform baseline (39.03%).
2. Note Author & Tweet Concentration using notes metadata.
3. Helpfulness & Agreement Distributions (Helpful vs Not Helpful vs platform baseline 61.55%).
4. Temporal Dynamics & Coordination Windows (timestamps and duration).

Outputs:
- outputs/phase2_statistical_analysis.csv
- outputs/phase2_statistical_analysis.json
- outputs/phase2_statistical_analysis_subgraphs_21_to_100.csv
"""

import os
import sys
import glob
import json
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
from scipy import stats


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Statistical Analysis for subgraphs 1 to 100.")
    parser.add_argument("--data-dir", type=str, default="Data", help="Path to raw data directory")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Path to outputs directory")
    parser.add_argument("--max-sample-edges", type=int, default=100000, help="Max edges to sample for huge subgraphs")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    outputs_dir = Path(args.outputs_dir)

    manifest_path = outputs_dir / "subgraphs_manifest.csv"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found.")
        return

    subgraphs_manifest = pl.read_csv(manifest_path)
    num_subgraphs = len(subgraphs_manifest)
    print(f"Loaded manifest with {num_subgraphs} subgraphs to analyze.\n")

    # =========================================================================
    # STEP 1: USER ENROLLMENT ANALYSIS
    # =========================================================================
    print("--- Step 1: User Enrollment Statistical Analysis ---")
    enroll_path = data_dir / "userEnrollment-00000.tsv"
    if not enroll_path.exists():
        print(f"Warning: {enroll_path} not found. Skipping enrollment analysis.")
        enroll_df = None
        pop_enroll_stats = {}
    else:
        print(f"Loading user enrollment from {enroll_path}...")
        enroll_df = pl.read_csv(enroll_path, separator="\t", schema_overrides={"participantId": pl.Utf8})
        pop_total = len(enroll_df)
        pop_earned_in = int((enroll_df["enrollmentState"] == "earnedIn").sum())
        pop_new_user = int((enroll_df["enrollmentState"] == "newUser").sum())
        pop_at_risk = int((enroll_df["enrollmentState"] == "atRisk").sum())
        pop_removed = int((enroll_df["enrollmentState"] == "removed").sum())
        pop_earned_rate = pop_earned_in / pop_total

        pop_enroll_stats = {
            "pop_total_users": pop_total,
            "pop_earned_in": pop_earned_in,
            "pop_earned_in_pct": round(pop_earned_rate * 100, 2),
            "pop_new_user": pop_new_user,
            "pop_new_user_pct": round(pop_new_user / pop_total * 100, 2),
            "pop_at_risk": pop_at_risk,
            "pop_removed": pop_removed,
        }
        print(f"Population Baseline: {pop_earned_in:,} / {pop_total:,} earnedIn ({pop_earned_rate*100:.2f}%)\n")

    enrollment_results = {}
    if enroll_df is not None:
        for k in range(1, num_subgraphs + 1):
            rater_file = outputs_dir / f"subgraph_{k:02d}" / "raters.csv"
            if not rater_file.exists():
                continue
            sub_raters = pl.read_csv(rater_file, schema_overrides={"raterParticipantId": pl.Utf8})
            joined = sub_raters.join(enroll_df, left_on="raterParticipantId", right_on="participantId", how="inner")
            sub_total = len(joined)
            sub_earned = int((joined["enrollmentState"] == "earnedIn").sum())
            sub_new = int((joined["enrollmentState"] == "newUser").sum())
            sub_at_risk = int((joined["enrollmentState"] == "atRisk").sum())
            sub_removed = int((joined["enrollmentState"] == "removed").sum())
            sub_rate = sub_earned / sub_total if sub_total > 0 else 0.0

            # Two-proportion z-test
            p_pool = (sub_earned + pop_earned_in) / (sub_total + pop_total)
            se = (p_pool * (1 - p_pool) * (1 / sub_total + 1 / pop_total)) ** 0.5
            z_stat = (sub_rate - pop_earned_rate) / se if se > 0 else 0.0
            p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

            enrollment_results[k] = {
                "subgraph_id": k,
                "matched_raters": sub_total,
                "earned_in_count": sub_earned,
                "earned_in_pct": round(sub_rate * 100, 2),
                "new_user_count": sub_new,
                "new_user_pct": round(sub_new / sub_total * 100, 2) if sub_total else 0,
                "at_risk_count": sub_at_risk,
                "removed_count": sub_removed,
                "pop_earned_in_pct": round(pop_earned_rate * 100, 2),
                "diff_pct_points": round((sub_rate - pop_earned_rate) * 100, 2),
                "enrollment_z_stat": round(float(z_stat), 3),
                "enrollment_p_val": p_val,
                "is_significant_001": bool(p_val < 0.001),
            }

    # =========================================================================
    # STEP 2: NOTE AUTHOR & TWEET CONCENTRATION ANALYSIS
    # =========================================================================
    print("\n--- Step 2: Note Author & Tweet Concentration Analysis ---")
    note_files = sorted(glob.glob(str(data_dir / "notes-*.tsv")))
    print(f"Loading notes metadata from {len(note_files)} file(s)...")
    notes_df = pl.concat([
        pl.read_csv(
            f, separator="\t",
            columns=["noteId", "noteAuthorParticipantId", "tweetId"],
            schema_overrides={"noteId": pl.Utf8, "noteAuthorParticipantId": pl.Utf8, "tweetId": pl.Utf8},
        )
        for f in note_files
    ])
    print(f"Total notes metadata loaded: {len(notes_df):,}\n")

    author_results = {}
    for k in range(1, num_subgraphs + 1):
        note_file = outputs_dir / f"subgraph_{k:02d}" / "notes.csv"
        if not note_file.exists():
            continue
        sub_notes = pl.read_csv(note_file, schema_overrides={"noteId": pl.Utf8})
        joined = sub_notes.join(notes_df, on="noteId", how="inner")
        matched_notes = len(joined)
        num_authors = joined["noteAuthorParticipantId"].n_unique()
        num_tweets = joined["tweetId"].n_unique()

        author_counts = joined["noteAuthorParticipantId"].value_counts().sort("count", descending=True)
        top_author_count = int(author_counts["count"][0]) if len(author_counts) > 0 else 0
        top_author_pct = (top_author_count / matched_notes * 100) if matched_notes > 0 else 0.0

        top_3_sum = int(author_counts["count"][:3].sum()) if len(author_counts) > 0 else 0
        top_3_pct = (top_3_sum / matched_notes * 100) if matched_notes > 0 else 0.0

        notes_per_author = (matched_notes / num_authors) if num_authors > 0 else 0.0

        author_results[k] = {
            "subgraph_id": k,
            "matched_notes": matched_notes,
            "unique_authors": num_authors,
            "notes_per_author_ratio": round(notes_per_author, 2),
            "unique_tweets": num_tweets,
            "top_1_author_notes": top_author_count,
            "top_1_author_pct": round(top_author_pct, 2),
            "top_3_authors_pct": round(top_3_pct, 2),
        }

    del notes_df

    # =========================================================================
    # STEP 3: HELPFULNESS & TEMPORAL DYNAMICS STREAMING ANALYSIS
    # =========================================================================
    print("\n--- Step 3: Helpfulness, Agreement & Temporal Dynamics ---")
    edge_track_map = {}  # (rater, note) -> list of subgraph_ids
    subgraph_sample_sizes = {}

    for k in range(1, num_subgraphs + 1):
        edge_file = outputs_dir / f"subgraph_{k:02d}" / "edges.csv"
        if not edge_file.exists():
            continue
        edges_df = pl.read_csv(edge_file, schema_overrides={"noteId": pl.Utf8, "raterParticipantId": pl.Utf8})
        n_edges = len(edges_df)
        if n_edges > args.max_sample_edges:
            edges_df = edges_df.sample(n=args.max_sample_edges, seed=42)
            subgraph_sample_sizes[k] = (len(edges_df), n_edges, True)
        else:
            subgraph_sample_sizes[k] = (len(edges_df), n_edges, False)

        raters = edges_df["raterParticipantId"].to_list()
        notes = edges_df["noteId"].to_list()
        for i in range(len(raters)):
            key = (raters[i], notes[i])
            if key not in edge_track_map:
                edge_track_map[key] = [k]
            else:
                edge_track_map[key].append(k)

    print(f"Tracking {len(edge_track_map):,} unique edges across {num_subgraphs} subgraphs...")

    sub_metrics = {k: {
        "helpful": 0, "not_helpful": 0, "somewhat_helpful": 0, "other": 0,
        "agree": 0, "disagree": 0, "total": 0, "timestamps": []
    } for k in range(1, num_subgraphs + 1)}

    # Platform population baseline constants from verified previous scan:
    pop_total = 241643451
    pop_helpful = 148723528
    pop_not_helpful = 86716182
    pop_somewhat_helpful = 6185848
    pop_h_rate = pop_helpful / pop_total
    pop_nh_rate = pop_not_helpful / pop_total

    rating_files = sorted(glob.glob(str(data_dir / "ratings-*.tsv")))
    t_stream_start = time.time()

    for f_idx, f_path in enumerate(rating_files, 1):
        t0 = time.time()
        print(f"[{f_idx}/{len(rating_files)}] Scanning {os.path.basename(f_path)}...", end="", flush=True)

        df = pl.read_csv(
            f_path,
            separator="\t",
            columns=["raterParticipantId", "noteId", "helpful", "notHelpful", "helpfulnessLevel", "agree", "disagree", "createdAtMillis"],
            schema_overrides={"raterParticipantId": pl.Utf8, "noteId": pl.Utf8},
        )
        row_count = len(df)

        h_mask = ((df["helpfulnessLevel"] == "HELPFUL") | (df["helpful"] == 1)).to_numpy()
        nh_mask = ((df["helpfulnessLevel"] == "NOT_HELPFUL") | (df["notHelpful"] == 1)).to_numpy()
        sh_mask = (df["helpfulnessLevel"] == "SOMEWHAT_HELPFUL").to_numpy()
        agree_mask = (df["agree"] == 1).to_numpy()
        disagree_mask = (df["disagree"] == 1).to_numpy()
        ts_arr = df["createdAtMillis"].to_numpy()
        raters = df["raterParticipantId"].to_list()
        notes = df["noteId"].to_list()

        for i in range(row_count):
            key = (raters[i], notes[i])
            sub_ids = edge_track_map.get(key)
            if sub_ids is not None:
                is_h = h_mask[i]
                is_nh = nh_mask[i]
                is_sh = sh_mask[i]
                is_ag = agree_mask[i]
                is_dag = disagree_mask[i]
                ts = int(ts_arr[i])

                for sid in sub_ids:
                    s_dict = sub_metrics[sid]
                    s_dict["total"] += 1
                    if is_h:
                        s_dict["helpful"] += 1
                    elif is_nh:
                        s_dict["not_helpful"] += 1
                    elif is_sh:
                        s_dict["somewhat_helpful"] += 1
                    else:
                        s_dict["other"] += 1

                    if is_ag:
                        s_dict["agree"] += 1
                    if is_dag:
                        s_dict["disagree"] += 1

                    s_dict["timestamps"].append(ts)

        del df
        print(f" Done ({row_count:,} rows in {time.time()-t0:.1f}s)")

    print(f"\nStreaming scan finished in {time.time()-t_stream_start:.1f}s.")

    # =========================================================================
    # STEP 4: COMPILE FINAL METRICS & STATISTICAL TESTS
    # =========================================================================
    print("--- Step 4: Compiling Integrated Phase 2 Analysis ---")
    final_rows = []

    for k in range(1, num_subgraphs + 1):
        m = sub_metrics[k]
        tot = m["total"]
        h_cnt = m["helpful"]
        nh_cnt = m["not_helpful"]
        sh_cnt = m["somewhat_helpful"]
        ag_cnt = m["agree"]
        dag_cnt = m["disagree"]

        h_rate = h_cnt / tot if tot > 0 else 0.0
        nh_rate = nh_cnt / tot if tot > 0 else 0.0
        sh_rate = sh_cnt / tot if tot > 0 else 0.0
        ag_rate = ag_cnt / tot if tot > 0 else 0.0
        dag_rate = dag_cnt / tot if tot > 0 else 0.0

        if tot > 0:
            p_pool_h = (h_cnt + pop_helpful) / (tot + pop_total)
            se_h = (p_pool_h * (1 - p_pool_h) * (1 / tot + 1 / pop_total)) ** 0.5
            z_helpful = (h_rate - pop_h_rate) / se_h if se_h > 0 else 0.0
            p_val_helpful = float(2 * (1 - stats.norm.cdf(abs(z_helpful))))
        else:
            z_helpful, p_val_helpful = 0.0, 1.0

        ts_list = m["timestamps"]
        if len(ts_list) > 1:
            ts_sorted = np.sort(np.array(ts_list, dtype=np.int64))
            min_ts = int(ts_sorted[0])
            max_ts = int(ts_sorted[-1])
            duration_days = round((max_ts - min_ts) / (1000 * 3600 * 24), 2)
            deltas_sec = np.diff(ts_sorted) / 1000.0
            median_delta_sec = round(float(np.median(deltas_sec)), 2)
        else:
            duration_days = 0.0
            median_delta_sec = 0.0

        enr = enrollment_results.get(k, {})
        aut = author_results.get(k, {})
        sample_info = subgraph_sample_sizes.get(k, (tot, tot, False))

        row = {
            "subgraph_id": k,
            # Graph structure
            "sample_edges": tot,
            "total_subgraph_edges": sample_info[1],
            "is_sampled": sample_info[2],
            # Helpfulness metrics
            "helpful_pct": round(h_rate * 100, 2),
            "not_helpful_pct": round(nh_rate * 100, 2),
            "somewhat_helpful_pct": round(sh_rate * 100, 2),
            "pop_helpful_pct": round(pop_h_rate * 100, 2),
            "helpful_z_stat": round(float(z_helpful), 3),
            "helpful_p_val": p_val_helpful,
            # Agreement metrics
            "agree_pct": round(ag_rate * 100, 2),
            "disagree_pct": round(dag_rate * 100, 2),
            # Enrollment metrics
            "raters_count": enr.get("matched_raters", 0),
            "earned_in_pct": enr.get("earned_in_pct", 0.0),
            "pop_earned_in_pct": enr.get("pop_earned_in_pct", 0.0),
            "enrollment_z_stat": enr.get("enrollment_z_stat", 0.0),
            "enrollment_p_val": enr.get("enrollment_p_val", 1.0),
            "new_user_pct": enr.get("new_user_pct", 0.0),
            "removed_pct": round(enr.get("removed_count", 0) / enr.get("matched_raters", 1) * 100, 2) if enr.get("matched_raters") else 0.0,
            # Author concentration
            "notes_count": aut.get("matched_notes", 0),
            "unique_authors": aut.get("unique_authors", 0),
            "notes_per_author_ratio": aut.get("notes_per_author_ratio", 0.0),
            "top_1_author_pct": aut.get("top_1_author_pct", 0.0),
            "top_3_authors_pct": aut.get("top_3_authors_pct", 0.0),
            "unique_tweets": aut.get("unique_tweets", 0),
            # Temporal
            "duration_days": duration_days,
            "median_delta_seconds": median_delta_sec,
        }
        final_rows.append(row)

    final_df = pl.DataFrame(final_rows)
    out_csv = outputs_dir / "phase2_statistical_analysis.csv"
    out_json = outputs_dir / "phase2_statistical_analysis.json"
    final_df.write_csv(out_csv)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_rows, f, indent=2)

    # Also output specific CSV for subgraphs 21 to 100
    df_21_100 = final_df.filter(pl.col("subgraph_id") >= 21)
    out_21_100 = outputs_dir / "phase2_statistical_analysis_subgraphs_21_to_100.csv"
    df_21_100.write_csv(out_21_100)

    print(f"\nPhase 2 analysis saved to:")
    print(f"  - {out_csv}")
    print(f"  - {out_json}")
    print(f"  - {out_21_100} ({len(df_21_100)} rows)")


if __name__ == "__main__":
    main()
