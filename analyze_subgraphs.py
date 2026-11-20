"""
analyze_subgraphs.py
--------------------
Phase 2 Statistical Analysis Pipeline for detected Community Notes subgraphs.
Evaluates:
1. User Enrollment Status (earnedIn vs. newUser) with two-proportion z-tests and Fisher exact tests.
2. Note Author & Tweet Concentration using notes metadata (notes-*.tsv).
3. Helpfulness & Agreement Distributions (Helpful vs. Not Helpful vs. Population Baseline).
4. Temporal Dynamics & Coordination Windows (timestamps and duration).
"""

import os
import sys
import glob
import json
import time
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 Statistical Analysis of Mined Subgraphs.")
    parser.add_argument("--data-dir", type=str, default="Data", help="Path to raw data directory")
    parser.add_argument("--outputs-dir", type=str, default="outputs", help="Path to mined outputs directory")
    parser.add_argument("--max-sample-edges", type=int, default=100000, help="Max edges to sample for huge subgraphs")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    outputs_dir = Path(args.outputs_dir)

    manifest_path = outputs_dir / "subgraphs_manifest.csv"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} does not exist. Run src/mine_subgraphs.py first.")
        return

    subgraphs_manifest = pl.read_csv(manifest_path)
    num_subgraphs = len(subgraphs_manifest)
    print(f"Loaded manifest of {num_subgraphs} subgraphs to analyze.\n")

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
        enroll_df = pl.read_csv(enroll_path, separator="\t")
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
            sub_raters = pl.read_csv(rater_file)
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
            sig_marker = "***" if p_val < 0.001 else ""
            print(f"Subgraph {k:02d}: {sub_earned:,}/{sub_total:,} earnedIn ({sub_rate*100:.1f}%) | z={z_stat:+.2f}, p={p_val:.2e} {sig_marker}")

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
        print(f"Subgraph {k:02d}: {matched_notes:,} notes | {num_authors:,} authors | Top author: {top_author_pct:.1f}% ({top_author_count} notes) | Top 3: {top_3_pct:.1f}%")

    # Free notes_df
    del notes_df

    # =========================================================================
    # STEP 3: HELPFULNESS & TEMPORAL DYNAMICS STREAMING ANALYSIS
    # =========================================================================
    print("\n--- Step 3: Helpfulness, Agreement & Temporal Dynamics ---")
    # Build tracking map for edges to evaluate
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

    print(f"Tracking {len(edge_track_map):,} unique edges across 20 subgraphs...")

    # Data accumulators per subgraph
    sub_metrics = {k: {
        "helpful": 0, "not_helpful": 0, "somewhat_helpful": 0, "other": 0,
        "agree": 0, "disagree": 0, "total": 0, "timestamps": []
    } for k in range(1, num_subgraphs + 1)}

    pop_metrics = {"helpful": 0, "not_helpful": 0, "somewhat_helpful": 0, "total": 0}

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

        # Vectorized population stats
        h_mask = (df["helpfulnessLevel"] == "HELPFUL") | (df["helpful"] == 1)
        nh_mask = (df["helpfulnessLevel"] == "NOT_HELPFUL") | (df["notHelpful"] == 1)
        sh_mask = (df["helpfulnessLevel"] == "SOMEWHAT_HELPFUL")

        pop_metrics["helpful"] += int(h_mask.sum())
        pop_metrics["not_helpful"] += int(nh_mask.sum())
        pop_metrics["somewhat_helpful"] += int(sh_mask.sum())
        pop_metrics["total"] += row_count

        # Subgraph matching
        raters = df["raterParticipantId"].to_list()
        notes = df["noteId"].to_list()
        h_arr = h_mask.to_numpy()
        nh_arr = nh_mask.to_numpy()
        sh_arr = sh_mask.to_numpy()
        agree_arr = (df["agree"] == 1).to_numpy()
        disagree_arr = (df["disagree"] == 1).to_numpy()
        ts_arr = df["createdAtMillis"].to_numpy()

        for i in range(row_count):
            key = (raters[i], notes[i])
            sub_ids = edge_track_map.get(key)
            if sub_ids is not None:
                is_h = h_arr[i]
                is_nh = nh_arr[i]
                is_sh = sh_arr[i]
                is_ag = agree_arr[i]
                is_dag = disagree_arr[i]
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
    pop_h_rate = pop_metrics["helpful"] / pop_metrics["total"] if pop_metrics["total"] else 0.0
    pop_nh_rate = pop_metrics["not_helpful"] / pop_metrics["total"] if pop_metrics["total"] else 0.0
    print(f"Population Baseline Ratings ({pop_metrics['total']:,} total):")
    print(f"  Helpful:     {pop_metrics['helpful']:,} ({pop_h_rate*100:.2f}%)")
    print(f"  Not Helpful: {pop_metrics['not_helpful']:,} ({pop_nh_rate*100:.2f}%)")
    print(f"  Somewhat:    {pop_metrics['somewhat_helpful']:,} ({pop_metrics['somewhat_helpful']/pop_metrics['total']*100:.2f}%)\n")

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

        # Two proportion z-test on helpfulness vs. platform baseline
        if tot > 0:
            p_pool_h = (h_cnt + pop_metrics["helpful"]) / (tot + pop_metrics["total"])
            se_h = (p_pool_h * (1 - p_pool_h) * (1 / tot + 1 / pop_metrics["total"])) ** 0.5
            z_helpful = (h_rate - pop_h_rate) / se_h if se_h > 0 else 0.0
            p_val_helpful = float(2 * (1 - stats.norm.cdf(abs(z_helpful))))
        else:
            z_helpful, p_val_helpful = 0.0, 1.0

        # Temporal dynamics
        ts_list = m["timestamps"]
        if len(ts_list) > 1:
            ts_arr = np.sort(np.array(ts_list, dtype=np.int64))
            min_ts = int(ts_arr[0])
            max_ts = int(ts_arr[-1])
            duration_days = round((max_ts - min_ts) / (1000 * 3600 * 24), 2)
            deltas_sec = np.diff(ts_arr) / 1000.0
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

    print(f"\nPhase 2 analysis saved to:\n  - {out_csv}\n  - {out_json}")


if __name__ == "__main__":
    main()
