"""
inspect_sample_notes.py
-----------------------
Quick script to display top targeted notes and summaries from key subgraphs.
"""

import polars as pl

notes_df = pl.read_csv("outputs/manual_inspection_notes.csv")

for sg_id in [14, 17, 15, 2]:
    print("=" * 64)
    print(f"                 SUBGRAPH {sg_id:02d} TOP TARGETED NOTES")
    print("=" * 64)
    sub = notes_df.filter(pl.col("subgraph_id") == sg_id).head(3)
    for row in sub.iter_rows(named=True):
        print(f"Note ID: {row['noteId']}")
        print(f"Writer:  {row['writer_user_id'][:16]}...")
        print(f"Status:  {row['status']}")
        print(f"Date:    {row['timestamp_utc']}")
        print(f"Tweet:   {row['tweet_url']}")
        print(f"Votes:   {row['subgraph_helpful_count']}/{row['subgraph_ratings_count']} Helpful ({row['pct_of_ratings_helpful']}%) | {row['pct_of_subgraph_raters_helpful']}% of Subgraph")
        summary_clean = row['note_summary'].replace('\n', ' ').strip()
        print(f"Summary: {summary_clean[:140]}...")
        print("-" * 64)
