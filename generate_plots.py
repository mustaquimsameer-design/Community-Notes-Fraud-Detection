"""
generate_plots.py
-----------------
Generates publication-quality figures from Phase 1 and Phase 2 fraud detection results:
1. Fig 1: Bipartite Density vs. Helpfulness Skew (Scatter plot with baselines).
2. Fig 2: Earned-In Account Proportions vs. Platform Baseline (Bar chart with z-scores).
3. Fig 3: Author Dominance & Concentration (Grouped bar chart).
4. Fig 4: Subgraph 14 Adjacency Matrix Spy Plot (Visualizing near-perfect clique).
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Set clean aesthetic style
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 300,
})


def main():
    outputs_dir = Path("outputs")
    figures_dir = outputs_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = outputs_dir / "subgraphs_manifest.csv"
    phase2_path = outputs_dir / "phase2_statistical_analysis.csv"

    if not manifest_path.exists() or not phase2_path.exists():
        print("Error: Missing manifest or phase2 analysis files in outputs/")
        return

    m_df = pl.read_csv(manifest_path)
    p_df = pl.read_csv(phase2_path)
    df = m_df.join(p_df, on="subgraph_id").to_pandas()

    print(f"Loaded {len(df)} subgraphs for plotting.")

    # =========================================================================
    # FIGURE 1: BIPARTITE DENSITY VS. HELPFULNESS SKEW
    # =========================================================================
    print("Generating Figure 1: Bipartite Density vs. Helpfulness Skew...")
    fig, ax = plt.subplots(figsize=(10, 6.5))

    pop_helpful_pct = 61.55
    pop_density_pct = 0.00695

    # Color code by behavior
    colors = []
    for _, row in df.iterrows():
        if row["helpful_pct"] > 90 and row["bipartite_density"] * 100 > 10:
            colors.append("#d62728")  # Red: High density boosting rings
        elif row["helpful_pct"] < 50:
            colors.append("#1f77b4")  # Blue: Note-killing suppression ring
        elif row["helpful_pct"] > 75:
            colors.append("#ff7f0e")  # Orange: Moderate boosting
        else:
            colors.append("#7f7f7f")  # Gray: General active core

    scatter = ax.scatter(
        df["bipartite_density"] * 100,
        df["helpful_pct"],
        s=np.log10(df["total_subgraph_edges"]) * 45,
        c=colors,
        alpha=0.85,
        edgecolors="black",
        linewidths=1.2,
        zorder=4,
    )

    # Reference lines
    ax.axhline(pop_helpful_pct, color="gray", linestyle="--", linewidth=1.5, label=f"Platform Helpful Baseline ({pop_helpful_pct}%)", zorder=2)
    ax.axvline(pop_density_pct, color="navy", linestyle=":", linewidth=1.5, label=f"Platform Graph Density ({pop_density_pct}%)", zorder=2)

    # Annotations for key subgraphs
    key_annotations = {
        14: "SG-14: 89.8% Density\n99.8% Helpful (33 users)",
        12: "SG-12: 62.1% Density\n98.9% Helpful",
        15: "SG-15: 57.6% Density\n97.3% Helpful",
        17: "SG-17: 25.1% Density\n94.4% Helpful (33% 1 Author)",
        1: "SG-01: Note-Killing Ring\n54.7% Not Helpful (z=-114.5)",
        2: "SG-02: 8.4% Density\n90.6% Helpful (95.4% Earned-In)",
    }

    for sg_id, text in key_annotations.items():
        row = df[df["subgraph_id"] == sg_id].iloc[0]
        x = row["bipartite_density"] * 100
        y = row["helpful_pct"]
        offset_y = 4 if y < 90 else -5
        offset_x = 1.3 if x < 20 else 0.7
        ax.annotate(
            text,
            (x, y),
            xytext=(x * offset_x, y + offset_y),
            textcoords="data",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3, edgecolor="gray"),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="black", lw=1),
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.003, 120)
    ax.set_ylim(35, 105)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g%%"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d%%"))

    ax.set_xlabel("Bipartite Graph Density (%) [Log Scale]", fontweight="bold")
    ax.set_ylabel("Percentage of Ratings Rated 'Helpful' (%)", fontweight="bold")
    ax.set_title("Detection of Coordinated Subgraphs in Community Notes\nDensity vs. Helpfulness Polarization", fontweight="bold", pad=12)
    ax.legend(loc="lower left", framealpha=0.9)

    fig1_path = figures_dir / "fig1_density_vs_helpfulness.png"
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved {fig1_path}")

    # =========================================================================
    # FIGURE 2: EARNED-IN PROPORTIONS ACROSS SUBGRAPHS
    # =========================================================================
    print("Generating Figure 2: Earned-In Proportions...")
    fig, ax = plt.subplots(figsize=(11, 5.5))

    sub_ids = df["subgraph_id"].values
    earned_pcts = df["earned_in_pct"].values
    pop_earned_pct = 39.03

    bars = ax.bar(sub_ids, earned_pcts, color="#2ca02c", alpha=0.8, edgecolor="black", width=0.65, zorder=3)
    ax.axhline(pop_earned_pct, color="red", linestyle="--", linewidth=1.8, label=f"Platform Population Baseline ({pop_earned_pct}%)", zorder=4)

    # Highlight top earned-in
    for bar, val in zip(bars, earned_pcts):
        if val > 90:
            bar.set_color("#1b7837")

    ax.set_xticks(sub_ids)
    ax.set_xlabel("Mined Subgraph ID (k = 1 ... 20)", fontweight="bold")
    ax.set_ylabel("Proportion of 'Earned-In' Raters (%)", fontweight="bold")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d%%"))
    ax.set_title("Privilege Concentration in Suspicious Subgraphs\nProportion of 'Earned-In' Scoring Accounts vs. Platform Baseline", fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.9)

    # Annotate that all are statistically significant
    ax.text(
        0.02, 0.92,
        "All subgraphs significantly exceed platform baseline\n(Two-proportion z-tests: p < 10^-8 to p = 0.00)",
        transform=ax.transAxes,
        fontsize=9.5,
        fontstyle="italic",
        bbox=dict(boxstyle="round,pad=0.4", fc="whitesmoke", edgecolor="gray"),
    )

    fig2_path = figures_dir / "fig2_earned_in_proportions.png"
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved {fig2_path}")

    # =========================================================================
    # FIGURE 3: AUTHOR CONCENTRATION
    # =========================================================================
    print("Generating Figure 3: Author Concentration...")
    fig, ax = plt.subplots(figsize=(11, 5.5))

    x = np.arange(len(sub_ids))
    width = 0.35

    ax.bar(x - width/2, df["top_1_author_pct"], width, label="Top-1 Author Share (%)", color="#9467bd", edgecolor="black", zorder=3)
    ax.bar(x + width/2, df["top_3_authors_pct"], width, label="Top-3 Authors Share (%)", color="#c5b0d5", edgecolor="black", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(sub_ids)
    ax.set_xlabel("Mined Subgraph ID (k = 1 ... 20)", fontweight="bold")
    ax.set_ylabel("Percentage of Subgraph Notes Authored (%)", fontweight="bold")
    ax.set_ylim(0, 52)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d%%"))
    ax.set_title("Author Dominance & Concentration in Suspicious Subgraphs\nEvidence of Author-Targeted Note Boosting", fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.9)

    # Annotate Subgraphs 17 and 15
    ax.annotate(
        "SG-17: 33.1% Top-1 Author\n(44.5% Top-3)",
        (16, 44.5),
        xytext=(13.5, 47),
        fontsize=9,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
    )
    ax.annotate(
        "SG-15: 29.0% Top-1 Author",
        (14, 33.0),
        xytext=(11.5, 36),
        fontsize=9,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
    )

    fig3_path = figures_dir / "fig3_author_concentration.png"
    plt.tight_layout()
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved {fig3_path}")

    # =========================================================================
    # FIGURE 4: SUBGRAPH 14 ADJACENCY MATRIX SPY PLOT
    # =========================================================================
    print("Generating Figure 4: Subgraph 14 Spy Plot...")
    sub14_edges_file = outputs_dir / "subgraph_14" / "edges.csv"
    if sub14_edges_file.exists():
        edges14 = pl.read_csv(sub14_edges_file)
        unique_raters = {r: i for i, r in enumerate(edges14["raterParticipantId"].unique())}
        unique_notes = {n: j for j, n in enumerate(edges14["noteId"].unique())}

        grid = np.zeros((len(unique_raters), len(unique_notes)), dtype=np.uint8)
        for r, n in zip(edges14["raterParticipantId"], edges14["noteId"]):
            grid[unique_raters[r], unique_notes[n]] = 1

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.imshow(grid, aspect="auto", cmap="Blues", interpolation="none")
        ax.set_title(f"Subgraph 14 Adjacency Matrix: 33 Raters x 520 Notes\nBipartite Density: 89.8% ({len(edges14):,} Edges | 99.8% Helpful)", fontweight="bold", pad=12)
        ax.set_xlabel("Note Index (0 to 519)", fontweight="bold")
        ax.set_ylabel("Rater Index (0 to 32)", fontweight="bold")

        fig4_path = figures_dir / "fig4_subgraph14_adjacency_matrix.png"
        plt.tight_layout()
        plt.savefig(fig4_path, dpi=300)
        plt.close()
        print(f"Saved {fig4_path}")

    print("\nAll figures generated successfully in outputs/figures/")


if __name__ == "__main__":
    main()
