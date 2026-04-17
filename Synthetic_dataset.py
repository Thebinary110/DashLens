import pandas as pd
import sys
from collections import Counter

# ──────────────────────────────────────────────
# CONFIG — edit these paths before running
# ──────────────────────────────────────────────
INTERN_CSV   = "internvideo_results.csv"   # InternVideo2.5 output  (tiebreaker)
CSV_2        = "llava_results.csv"
CSV_3        = "qwen+yolo_results.csv"
OUTPUT_CSV   = "sub_res.csv"
# ──────────────────────────────────────────────

def load(path):
    df = pd.read_csv(path)
    assert df.shape == (661, 25), (
        f"{path} has shape {df.shape}, expected (661, 25)"
    )
    return df

def majority_vote(intern_df, df2, df3):
    result = intern_df.copy()

    for row in range(661):
        for col in intern_df.columns:
            v1 = intern_df.at[row, col]
            v2 = df2.at[row, col]
            v3 = df3.at[row, col]

            counts = Counter([v1, v2, v3])
            best_val, best_count = counts.most_common(1)[0]

            if best_count >= 2:
                # At least two CSVs agree → use that value
                result.at[row, col] = best_val
            else:
                # All three differ → fall back to InternVideo2.5
                result.at[row, col] = v1

    return result

def main():
    print("Loading CSVs...")
    intern_df = load(INTERN_CSV)
    df2       = load(CSV_2)
    df3       = load(CSV_3)

    # Align indices just in case
    df2.index = intern_df.index
    df3.index = intern_df.index

    print("Running majority vote...")
    result = majority_vote(intern_df, df2, df3)

    result.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved → {OUTPUT_CSV}  (shape: {result.shape})")

    # Quick stats
    total_cells = 661 * 25
    agreed = 0
    tiebroken = 0
    intern_df2 = load(INTERN_CSV)   # reload clean copy for comparison
    df2b = load(CSV_2)
    df3b = load(CSV_3)
    for row in range(661):
        for col in intern_df2.columns:
            v1, v2, v3 = intern_df2.at[row, col], df2b.at[row, col], df3b.at[row, col]
            c = Counter([v1, v2, v3])
            if c.most_common(1)[0][1] >= 2:
                agreed += 1
            else:
                tiebroken += 1
    print(f"\nStats:")
    print(f"  Majority agreement : {agreed:>6} / {total_cells}  ({100*agreed/total_cells:.1f}%)")
    print(f"  Tiebreak (InternV) : {tiebroken:>6} / {total_cells}  ({100*tiebroken/total_cells:.1f}%)")

if __name__ == "__main__":
    main()