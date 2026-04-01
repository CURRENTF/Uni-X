import argparse
import re
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "analysis_data" / "conditional_entropy.csv"
DEFAULT_OUTPUT = ROOT / "pics" / "conditional_entropy.pdf"
LOG_PATTERN = re.compile(r"(.+), n=(\d+) entropy = ([\d.]+) bits")
SOURCE_ORDER = ["English", "German", "Chinese", "Image"]
PALETTE = {
    "English": "cornflowerblue",
    "German": "royalblue",
    "Chinese": "skyblue",
    "Image": "crimson",
}


def load_entropy_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            match = LOG_PATTERN.match(line.strip())
            if not match:
                continue
            source, n, entropy = match.groups()
            rows.append(
                {
                    "source": source.strip(),
                    "n": int(n),
                    "entropy": float(entropy),
                }
            )
        df = pd.DataFrame(rows)

    required_columns = {"source", "n", "entropy"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    return df


def plot_conditional_entropy(df: pd.DataFrame, output_path: Path, max_n: int) -> None:
    df = df[["source", "n", "entropy"]].copy()
    df = df[df["n"] <= max_n].sort_values(["source", "n"])
    if df.empty:
        raise ValueError(f"No rows remain after filtering with --max-n {max_n}.")

    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
        }
    )

    plt.figure(figsize=(4, 3))
    hue_order = [source for source in SOURCE_ORDER if source in set(df["source"])]
    sns.lineplot(
        data=df,
        x="n",
        y="entropy",
        hue="source",
        hue_order=hue_order,
        marker="o",
        palette=PALETTE,
    )

    plt.xlabel("N-gram (n)")
    plt.ylabel("Conditional Entropy (bits)")
    plt.legend(title=None)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the conditional-entropy curve used in Figure 2.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="CSV file or raw log file containing entropy results.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output figure path.")
    parser.add_argument("--max-n", type=int, default=4, help="Maximum n value to include in the plot.")
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_entropy_table(Path(args.input))
    plot_conditional_entropy(df, Path(args.output), max_n=args.max_n)


if __name__ == "__main__":
    main()
