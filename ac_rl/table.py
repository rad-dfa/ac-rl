import argparse
import os
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "storage" / "exp_test_results.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "storage" / "tables"

ENV_NAME = "TokenEnv"

POLICIES = [
    ("rad", 5),
    ("rad", 10),
    ("rad", 20),
    ("no_rad", 5),
    ("no_rad", 10),
    ("no_rad", 20),
]

EMBEDDING_LABELS = {
    "rad": "RAD Embd",
    "no_rad": "No RAD Embd",
}

SAMPLERS = [
    ("R", False, ["Reach"]),
    ("RA", False, ["ReachAvoid"]),
    ("RAD", False, ["RAD"]),
    ("R", True, ["Reach", "(OOD)"]),
    ("RA", True, ["ReachAvoid", "(OOD)"]),
    ("RAD", True, ["RAD", "(OOD)"]),
]


def parse_val_err(cell):
    if pd.isna(cell):
        return None, None
    try:
        val, err = str(cell).split(" +/- ")
        return float(val), float(err)
    except (ValueError, AttributeError):
        return None, None


def load_results(csv_path):
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df[["SuccVal", "SuccErr"]] = df["Success Probability"].apply(
        lambda x: pd.Series(parse_val_err(x))
    )
    return df


def lookup_cell(df, rad, n_events, sampler, ood):
    row = df[
        (df["RAD"] == rad)
        & (df["N_Events"] == n_events)
        & (df["Sampler"] == sampler)
        & (df["OOD"] == ood)
    ]
    if row.empty:
        return None, None
    entry = row.iloc[0]
    return entry["SuccVal"], entry["SuccErr"]


ARRAY_STRETCH = 1.15


def format_stacked_cell(*lines):
    body = " \\\\ ".join(lines)
    return f"\\begin{{tabular}}[c]{{@{{}}c@{{}}}}{body}\\end{{tabular}}"


def format_policy_cell(rad, n_events):
    emb = EMBEDDING_LABELS[rad]
    return format_stacked_cell(f"{emb};", f"{n_events} Events")


def format_header_cell(lines):
    if len(lines) == 1:
        return f"\\textbf{{{lines[0]}}}"
    return format_stacked_cell(
        f"\\textbf{{{lines[0]}}}",
        f"\\textbf{{{lines[1]}}}",
    )


def format_prob(val, err):
    if val is None:
        return "---"
    return f"{val:.3f} $\\pm$ {err:.3f}"


def make_table(df):
    num_samplers = len(SAMPLERS)
    num_cols = 1 + num_samplers

    tex = []
    tex.append("% Requires: \\usepackage{array, graphicx}")
    tex.append("\\begin{table*}[t]")
    tex.append("\\centering")
    tex.append("\\footnotesize")
    tex.append(f"\\renewcommand{{\\arraystretch}}{{{ARRAY_STRETCH}}}")
    tex.append("\\setlength{\\tabcolsep}{3pt}")
    tex.append("\\resizebox{\\linewidth}{!}{%")
    tex.append(
        "\\begin{tabular}{|c||"
        + "c|" * (num_samplers // 2)
        + "|"
        + "c|" * (num_samplers // 2)
        + "}"
    )
    tex.append("\\hline")
    tex.append(
        f"\\multicolumn{{{num_cols}}}{{|c|}}{{\\textbf{{Success Probability}}}} \\\\"
    )
    tex.append("\\hline")
    sampler_headers = " & ".join(format_header_cell(lines) for _, _, lines in SAMPLERS)
    tex.append(f"\\textbf{{Policy}} & {sampler_headers} \\\\")
    tex.append("\\hline")

    for rad, n_events in POLICIES:
        row = [format_policy_cell(rad, n_events)]
        for sampler, ood, _ in SAMPLERS:
            val, err = lookup_cell(df, rad, n_events, sampler, ood)
            row.append(format_prob(val, err))
        tex.append(" & ".join(row) + " \\\\")
        tex.append("\\hline")
    tex.append("\\end{tabular}%")
    tex.append("}")
    tex.append(
        "\\caption{Success probability on \\texttt{TokenEnv}. "
        "Results are averaged over 5 seeds, each run for 1{,}000 episodes.}"
    )
    tex.append("\\label{table:tokenenv_probs}")
    tex.append("\\end{table*}")

    return "\n".join(tex)


def main():
    parser = argparse.ArgumentParser(description="Generate TokenEnv test results LaTeX table")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(DEFAULT_CSV),
        help="Path to test results CSV (default: ac_rl/storage/exp_test_results.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for output table (default: ac_rl/storage/tables)",
    )
    args = parser.parse_args()

    df = load_results(args.csv)
    table_tex = make_table(df)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{ENV_NAME}_success_probability.tex")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(table_tex)

    print(table_tex)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()

