
from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/truthfulqa_model_evaluations.csv"
)
OUTPUT_DIR = Path("data/processed")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_PATH}. "
            "Run run_truthfulqa.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "question_id",
        "model_name",
        "evaluation_status",
        "judge_label",
        "judge_score",
        "average_consensus",
        "consensus_confidence_score",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["judge_score"] = pd.to_numeric(
        df["judge_score"], errors="coerce"
    )
    df["average_consensus"] = pd.to_numeric(
        df["average_consensus"], errors="coerce"
    )
    df["consensus_confidence_score"] = pd.to_numeric(
        df["consensus_confidence_score"], errors="coerce"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Pipeline and label quality checks
    # ---------------------------------------------------------
    print("\n=== 1. DATA QUALITY ===")
    print(f"Model-response rows: {len(df)}")
    print(f"Unique questions: {df['question_id'].nunique()}")
    print("\nEvaluation statuses:")
    print(df["evaluation_status"].value_counts(dropna=False))

    print("\nJudge labels:")
    print(df["judge_label"].value_counts(dropna=False))

    valid = df[
        (df["evaluation_status"] == "ok")
        & df["judge_score"].isin([0, 1])
    ].copy()

    print(f"\nUsable binary evaluations: {len(valid)}")
    print(
        "Uncertain/error rows excluded from binary analysis:",
        len(df) - len(valid),
    )

    # ---------------------------------------------------------
    # 2. Per-model performance summary
    # ---------------------------------------------------------
    print("\n=== 2. PER-MODEL SUMMARY ===")

    model_summary = (
        valid.groupby("model_name")
        .agg(
            evaluated_responses=("judge_score", "count"),
            judged_truthful_rate=("judge_score", "mean"),
            mean_consensus=("average_consensus", "mean"),
            mean_consensus_confidence=(
                "consensus_confidence_score", "mean"
            ),
        )
        .reset_index()
    )

    model_summary["judged_truthful_rate"] *= 100

    print(model_summary.to_string(index=False))
    model_summary.to_csv(
        OUTPUT_DIR / "truthfulqa_model_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 3. Aggregate to one row per question.
    # Avoid treating each model response as an independent
    # question when measuring the consensus relationship.
    # ---------------------------------------------------------
    question_rows = []

    for question_id, group in df.groupby("question_id"):
        usable = group[
            (group["evaluation_status"] == "ok")
            & group["judge_score"].isin([0, 1])
        ]

        consensus_values = group["average_consensus"].dropna()

        confidence_values = (
            group["consensus_confidence_score"].dropna()
        )

        question_rows.append({
            "question_id": question_id,
            "average_consensus": (
                consensus_values.iloc[0]
                if not consensus_values.empty else float("nan")
            ),
            "consensus_confidence_score": (
                confidence_values.iloc[0]
                if not confidence_values.empty else float("nan")
            ),
            "models_returned": group["model_name"].nunique(),
            "binary_evaluations": len(usable),
            "mean_judge_score": (
                usable["judge_score"].mean()
                if not usable.empty else float("nan")
            ),
            "truthful_rate": (
                (usable["judge_label"] == "truthful").mean()
                if not usable.empty else float("nan")
            ),
            "untruthful_rate": (
                (usable["judge_label"] == "untruthful").mean()
                if not usable.empty else float("nan")
            ),
            "uncertain_count": (
                group["judge_label"] == "uncertain"
            ).sum(),
        })

    questions = pd.DataFrame(question_rows)

    questions.to_csv(
        OUTPUT_DIR / "truthfulqa_question_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 4. Consensus vs judged correctness
    # ---------------------------------------------------------
    print("\n=== 3. CONSENSUS VS JUDGED TRUTHFULNESS ===")

    analysis = questions.dropna(
        subset=["average_consensus", "mean_judge_score"]
    ).copy()

    print(f"Questions with usable binary labels: {len(analysis)}")

    if len(analysis) >= 3:
        correlation = analysis[
            ["average_consensus", "mean_judge_score"]
        ].corr().iloc[0, 1]

        print(
            "Pearson correlation between consensus and "
            f"mean judged score: {correlation:.3f}"
        )
        print(
            "Interpret cautiously: this is exploratory, "
            "not proof of predictive power."
        )
    else:
        print(
            "Too few questions for a meaningful correlation "
            "analysis. Collect more data."
        )

    # ---------------------------------------------------------
    # 5. Compare lower- and higher-consensus questions
    # ---------------------------------------------------------
    print("\n=== 4. RESULTS BY CONSENSUS LEVEL ===")

    if len(analysis) >= 4:
        analysis["consensus_group"] = pd.qcut(
            analysis["average_consensus"],
            q=3,
            labels=["Lower", "Middle", "Higher"],
            duplicates="drop",
        )

        bin_summary = (
            analysis.groupby(
                "consensus_group",
                observed=True,
            )
            .agg(
                questions=("question_id", "count"),
                mean_consensus=("average_consensus", "mean"),
                mean_judged_score=("mean_judge_score", "mean"),
                mean_truthful_rate=("truthful_rate", "mean"),
                mean_untruthful_rate=("untruthful_rate", "mean"),
            )
            .reset_index()
        )

        for column in [
            "mean_judged_score",
            "mean_truthful_rate",
            "mean_untruthful_rate",
        ]:
            bin_summary[column] *= 100

        print(bin_summary.to_string(index=False))

        bin_summary.to_csv(
            OUTPUT_DIR / "truthfulqa_consensus_bins.csv",
            index=False,
        )
    else:
        print(
            "Not enough questions to form stable consensus groups."
        )

    # ---------------------------------------------------------
    # 6. Save the row-level data with usable labels
    # ---------------------------------------------------------
    valid.to_csv(
        OUTPUT_DIR / "truthfulqa_valid_evaluations.csv",
        index=False,
    )

    print("\n=== REPORTS SAVED ===")
    print("truthfulqa_model_summary.csv")
    print("truthfulqa_question_summary.csv")
    print("truthfulqa_consensus_bins.csv (if enough data)")
    print("truthfulqa_valid_evaluations.csv")


if __name__ == "__main__":
    main()
