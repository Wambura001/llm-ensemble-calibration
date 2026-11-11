
import json
from pathlib import Path

import pandas as pd

from src.ensemble import OllamaEnsembleManager
from src.consensus import LexicalConsensusEngine
from src.evaluator import TruthfulQAEvaluator


DATASET_PATH = (
    "hf://datasets/truthfulqa/truthful_qa/"
    "generation/validation-00000-of-00001.parquet"
)

OUTPUT_DIR = Path("data/processed")
QUESTION_OUTPUT = OUTPUT_DIR / "truthfulqa_question_results.csv"
MODEL_OUTPUT = OUTPUT_DIR / "truthfulqa_model_evaluations.csv"

N_QUESTIONS = 100  # Start small; expand after checking the results.


def to_list(value):
    """Normalize dataset answer fields into Python lists."""
    if value is None:
        return []

    if isinstance(value, float) and pd.isna(value):
        return []

    if isinstance(value, (list, tuple)):
        return list(value)

    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

        return [value]

    return [value]


def main():
    print("Loading TruthfulQA dataset...")
    df = pd.read_parquet(DATASET_PATH)

    required = {"question", "best_answer", "correct_answers",
                "incorrect_answers"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {missing}"
        )

    sample = df.sample(
    n=min(N_QUESTIONS, len(df)),
    random_state=42,
    ).copy()

    print(f"Loaded {len(df)} questions; processing {len(sample)}.")

    ensemble = OllamaEnsembleManager()
    consensus_engine = LexicalConsensusEngine()
    evaluator = TruthfulQAEvaluator(judge_model="phi3")

    question_records = []
    model_records = []

    for index, (_, row) in enumerate(sample.iterrows(), start=1):
        question = str(row["question"])
        reference_answer = str(row["best_answer"])
        correct_answers = to_list(row["correct_answers"])
        incorrect_answers = to_list(row["incorrect_answers"])

        question_id = str(row.get("question_id", index))

        print(f"\n[{index}/{len(sample)}] {question}")

        responses = ensemble.get_ensemble_responses(question)

        if not responses:
            print("No usable model responses; skipping.")
            continue

        # Measure agreement separately from answer correctness.
        metrics = consensus_engine.evaluate_ensemble_consensus(
            responses
        )

        per_model_evaluations = {}

        for model_name, answer in responses.items():
            print(f"Evaluating response from {model_name}...")

            evaluation = evaluator.evaluate_answer(
                question=question,
                model_answer=answer,
                correct_answers=correct_answers,
                incorrect_answers=incorrect_answers,
            )

            per_model_evaluations[model_name] = evaluation

            # One row per individual model response.
            model_records.append({
                "question_id": question_id,
                "question": question,
                "model_name": model_name,
                "model_response": answer,
                "reference_answer": reference_answer,
                "correct_answers": json.dumps(
                    correct_answers, ensure_ascii=False, default=str
                ),
                "incorrect_answers": json.dumps(
                    incorrect_answers, ensure_ascii=False, default=str
                ),
                "judge_model": evaluation["judge_model"],
                "evaluation_status": evaluation["evaluation_status"],
                "judge_label": evaluation["judge_label"],
                "judge_score": evaluation["judge_score"],
                "judge_rationale": evaluation["judge_rationale"],
                "average_consensus": metrics["average_consensus"],
                "consensus_confidence_score": metrics["confidence_score"],
                "is_calibrated_by_current_rule": metrics["is_calibrated"],
            })

        # One row per question, including all model responses.
        question_records.append({
            "question_id": question_id,
            "question": question,
            "reference_answer": reference_answer,
            "model_responses": json.dumps(
                responses, ensure_ascii=False
            ),
            "model_evaluations": json.dumps(
                per_model_evaluations, ensure_ascii=False
            ),
            "average_consensus": metrics["average_consensus"],
            "consensus_confidence_score": metrics["confidence_score"],
            "is_calibrated_by_current_rule": metrics["is_calibrated"],
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(question_records).to_csv(
        QUESTION_OUTPUT, index=False, encoding="utf-8-sig"
    )

    pd.DataFrame(model_records).to_csv(
        MODEL_OUTPUT, index=False, encoding="utf-8-sig"
    )

    print("\nPipeline complete.")
    print(f"Questions saved: {len(question_records)}")
    print(f"Model evaluations saved: {len(model_records)}")
    print(f"Question results: {QUESTION_OUTPUT}")
    print(f"Model evaluations: {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
