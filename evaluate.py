"""
Local evaluation: compute accuracy of predictions against train.json labels.

Usage:
    python evaluate.py submit_base.csv     # Evaluate on training set
    python evaluate.py submit_sft.csv
"""
import argparse
from utils import load_csv_submit, evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction accuracy")
    parser.add_argument("submit_csv", help="Path to submit CSV file")
    parser.add_argument("--ground_truth", default="train.json",
                        help="Path to labeled JSON (default: train.json)")
    args = parser.parse_args()

    predictions = load_csv_submit(args.submit_csv)
    result = evaluate(predictions, args.ground_truth)

    print(f"File: {args.submit_csv}")
    print(f"Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
    print(f"Correct: {result['correct']}/{result['total']}")


if __name__ == "__main__":
    main()
