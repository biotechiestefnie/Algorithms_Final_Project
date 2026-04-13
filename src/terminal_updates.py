#!/usr/bin/env python3
import sys
from src.data_loading import load_class_seqs
from src.train_models import train_markov
from src.analyze_data import classify_test_set
from src.output_to_file import write_all_outputs
from src.analyze_data import compute_accuracy

def main():
    if len(sys.argv) != 4:
        print("Usage: Python terminal_updates.py <data_dir> <output_dir> <k>")
        sys.exit(1)

    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    k = int(sys.argv[3])

    # Load sequences
    sequences_by_class = load_class_seqs(data_dir)
    total = sum(len(v) for v in sequences_by_class.values())
    num_classes = len(sequences_by_class)

    print(f"Loaded {total} sequences across {num_classes} classes")

    # Train models
    print("Training Markov models...", end="", flush=True)
    models = train_markov(sequences_by_class, k)
    print(" done")

    # 3. Classify
    print("Classifying sequences...", end="", flush=True)
    results = classify_test_set(models, sequences_by_class, k)
    print(" done")

    # Write results
    print(f"Writing results to {output_dir}...", end="", flush=True)
    write_all_outputs(results, output_dir)
    print(" complete")

    # Compute accuracy
    accuracy = compute_accuracy(results)
    print(f"Accuracy at k={k}: {accuracy:.2f}")

if __name__ == "__main__":
    main()
