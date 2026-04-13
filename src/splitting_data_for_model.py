import random
import subprocess
import os

# Set seed for reproducibility
random.seed(43)

# Original prototype FASTA files
prototype_paths = {
    "promoter_positive": "data/prototype/promoters_positive.fa",
    "promoter_negative": "data/prototype/promoters_negative.fa",
    "exon_positive":     "data/prototype/exons_positive.fa",
    "exon_negative":     "data/prototype/exons_negative.fa",
    "intron_positive":   "data/prototype/introns_positive.fa",
    "intron_negative":   "data/prototype/introns_negative.fa",
    "repeat_positive":   "data/prototype/repeats_positive.fa",
    "repeat_negative":   "data/prototype/repeats_negative.fa"
}

train_ratio = 0.8  # 80/20 train/test for pos and neg of all classes

for class_label, fasta_path in prototype_paths.items():

    # Extract headers directly from FASTA file
    headers = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                headers.append(line[1:].strip())

    # Shuffle and split
    random.shuffle(headers)
    split = int(len(headers) * train_ratio)
    train_headers = headers[:split]
    test_headers  = headers[split:]

    # Write header lists
    train_header_file = f"{class_label}_train_headers.txt"
    test_header_file  = f"{class_label}_test_headers.txt"

    with open(train_header_file, "w") as f:
        f.write("\n".join(train_headers))

    with open(test_header_file, "w") as f:
        f.write("\n".join(test_headers))

    # seqtk to write FASTA files
    train_out = f"data/prototype/train/{class_label}_train.fa"
    test_out  = f"data/prototype/test/{class_label}_test.fa"

    subprocess.run(["seqtk", "subseq", fasta_path, train_header_file], stdout=open(train_out, "w"))
    subprocess.run(["seqtk", "subseq", fasta_path, test_header_file], stdout=open(test_out, "w"))

    # Clean up header lists
    os.remove(train_header_file)
    os.remove(test_header_file)
