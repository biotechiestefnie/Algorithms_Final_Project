"""

data_preparation.py

The following script holds all functions used to extract sequences from raw files, create
prototype positive and negative (control) datasets for each class, and generate training
and testing datasets containing all classes combined in each, for the final run.
"""

# Section 1: Imports, Seeds, Global Params

# Import packages and modules
import os
from Bio import SeqIO  # For parsing FASTA file
import random
import subprocess
import csv
import statistics
from typing import TextIO
import numpy as np

# Set random seed for sampling pos/neg prototype datasets from all classes
random.seed(42)
np.random.seed(42)

# Set global params
MIN_EXON_LENGTH = 150
MIN_INTRON_LENGTH = 500
MAX_INTRON_LENGTH = 1500
TRAIN_RATIO = 0.8


# Section 2: Shared Functions

# Helper functions
def infer_dataclass(path: str) -> str:
    """
    Infer feature from raw FASTA filename to track throughout script for
    data identification.
    Parameter:
        path (str): filesystem path to FASTA file
    Returns:
        str: inferred feature (e.g., 'promoters', 'introns')
    """

    name = os.path.basename(path).lower()

    # simple keyword matching
    if "prom" in name:
        return "promoters"
    elif "intron" in name:
        return "introns"
    elif "exon" in name:
        return "exons"
    elif "repeat" in name:
        return "repeats"
    elif "indel" in name:
        return "indels"

    # If filename does not match expected feature classes raise error
    raise ValueError(f"Cannot infer genomic feature from filename: {name}")


def rewrite_headers(records, class_label):
    """
    Rewrite FASTA headers to simple, model-friendly IDs
    Parameters:
        records (list[SeqRecord]): SeqRecords with arbitrary or metadata-heavy headers
        class_label (str): e.g., 'promoter_positive'
    Returns:
        list[SeqRecord]: new SeqRecords with rewritten .id, .name, .description
    """

    new_records = []
    for i, rec in enumerate(records, start=1):
        new_id = f"{class_label}_{i:03d}"
        rec.id = new_id
        rec.name = new_id
        rec.description = ""
        new_records.append(rec)

    return new_records


# Data Input
def load_fasta(path):
    """
    Load FASTA file to create dictionary mapping IDs to SeqRecord for model
    and call helper function to track data identification throughout script,
    return tuple of data class associated with dictionary of sequences
    Parameters:
        path (str): filesystem path to FASTA file
    Returns:
        tuple[str, dict[str, SeqRecord]]: (feature, {sequence_id: SeqRecord})
    """
    
    feature = infer_dataclass(path)

    seq_dict = {}
    for record in SeqIO.parse(path, "fasta"):
        seq_dict[record.id] = record

    return feature, seq_dict


def count_seqs(seq_dict, feature):
    """
    Print total number of sequences for each class's raw FASTA prior to sampling.
    Parameter:
        seq_dict (dict[str, SeqRecord]): ID → SeqRecord mapping created
            directly from the raw FASTA input
        feature (str): data class for labeling purposes in print statement
    """

    print(f"Total sequences in {feature} raw file: {len(seq_dict)}")


def filter_by_length(
        seq_dict, feature,
        min_exon_len=150,
        min_intron_len=500, max_intron_len=None
):
    """
    Filter sequences by feature-specific minimum (and optional maximum) length.
    Parameters:
        seq_dict (dict): dictionary mapping ID to SeqRecord
        feature (str): feature class of sequences (eg, promoters, introns)
        min_exon_len (int): minimum length of exons set to filter for high prob exon
        min_intron_len (int): minimum length of introns set to filter for high prob intron
        max_intron_len (int): maximum length of introns set to filter for high prob intron
    Returns:
        filtered (dict[str, SeqRecord]): dictionary mapping ID to SeqRecord,
                                         to be overwritten in driver block back
                                         to seq_dict
    """

    filtered = {}  # Initialize dictionary for filtered seqs

    for seq_id, rec in seq_dict.items():
        L = len(rec.seq)  # Establish length param

        # Mean exon len 170bp, 80% < 200bp
        if feature == "exons":

            if L < min_exon_len:
                continue

        # Mean intron len 5419bp, <10% are >11kbp
        elif feature == "introns":

            if L < min_intron_len:
                continue

            # Optional for setting max intron len for model optimization
            if max_intron_len is not None and L > max_intron_len:
                continue

        # No filtering for promoters and repeats
        filtered[seq_id] = rec

    return filtered


def sample_fasta(seq_dict, k: int):
    """
    Randomly sample up to k SeqRecords from a dictionary of sequences.
    """

    total = len(seq_dict)

    if total < k:
        raise ValueError(f"Requested {k} sequences but only {total} available.")

    seqs = list(seq_dict.values())

    if len(seqs) <= k:
         return seqs

    return random.sample(seqs, k)


def compute_class_stats(fasta_path, feature, label, source_file):
    """
    Reads prototype pos/neg files for each genomic feature and calculates preliminary overview
    of properties across all sequences for each file. These aggregated values form
    one row in a combined csv, yielding insight into composition and length distribution
    for each feature dataset pair
    Statistics computed for each feature in prototype run::
        total number of sequences
        length distribution: min, max, mean, median, standard deviation,
            interquartile range (IQR), coefficient of variation (CV)
        mean base composition across sequences (%A, %T, %G, %C, %GC)
        source raw FASTA file used to generate data
    Parameters:
        fasta_path (str): Path to feature class FASTA file (positive or negative) for prototype run
        feature (str): Genomic structural feature (e.g., 'promoters', 'exons')
        label (str): dataset label ('positive' or 'negative')
        source_file (str): Name of  raw FASTA file from which sequences
                           for this feature were originally drawn
    Returns:
        dict: One row of aggregated statistics for each feature daataset, for inclusion in csv
    """

    lengths = []
    len_A = []
    len_T = []
    len_G = []
    len_C = []
    GC_cont = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()
        L = len(seq)
        lengths.append(L)

        if L > 0:
            A = seq.count("A") / L
            T = seq.count("T") / L
            G = seq.count("G") / L
            C = seq.count("C") / L
            GC = G + C

        else:
            A = T = G = C = GC = 0

        len_A.append(A)
        len_T.append(T)
        len_G.append(G)
        len_C.append(C)
        GC_cont.append(GC)

    # Basic length stats
    total = len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)
    mean_len = sum(lengths) / total
    median_len = statistics.median(lengths)
    standard_dev = statistics.pstdev(lengths)  # population std
    q1 = statistics.quantiles(lengths, n=4)[0]
    q3 = statistics.quantiles(lengths, n=4)[2]
    iqr = q3 - q1
    cv = standard_dev / mean_len if mean_len > 0 else 0

    # Base composition percentages
    pct_A = (sum(len_A) / total) * 100
    pct_T = (sum(len_T) / total) * 100
    pct_G = (sum(len_G) / total) * 100
    pct_C = (sum(len_C) / total) * 100
    pct_GC_cont = (sum(GC_cont) / total) * 100

    return {
        "feature": feature,
        "label": label,
        "total_seqs": total,
        "min_len": min_len,
        "max_len": max_len,
        "mean_len": round(mean_len, 2),
        "median_len": round(median_len, 2),
        "standard_dev": round(standard_dev, 2),
        "iq_range": round(iqr, 1),
        "coeff_var": round(cv, 3),
        "% A": round(pct_A, 2),
        "% T": round(pct_T, 2),
        "% G": round(pct_G, 2),
        "% C": round(pct_C, 2),
        "% GC_cont": round(pct_GC_cont, 2),
        "source_file": source_file
    }


def write_preliminary_csv(rows, out_csv_path):
    """
    Write combined preliminary characteristics table to single csv file.
    Each row represents one (feature, label) prototype run and contains
    aggregated length and composition statistics for that file
    """

    # Headers for pos/neg datasets of each genomic feature
    fieldnames = [
        "feature", "label", "total_seqs",
        "min_len", "max_len", "mean_len", "median_len",
        "standard_dev", "iq_range", "coeff_var",
        "% A", "% T", "% G", "% C", "% GC_cont",
        "source_file"
    ]

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    with open(out_csv_path, "w", newline="") as f:  # type: TextIO
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# Section 3: Prototype Run Only Functions

def remove_sampled_from_dict(seq_dict, sampled_list):
    """
    Remove sampled SeqRecord objects from the original FASTA dictionary for positive
    prototype generation.
    Parameters:
        seq_dict (dict[str, SeqRecord]): ID → SeqRecord mapping
        sampled_list (list[SeqRecord]): sampled SeqRecord objects
    Returns:
        dict[str, SeqRecord]: updated dictionary with sampled removed
    """

    sampled_ids = {rec.id for rec in sampled_list}

    return {seq_id: rec for seq_id, rec in seq_dict.items() if seq_id not in sampled_ids}


def sample_negative_candidates(seq_dict_remaining, k=300):
    """
    Sample k SeqRecord objects from remaining sequence dictionary for
    negative prototype data generation
    Parameters:
        seq_dict_remaining (dict[str, SeqRecord]): ID → SeqRecord mapping
            post-removal of already-sampled positive sequences
        k (int): number of sequences to sample (default=300)
    Returns:
        list[SeqRecord]: list of sampled SeqRecord objects taken directly from
            seq_dict_remaining.values(); this list is then passed to
            shuffle_records_dinuc to generate dinucleotide-shuffled negative set for control
    """

    remaining_list = list(seq_dict_remaining.values())

    if len(remaining_list) < k:
        raise ValueError(f"Requested {k} negatives but only {len(remaining_list)} available.")

    return random.sample(remaining_list, k)


def dinuc_shuffle(seq):
    """
    Return a dinucleotide shuffle of DNA sequence to obscure class interdependencies for negative
    prototype data generation
    Parameters:
        seq (str or Seq): original negative sampled sequence
    Returns:
        str: dinucleotide-shuffled sequence
    """

    seq = str(seq)

    # Sequence must be ≥ 2 bp to form at least one dinucleotide
    if len(seq) < 2:
        # Raise error to indicate imbalance in neg dataset
        raise ValueError(
            "Sequence too short to shuffle, resulting in imbalanced datasets "
        )

    # Build list of overlapping dinucleotides from original sequence
    dinucs = [seq[i:i + 2] for i in range(len(seq) - 1)]

    # Randomly permute dinucleotide order
    random.shuffle(dinucs)

    # Reconstruct sequence by taking 1st dinuc, appending 2nd base of each subsequent dinuc
    shuffled = dinucs[0]
    for d in dinucs[1:]:
        shuffled += d[1]

    return shuffled


def shuffle_records_dinuc(sampled_list):
    """
    Apply dinucleotide-preserving shuffle to each SeqRecord in negative dataset for each feature
    Parameters:
        sampled_list (list[SeqRecord]): list of SeqRecord objects sampled for
            negative dataset generation for prototype run
    Returns:
        list[SeqRecord]: list of new SeqRecord objects whose sequences have
            been replaced with dinucleotide-shuffled versions of the originals;
            this list is then written to FASTA as negative data for prototype run
    """

    neg_records = []  # Initiate list for negative sequence data

    for rec in sampled_list:
        shuffled_seq = dinuc_shuffle(rec.seq)
        new_rec = rec[:]
        new_rec.seq = rec.seq.__class__(shuffled_seq)
        neg_records.append(new_rec)

    return neg_records


def write_negative_fasta(neg_records, outpath):
    """
    Write dinucleotide-shuffled negative SeqRecord objects to output FASTA file
    Parameters:
        neg_records (list[SeqRecord]): list of shuffled negative SeqRecord
            objects produced by shuffle_records_dinuc
        outpath (str): filesystem path for output FASTA file
    Returns:
        None: writes FASTA file to disk
    """

    SeqIO.write(neg_records, outpath, "fasta")


# Section 3: Final Run Only Functions
def extract_fasta_headers(fasta_path: str):
    """
    Extract all FASTA headers (without the leading '>') from a file
    Parameters:
    fasta_path (str): Path to the FASTA file
    Returns:
        list[str]: List of header strings
    Notes:
    - This function does not load sequences, only headers
    - Used for header-based splitting to avoid memory overhead
    """

    headers = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                headers.append(line[1:].strip())

    return headers


def split_headers(headers, train_ratio: float):
    """
    Shuffle and split a list of FASTA headers into train/test partitions
    Parameters:
        headers (list[str]): List of FASTA headers
    train_ratio (float): Fraction of headers to allocate to the training set
    Returns:
        (train_headers, test_headers): tuple[list[str], list[str]]
    Notes
    - Uses global random seed for reproducibility
    """

    random.shuffle(headers)
    split_idx = int(len(headers) * train_ratio)

    return headers[:split_idx], headers[split_idx:]


def extract_fasta_by_headers(
        fasta_path: str,
        header_list: list,
        out_fasta: str,
        temp_header_file: str
):
    """
    Extract sequences from FASTA file using seqtk and list of headers
    Parameters:
        fasta_path (str): Path to source FASTA file
        header_list (list[str]): Headers to extract (must match FASTA IDs)
        out_fasta (str): Output FASTA path
        temp_header_file (str): Temporary file to store header list for seqtk
    """

    # Write header list to temporary file
    with open(temp_header_file, "w") as f:
        f.write("\n".join(header_list))

    # Run seqtk subseq
    subprocess.run(
        ["seqtk", "subseq", fasta_path, temp_header_file],
        stdout=open(out_fasta, "w")
    )

    # Remove temporary file
    os.remove(temp_header_file)


def combine_and_shuffle(input_fastas, output_fasta):
    """
    Combine multiple FASTA files into a single dataset and shuffle sequence order
    Parameters:
    input_fastas (list[str]): List of FASTA filepaths to merge
    output_fasta (str): Path to the combined, shuffled FASTA file
    """

    all_records = []

    # Load sequences from each FASTA
    for path in input_fastas:
        for rec in SeqIO.parse(path, "fasta"):
            all_records.append(rec)

    # Shuffle using global seed
    random.shuffle(all_records)

    # Write combined shuffled FASTA
    SeqIO.write(all_records, output_fasta, "fasta")


def count_seqs_in_fasta(filepath):
    """
    Return the number of sequences in a single FASTA file.
    """
    return sum(1 for _ in SeqIO.parse(filepath, "fasta"))


def total_bp_in_fasta(filepath):
    """
    Return the total number of base pairs across all sequences in a FASTA file.
    """
    total_bp = 0
    for record in SeqIO.parse(filepath, "fasta"):
        total_bp += len(record.seq)
    return total_bp

