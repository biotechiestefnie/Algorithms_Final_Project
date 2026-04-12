#!/usr/bin/env python  # Shebang line for command line execution with proper interpreter

# Import packages and modules
import os
from Bio import SeqIO  # For parsing FASTA file
import random
import csv
import statistics
from typing import TextIO
import numpy as np

# Set random seed for sampling pos/neg prototype datasets from all classes
random.seed(42)
np.random.seed(42)


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


def sample_positive_fasta(seq_dict, feature, outpath, k=300):
    """
    Sample k positive sequences from FASTA dictionary and write to new FASTA
    file for positive prototype data in each class.
    Parameters:
        seq_dict (dict[str, SeqRecord]): ID → SeqRecord mapping
        feature (str): inferred genomic feature from raw filename
        outpath (str): filesystem path for output FASTA file
        k (int): number of sequences to sample (default=300)
    Returns:
        list[SeqRecord]: the sampled SeqRecord objects taken directly from
            seq_dict.values(); list is passed to remove_sampled_from_dict
            to eliminate previously sampled entries from original dictionary
    """

    total = len(seq_dict)

    if total < k:
        raise ValueError(f"Requested {k} sequences but only {total} available.")

    sampled_list = random.sample(list(seq_dict.values()), k)

    SeqIO.write(sampled_list, outpath, "fasta")

    return sampled_list


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


# Driver block for execution to create positive and negative prototype
# fasta files of each feature class for prototype run from raw data extracted
if __name__ == "__main__":

    raw_dir = "data/raw"
    out_dir = "data/prototype"

    # Files to process
    fasta_files = [
        "promoters_raw.fa",
        "exons_raw.fa",
        "introns_raw.fa",
        "repeats_raw.fa"
    ]

    for fname in fasta_files:
        inpath = os.path.join(raw_dir, fname)

        # Track feature class and create dictionary of seq ID:SeqRecord for each seq
        feature, seq_dict = load_fasta(inpath)
        print(f"\nProcessing {feature} from {inpath}")

        # ✔️ Count raw sequences in each input file
        count_seqs(seq_dict, feature)

        # ✔️ Filter for valid lengths in classes introns/exons
        seq_dict = filter_by_length(seq_dict, feature)

        # Positive output path
        pos_out = os.path.join(out_dir, f"{feature}_positive.fa")

        # sample raw positive SeqRecords
        sampled_pos = sample_positive_fasta(seq_dict, feature, pos_out, k=300)
        # rewrite headers BEFORE writing to disk
        clean_pos = rewrite_headers(sampled_pos, f"{feature}_positive")
        # write clean FASTA
        SeqIO.write(clean_pos, pos_out, "fasta")

        # Remove positives from dictionary
        seq_dict_remaining = remove_sampled_from_dict(seq_dict, sampled_pos)

        # Sample negatives
        neg_candidates = sample_negative_candidates(seq_dict_remaining, k=300)
        neg_records = shuffle_records_dinuc(neg_candidates)

        # Negative output path
        neg_out = os.path.join(out_dir, f"{feature}_negative.fa")
        # rewrite headers BEFORE writing to disk
        clean_neg = rewrite_headers(neg_records, f"{feature}_negative")
        # write clean FASTA for model
        SeqIO.write(clean_neg, neg_out, "fasta")

        print(f"Finished {feature}:")
        print(f"  → {pos_out}")
        print(f"  → {neg_out}")

    # After generating all prototype run files, compute preliminary stats
    prelim_rows = []
    prelim_csv = os.path.join(out_dir, "preliminary_characteristics.csv")

    for fname in fasta_files:
        feature = infer_dataclass(fname)
        source_file = fname

        pos_path = os.path.join(out_dir, f"{feature}_positive.fa")
        neg_path = os.path.join(out_dir, f"{feature}_negative.fa")

        prelim_rows.append(
            compute_class_stats(pos_path, feature, "positive", source_file)
        )
        prelim_rows.append(
            compute_class_stats(neg_path, feature, "negative", source_file)
        )

    write_preliminary_csv(prelim_rows, prelim_csv)
    print(f"\nWrote preliminary characteristics csv → {prelim_csv}")
