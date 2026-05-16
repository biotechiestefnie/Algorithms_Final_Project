from Bio import SeqIO

def read_fasta(filepath):
    """
    Read a FASTA file and return a dictionary mapping sequence IDs to
    plain DNA strings. This ensures consistent sequence representation
    for Markov model training.
    Parameter:
        filepath (str): path to FASTA file
    Returns:
        dict[str, str]: dictionary mapping sequence ID → DNA string
    """

    seq_dict = {}

    for record in SeqIO.parse(filepath, "fasta"):
        seq_dict[record.id] = str(record.seq)   # <-- CRITICAL: convert to string

    return seq_dict


def load_class_seqs(class_fasta_paths):
    """
    Load FASTA files for each class label and return a dictionary mapping
    class_label → list of DNA strings.
    Parameter:
        class_fasta_paths (dict): mapping of class_label → filepath
    Returns:
        dict[str, list[str]]: training sequences for each genomic feature dataset
    """

    if not isinstance(class_fasta_paths, dict):
        raise TypeError(
            f"class_fasta_paths must be a dict mapping class_label → filepath, "
            f"got {type(class_fasta_paths)}"
        )

    training_data = {}

    for class_label, filepath in class_fasta_paths.items():

        seq_dict = read_fasta(filepath)

        if not isinstance(seq_dict, dict):
            raise TypeError(
                f"read_fasta() must return a dict mapping ID → string, "
                f"got {type(seq_dict)} for class_label={class_label}"
            )

        # Convert dictionary values to list of DNA strings
        seq_list = list(seq_dict.values())

        training_data[class_label] = seq_list

    return training_data
