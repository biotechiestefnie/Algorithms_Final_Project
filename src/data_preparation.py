# Import packages and modules
from Bio import SeqIO
import random
import os

def count_promoters(fasta):
    """
    Count H. sapien promoters extracted from EDPnew ranging from -250
    to 250bp, with only "representative set of not closely related seqs"
    Parameter:
        fasta: path to promoters_rep.fa
    Returns:
        Integer count of promoter sequences
    """
    count = 0
    fh = open(fasta)
    for line in fh:
        if line[:1] == ">":
            count = count + 1
    fh.close()
    return count


def sample_promoters(fasta, out_fasta, k):
    """
    Randomly sample a subset of H. sapien promoters from the
    representative set to create prototype data
    Parameters:
        fasta: path to promoters_rep.fa
        out_fasta: output file path for prototype_promoters.fa
        k: number of promoters to sample
    Returns:
        None. Writes k sampled promoters to out_fasta
    """
    records = list(SeqIO.parse(fasta, "fasta"))
    subset = random.sample(records, k)
    SeqIO.write(subset, out_fasta, "fasta")


# Driver code for execution
if __name__ == "__main__":
    print(os.getcwd())

    n = count_promoters("data/prototype/promoters_rep.fa")
    print(n)

    sample_promoters(
        "data/prototype/promoters_rep.fa",
        "data/prototype/prototype_promoters_positive.fa",
        300
    )
