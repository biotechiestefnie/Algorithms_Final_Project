#!/usr/bin/env python

"""
Executable to extract intron sequences from a GENCODE-style GTF and a matching genome FASTA
Parameters:
    gtf_path (str): Path to the GTF annotation file (e.g., gencode.v19.annotation.gtf)
    genome_path (str): Path to the genome FASTA file (e.g., GRCh37.p13.genome.fa)
    output_path (str): Path to the output FASTA file containing extracted intron sequences
Returns:
    None
    Writes intron sequences to `output_path` in FASTA format.
"""

from collections import defaultdict
from Bio import SeqIO


def extract_introns(gtf_path, genome_path, output_path):
    """Extract introns from a GTF + genome FASTA and write them to FASTA."""
    print("Loading genome FASTA...")
    genome = SeqIO.to_dict(SeqIO.parse(genome_path, "fasta"))

    print("Parsing GTF and collecting exons...")
    transcripts = defaultdict(list)

    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if fields[2] != "exon":
                continue

            chrom = fields[0]
            start = int(fields[3])
            end = int(fields[4])
            strand = fields[6]

            attrs = fields[8]
            tid = [x for x in attrs.split(";") if "transcript_id" in x][0].split('"')[1]

            transcripts[tid].append((chrom, start, end, strand))

    print("Extracting introns and writing FASTA...")
    with open(output_path, "w") as out:
        for tid, exons in transcripts.items():
            exons.sort(key=lambda x: x[1])

            for i in range(len(exons) - 1):
                chrom, s1, e1, strand = exons[i]
                _, s2, e2, _ = exons[i + 1]

                intron_start = e1 + 1
                intron_end = s2 - 1

                if intron_end > intron_start:
                    seq = genome[chrom].seq[intron_start - 1:intron_end]
                    if strand == "-":
                        seq = seq.reverse_complement()

                    out.write(f">{tid}_intron{i+1}\n{seq}\n")

    print(f"Done. Output written to {output_path}")


# Execution block to create raw intron fasta
if __name__ == "__main__":
    extract_introns(
        gtf_path="data/raw/gencode.v19.annotation.gtf",
        genome_path="data/raw/GRCh37.p13.genome.fa",
        output_path="data/raw/introns_raw.fa"
    )
