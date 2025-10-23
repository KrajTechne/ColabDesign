from Bio import SeqIO
from antpack import SingleChainAnnotator
import pandas as pd
import os

def get_seqs_from_fasta(fasta_path: str, suffix_split: str = None, seq_type: str = None):
    """ Function to return a dictionary of sequence IDs and sequences from a fasta file
    Args:
        fasta_path (str): path to fasta file containing sequences
    Returns:
        dict: dictionary of sequence IDs and sequences
    """
    seqs = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        record_id_split = record.id.split(suffix_split)[0] if suffix_split else record.id
        record_id_final = f"{record_id_split}_{seq_type}" if seq_type else record_id_split
        seqs[record_id_final] = str(record.seq)
    return seqs