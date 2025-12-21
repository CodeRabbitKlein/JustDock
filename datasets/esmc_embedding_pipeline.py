"""
Generate protein language model embeddings with the ESMC-600M checkpoint.

This script mirrors the expected output format used by the existing
ESM2-based preprocessing utilities so that the resulting embeddings can be
consumed anywhere an ``esm_embeddings_path`` is accepted (training,
inference, scoring, etc.).

Typical usage:
    python datasets/esmc_embedding_pipeline.py \\
        --protein_ligand_csv data/protein_ligand_example_csv.csv \\
        --per_chain_output_dir data/esmc600m_chains \\
        --bundled_output_path data/esmc600m_embeddings.pt

Either ``--protein_ligand_csv`` (the CSV used elsewhere in this repo) or a
single ``--protein_path`` can be provided.
"""

import os
from argparse import ArgumentParser
from typing import Dict, Iterable, List, Tuple

import torch
from Bio.PDB import MMCIFParser, PDBParser
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--protein_ligand_csv",
        type=str,
        default="data/protein_ligand_example_csv.csv",
        help="CSV with a `protein_path` column as used elsewhere in the project.",
    )
    parser.add_argument(
        "--protein_path",
        type=str,
        default=None,
        help="Path to a single PDB/CIF file. Overrides --protein_ligand_csv when set.",
    )
    parser.add_argument(
        "--per_chain_output_dir",
        type=str,
        default="data/esmc600m_output",
        help="Directory where per-chain embeddings are saved. These files can be "
        "referenced directly via `esm_embeddings_path` for inference.",
    )
    parser.add_argument(
        "--bundled_output_path",
        type=str,
        default="data/esmc600m_embeddings.pt",
        help="Single .pt file mapping <pdb_basename>_chain_<idx> -> embedding tensor. "
        "Useful for training pipelines that load a single dictionary.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/esmc-600m",
        help="Hugging Face identifier for the ESMC-600M checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for embedding computation.",
    )
    parser.add_argument(
        "--representation_layer",
        type=int,
        default=33,
        help="Key used inside the saved `representations` dict to stay compatible with "
        "the existing ESM2-based loaders.",
    )
    return parser.parse_args()


BIOPYTHON_PDBPARSER = PDBParser(QUIET=True)
BIOPYTHON_CIFPARSER = MMCIFParser()

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",  # Selenomethionine -> Methionine
    "PHE": "F",
    "PRO": "P",
    "PYL": "O",
    "SER": "S",
    "SEC": "U",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "ASX": "B",
    "GLX": "Z",
    "XAA": "X",
    "XLE": "J",
}


def extract_sequences_from_structure(file_path: str) -> List[Tuple[str, str]]:
    """Return (chain_id, amino_acid_sequence) tuples for each valid chain."""
    if file_path.endswith(".pdb"):
        structure = BIOPYTHON_PDBPARSER.get_structure("pdb", file_path)
    elif file_path.endswith(".cif"):
        structure = BIOPYTHON_CIFPARSER.get_structure("cif", file_path)
    else:
        raise ValueError("protein is not pdb or cif")

    basename = os.path.basename(file_path)
    sequences: List[Tuple[str, str]] = []
    for chain_idx, chain in enumerate(structure[0]):
        seq = []
        for residue in chain:
            if residue.get_resname() == "HOH":
                continue
            c_alpha, n_atom, c_atom = None, None, None
            for atom in residue:
                if atom.name == "CA":
                    c_alpha = True
                elif atom.name == "N":
                    n_atom = True
                elif atom.name == "C":
                    c_atom = True
            if c_alpha and n_atom and c_atom:
                seq.append(THREE_TO_ONE.get(residue.get_resname(), "-"))
        sequences.append((f"{basename}_chain_{chain_idx}", "".join(seq)))
    return sequences


def collect_sequences(protein_paths: Iterable[str]) -> List[Tuple[str, str]]:
    collected: List[Tuple[str, str]] = []
    for path in protein_paths:
        collected.extend(extract_sequences_from_structure(path))
    return collected


def load_model_and_tokenizer(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def embed_sequences(
    sequences: List[Tuple[str, str]],
    tokenizer,
    model,
    device: str,
) -> Dict[str, torch.Tensor]:
    embeddings: Dict[str, torch.Tensor] = {}
    for chain_id, sequence in sequences:
        tokenized = tokenizer(
            sequence,
            add_special_tokens=True,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        with torch.no_grad():
            outputs = model(**tokenized)
        # Strip CLS/EOS to align with residue count expected downstream.
        residue_embeddings = outputs.last_hidden_state[0][1:-1].detach().cpu()
        embeddings[chain_id] = residue_embeddings
    return embeddings


def save_per_chain(
    embeddings: Dict[str, torch.Tensor], output_dir: str, representation_layer: int
):
    os.makedirs(output_dir, exist_ok=True)
    for chain_id, embedding in embeddings.items():
        out_path = os.path.join(output_dir, f"{chain_id}.pt")
        torch.save({"representations": {representation_layer: embedding}}, out_path)


def save_bundled(embeddings: Dict[str, torch.Tensor], bundled_output_path: str):
    target_dir = os.path.dirname(bundled_output_path)
    os.makedirs(target_dir if target_dir else ".", exist_ok=True)
    torch.save(embeddings, bundled_output_path)


def main():
    args = parse_args()
    if args.protein_path:
        protein_paths = [args.protein_path]
    else:
        import pandas as pd

        df = pd.read_csv(args.protein_ligand_csv)
        protein_paths = df["protein_path"].dropna().unique().tolist()

    sequences = collect_sequences(protein_paths)
    tokenizer, model = load_model_and_tokenizer(args.model_name, args.device)
    embeddings = embed_sequences(sequences, tokenizer, model, args.device)

    save_per_chain(embeddings, args.per_chain_output_dir, args.representation_layer)
    if args.bundled_output_path:
        save_bundled(embeddings, args.bundled_output_path)


if __name__ == "__main__":
    main()
