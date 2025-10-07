import numpy as np
import json
import os
import re

# ------------------------------------------------------------------------------
# AF2_MiniPAE_v7_non-zip.py
# 
# This script processes AlphaFold2 prediction outputs to extract a minimal 
# PAE (Predicted Aligned Error) region between an IDP (intrinsically disordered 
# protein, chain B) and a receptor (chain A), and identifies the most confident motif-like region.
# 
# Author: Martin Veinstein
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Function to parse the A3M header and extract sequence lengths and cardinality
# Example header format: "#100,300 1" (100 residues, 300 residues, cardinality 1)
# ------------------------------------------------------------------------------
def parse_a3m_header(a3mhead):
    lengths_str, cardinalities = a3mhead.strip().lstrip("#").split()
    lengths = list(map(int, lengths_str.split(",")))
    return lengths, cardinalities

# ------------------------------------------------------------------------------
# Main function to extract metrics from scores rank_001 (.json) file and sequence from (.a3m) file.
# folder_path:      path to experiment subfolder
# score_filename:   name of rank_001 AlphaFold score JSON file
# a3m_filename:     name of A3M alignment file
# threshold:        PAE confidence threshold to define "high-confidence" regions
# ------------------------------------------------------------------------------
def read_scores(folder_path, score_filename, a3m_filename, threshold):
    score_path = os.path.join(folder_path, score_filename)
    a3m_path   = os.path.join(folder_path, a3m_filename)

    # --- Load AlphaFold score JSON ---
    with open(score_path) as f:
        scores = json.load(f)

    # --- Parse A3M file to extract header and protein sequence ---
    protein_sequence = None
    with open(a3m_path) as f:
        lines   = f.readlines()
        a3mhead = lines[0].strip()
        lengths, card = parse_a3m_header(a3mhead)

        # Convert cardinality string to list of integers (if needed)
        card_list = list(map(int, card.split(",")))
        
        # Error handling for non-dimeric cases
        if len(lengths) == 2 and card == "1,1":
            idp_len, receptor_len = lengths
            marker = ">102"  # Heterodimer
        elif len(lengths) == 1 and card == "2":
            idp_len = receptor_len = lengths[0]
            marker = ">101"  # Homodimer
        elif len(lengths) == 1 and card == "1":
            raise ValueError("Monomeric structure detected — only dimeric models (homodimer or heterodimer) are supported.")
        else:
            raise ValueError(f"Unsupported structure configuration: lengths={lengths}, cardinalities={card}. Only dimers are supported.")


        # Extract sequence (next line after marker) and remove alignment gaps
        for i, line in enumerate(lines):
            if line.strip() == marker:
                if i + 1 < len(lines):
                    protein_sequence = lines[i + 1].strip().replace("-", "")
                break

    # --- Compute IDP and receptor lengths based on sequence type ---
    if len(lengths) == 2:
        idp_len, receptor_len = lengths
    else:
        idp_len = receptor_len = lengths[0]

    # --- Load PAE matrix and check dimensions ---
    pae = np.array(scores["pae"])
    expect = idp_len + receptor_len
    assert pae.shape == (expect, expect), f"PAE shape mismatch: {pae.shape} vs ({expect},{expect})"

    # --- Extract miniPAE: min PAE per receptor residue against IDP region ---
    pae_idp_receptor_min = pae[:idp_len, idp_len:].min(axis=0).tolist()

    # --- Format sequence: highlight high-confidence region using threshold ---
    if protein_sequence and len(protein_sequence) == len(pae_idp_receptor_min):
        formatted_sequence = "".join(
            aa.upper() if pae_val < threshold else aa.lower()
            for aa, pae_val in zip(protein_sequence, pae_idp_receptor_min)
        )
    else:
        formatted_sequence = "N/A"

    # --- Extract contiguous uppercase region (putative motif) and position ---
    match = re.search(r"[A-Z].*[A-Z]", formatted_sequence)
    if match:
        motif_sequence = match.group(0)
        motif_start    = match.start() + 1  # 1-based indexing
        motif_stop     = match.end()        # already 1-based length
    else:
        motif_sequence = motif_start = motif_stop = "N/A"

    # --- Return results as dictionary ---
    return {
        "pae_idp_receptor_min": pae_idp_receptor_min,
        "formatted_sequence":   formatted_sequence,
        "motif_sequence":       motif_sequence,
        "motif_start":          motif_start,
        "motif_stop":           motif_stop
    }

# ------------------------------------------------------------------------------
# MAIN SCRIPT EXECUTION
# Loops over experiment subfolders and extracts metrics from AF2 outputs
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Default path is ./data/example relative to current directory
    default_folder = os.path.join(os.getcwd(), "data", "example")
    folder_root    = input(f"Enter the folder path (default: {default_folder}): ").strip()
    folder_root    = folder_root or default_folder

    # Get PAE threshold (default = 4.0 Å)
    thresh_input   = input("Enter PAE threshold (default 4): ").strip()
    threshold      = float(thresh_input) if thresh_input else 4.0

    # Output file path
    output_file = os.path.join(folder_root, f"metrics_miniPAE-thresh{threshold}.csv")

    # Open output CSV file
    with open(output_file, "w") as csvfile:
        csvfile.write("experiment_name,miniPAE,pae_list,formatted_sequence,motif_sequence,motif_start,motif_stop\n")

        # Loop over subfolders (each representing a different prediction)
        for exp in os.listdir(folder_root):
            exp_path = os.path.join(folder_root, exp)
            if not os.path.isdir(exp_path):
                continue  # skip files

            # Find the score JSON file (usually named like rank_001_scores.json)
            scores_f = [f for f in os.listdir(exp_path)
                        if "scores" in f and "rank_001" in f and f.endswith(".json")]
            if not scores_f:
                print(f"No score file for {exp}")
                continue

            # Determine A3M filename from experiment name
            a3m_file = f"{exp}.a3m"

            # Try processing the folder
            try:
                out = read_scores(exp_path, scores_f[0], a3m_file, threshold)
                pae_list = out["pae_idp_receptor_min"]
                mini_pae = min(pae_list) if pae_list else "N/A"

                # Write results to CSV
                csvfile.write(
                    f"{exp},{mini_pae}," +
                    ";".join(map(str, pae_list)) + "," +
                    f"{out['formatted_sequence']}," +
                    f"{out['motif_sequence']}," +
                    f"{out['motif_start']}," +
                    f"{out['motif_stop']}\n"
                )

            except Exception as e:
                print(f"Error in {exp}: {e}")

    print(f"Results written to {output_file}")
