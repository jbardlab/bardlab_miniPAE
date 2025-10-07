import numpy as np
import json
import os
import re
import zipfile

# ------------------------------------------------------------------------------
# AF2_MiniPAE_v8.py
#
# This script processes AlphaFold2 prediction outputs (in .result.zip format)
# to extract the PAE (Predicted Aligned Error) region between an IDP (chain B)
# and a receptor (chain A), and identifies the most confident motif-like region
# based on a user-defined confidence threshold.
#
# Author: Martin Veinstein
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Function: parse_a3m_header
# Purpose: Parses the A3M header to extract sequence lengths and cardinality
# Example header: "#100,300 1" → lengths = [100, 300], cardinality = "1"
# ------------------------------------------------------------------------------

def parse_a3m_header(a3mhead):
    lengths_str, cardinalities = a3mhead.strip().lstrip("#").split()
    lengths = list(map(int, lengths_str.split(",")))
    return lengths, cardinalities

# ------------------------------------------------------------------------------
# Function: read_scores_from_zip
# Purpose: Extracts PAE scores and sequences from a .result.zip archive.
# Returns: Dictionary containing PAE minimums, formatted sequence, motif info.
# Arguments:
# - zip_path: path to the .result.zip file
# - threshold: PAE threshold to determine high-confidence interactions
# ------------------------------------------------------------------------------

def read_scores_from_zip(zip_path, threshold):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Find relevant JSON and A3M files inside the archive
        json_candidates = [f for f in zip_ref.namelist() if f.endswith(".json") and "rank_001" in f and "scores" in f]

        basename = os.path.basename(zip_path).replace(".result.zip", "")
        expected_a3m_name = f"{basename}.a3m"
        a3m_candidates = [f for f in zip_ref.namelist() if f.endswith(expected_a3m_name)]

        # Safety checks
        if not json_candidates:
            raise FileNotFoundError("No score JSON file found inside the archive.")
        if not a3m_candidates:
            raise FileNotFoundError("No A3M file found inside the archive.")

        # Load the score JSON and A3M sequence
        json_name = json_candidates[0]
        a3m_name = a3m_candidates[0]

        with zip_ref.open(json_name) as f:
            scores = json.load(f)

        with zip_ref.open(a3m_name) as f:
            lines = f.read().decode("utf-8").splitlines()
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


            # Extract the actual IDP sequence from the A3M file
            protein_sequence = None
            for i, line in enumerate(lines):
                if line.strip() == marker and i + 1 < len(lines):
                    protein_sequence = lines[i + 1].strip().replace("-", "")
                    break

        # Get the lengths of IDP and receptor
        if len(lengths) == 2:
            idp_len, receptor_len = lengths
        else:
            idp_len = receptor_len = lengths[0]  # monomer/homodimer

        # Load the full PAE matrix and verify its shape
        pae = np.array(scores["pae"])
        expect = idp_len + receptor_len
        assert pae.shape == (expect, expect), f"PAE shape mismatch: {pae.shape} vs ({expect},{expect})"

        # Extract minimum PAE values for each residue of the IDP to the receptor
        pae_idp_receptor_min = pae[:idp_len, idp_len:].min(axis=1).tolist()

        # Format the sequence: uppercase = confident residue; lowercase = not confident
        if protein_sequence and len(protein_sequence) == len(pae_idp_receptor_min):
            formatted_sequence = "".join(
                aa.upper() if pae_val < threshold else aa.lower()
                for aa, pae_val in zip(protein_sequence, pae_idp_receptor_min)
            )
        else:
            formatted_sequence = "N/A"

        # Identify longest high-confidence motif-like region in uppercase
        match = re.search(r"[A-Z].*[A-Z]", formatted_sequence)
        if match:
            motif_sequence = match.group(0)
            motif_start = match.start() + 1  # 1-based indexing
            motif_stop = match.end()
        else:
            motif_sequence = motif_start = motif_stop = "N/A"

        # Return all computed metrics
        return {
            "pae_idp_receptor_min": pae_idp_receptor_min,
            "formatted_sequence": formatted_sequence,
            "motif_sequence": motif_sequence,
            "motif_start": motif_start,
            "motif_stop": motif_stop
        }

# ------------------------------------------------------------------------------
# Script Entry Point
# Loop through .result.zip files in the given folder, extract PAE metrics,
# and write the results to a CSV file.
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Ask for user input (with defaults)
    folder_root = input("Enter the folder path (default: ./data/example): ").strip() or "./data/example"
    thresh_input = input("Enter PAE threshold (default 4): ").strip()
    threshold = float(thresh_input) if thresh_input else 4.0

    # Prepare output file
    output_file = os.path.join(folder_root, f"metrics_miniPAE-thresh{threshold}.csv")

    with open(output_file, "w") as csvfile:
        # Write header line
        csvfile.write("experiment_name,miniPAE,pae_list,formatted_sequence,motif_sequence,motif_start,motif_stop\n")

        # Loop through all AlphaFold zip files in the directory
        for filename in os.listdir(folder_root):
            if not filename.endswith(".result.zip"):
                continue

            zip_path = os.path.join(folder_root, filename)
            experiment_name = filename.replace(".result.zip", "")

            try:
                # Extract and write results
                out = read_scores_from_zip(zip_path, threshold)
                pae_list = out["pae_idp_receptor_min"]
                mini_pae = min(pae_list) if pae_list else "N/A"

                csvfile.write(
                    f"{experiment_name},{mini_pae}," +
                    ";".join(map(str, pae_list)) + "," +
                    f"{out['formatted_sequence']}," +
                    f"{out['motif_sequence']}," +
                    f"{out['motif_start']}," +
                    f"{out['motif_stop']}\n"
                )
            except Exception as e:
                # Print errors without interrupting the batch run
                print(f"Error in {filename}: {e}")

    print(f"Results written to {output_file}")
