import numpy as np
import json
import os
import re
import zipfile
import argparse
from pathlib import Path

# ------------------------------------------------------------------------------
# AF2_MiniPAE_JB_v1.py — Bidirectional miniPAE motif finder
# ------------------------------------------------------------------------------
# What it does
#   Analyzes AlphaFold-Multimer results to find confident, motif-like regions in
#   both binding directions using the PAE (Predicted Aligned Error) matrix.
#   For each rank model, it:
#     1) Parses the A3M header to get chain lengths and splits the query sequence
#        accordingly (robust to typical ">101 102" query markers; ignores gaps).
#     2) Computes, per residue in chain1, the minimum PAE to any residue in chain2
#        (pae_min_per_residue), for both directions (A->B and B->A).
#     3) Marks residues as confident (uppercase) either by:
#        - Threshold mode: pae_min_per_residue < --threshold (default 4.0), or
#        - Rolling mode: centered rolling average over a window (--window-size,
#          default 5) is < --window-avg (default 5.0).
#     4) Detects contiguous uppercase runs as candidate motifs, filters by
#        --motif-length (default 4), and records the longest such run per direction.
#     5) Selects the best binding direction by the shortest valid motif length;
#        if tied, the direction with the lower overall minimum PAE is chosen.
#
# Inputs supported
#   - A single .result.zip file
#   - A single unzipped AF2 result folder
#   - A directory containing .result.zip files and/or unzipped result folders
#     - Non-recursive by default (top-level only)
#     - Use -r/--recursive to search subdirectories
#     - Use --zip-dir to restrict to .result.zip files
#     - Use --unzipped-dir to restrict to unzipped result folders
#   If the provided path itself looks like an unzipped result, it’s processed
#   directly (no need to point at its parent).
#
# Key details and edge cases
#   - A3M parsing uses the header (e.g., "#100,300 1,1" or "#250 2") to split the
#     concatenated query into chain A and chain B. Supports homo- and heterodimers.
#   - If sequence length and PAE rows differ by exactly 1, the shorter is used to
#     continue (helps with occasional off-by-one indexing artifacts).
#   - Motif definition is the longest contiguous uppercase run meeting
#     --motif-length within a given direction; selection across directions favors
#     the shortest motif (tie-breaker: lower miniPAE).
#
# Output
#   A CSV with one row per rank model, columns:
#     experiment_name, rank, best_direction, miniPAE, alternative_miniPAE,
#     motif_length, motif_sequence, motif_start, motif_stop, pae_list,
#     formatted_sequence
#   - pae_list: semicolon-separated pae_min_per_residue values
#   - formatted_sequence: chain sequence with confident residues in uppercase,
#     as defined by the chosen mode (threshold or rolling)
#
# Command-line usage (examples)
#   python AF2_MiniPAE_JB_v1.py my.model.result.zip -o out.csv
#   python AF2_MiniPAE_JB_v1.py results/ --zip-dir -r -t 4.0 -m 4 -o out.csv
#   python AF2_MiniPAE_JB_v1.py results/ --unzipped-dir --rolling \
#       --window-size 5 --window-avg 5.0 -o out.csv
#   python AF2_MiniPAE_JB_v1.py some/unzipped_result_folder --rolling -o out.csv
#
# Flags summary
#   -t, --threshold     Float PAE cutoff for threshold mode (default: 4.0)
#   -m, --motif-length  Int minimum motif length (default: 4)
#   --rolling           Use rolling-average mode instead of per-residue threshold
#   --window-size       Int rolling window size (default: 5)
#   --window-avg        Float rolling average cutoff (default: 7.0)
#   -o, --output        Output CSV file path
#   -r, --recursive     Search directories recursively
#   --zip-dir           Treat input as directory of .result.zip files
#   --unzipped-dir      Treat input as directory of unzipped result folders
#   -v, --verbose       Print detailed parsing/progress info
#
# Credits
#   - Original concept/code adapted from work by Martin Veinstein; extended with
#     robust A3M parsing, unzipped directory handling, rolling-window mode, and
#     updated selection/CSV behavior.
# Last update by Jared Bard on 10/06/2025
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def _safe_motif_len(val):
    """Convert motif_length field to an int; return a large number when N/A."""
    try:
        return int(val)
    except Exception:
        return 10**9  # effectively infinity

def _looks_like_unzipped_result_dir(dir_path):
    """Heuristically decide if a directory looks like an unzipped AF2 result.
    Requires at least one .a3m and one JSON with 'scores' in its name.
    """
    if not os.path.isdir(dir_path):
        return False
    names = set(os.listdir(dir_path))
    has_a3m = any(n.endswith('.a3m') for n in names)
    has_scores_json = any(n.endswith('.json') and 'scores' in n for n in names)
    return has_a3m and has_scores_json

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
# Function: analyze_pae_direction
# Purpose: Analyzes PAE scores for a specific direction (chain1 -> chain2)
# Returns: Dictionary with PAE minimums, formatted sequence, and motif info
# ------------------------------------------------------------------------------

def analyze_pae_direction(pae, chain1_start, chain1_end, chain2_start, chain2_end, 
                         sequence, pae_threshold, motif_length_threshold, direction_name,
                         use_rolling=False, window_size=5, window_avg_threshold=5.0):
    """
    Analyze PAE scores from chain1 to chain2.
    
    Args:
        pae: Full PAE matrix
        chain1_start, chain1_end: Start and end indices for chain1
        chain2_start, chain2_end: Start and end indices for chain2
        sequence: The protein sequence for chain1
        pae_threshold: PAE threshold for confidence
        motif_length_threshold: Minimum length for valid motifs
        direction_name: String describing this direction (for debugging)
    
    Returns:
        Dictionary with analysis results
    """
    # Extract PAE submatrix: chain1 residues (rows) to chain2 residues (columns)
    pae_subset = pae[chain1_start:chain1_end, chain2_start:chain2_end]
    
    # Get minimum PAE for each residue in chain1 to any residue in chain2
    pae_min_per_residue = pae_subset.min(axis=1).tolist()
    # Sanity check: sequence length should match number of rows in pae_subset
    if sequence and len(sequence) != len(pae_min_per_residue):
        # Try to trim or pad if off-by-one due to indexing quirks; otherwise raise
        if abs(len(sequence) - len(pae_min_per_residue)) == 1:
            min_len = min(len(sequence), len(pae_min_per_residue))
            sequence = sequence[:min_len]
            pae_min_per_residue = pae_min_per_residue[:min_len]
        else:
            raise ValueError(
                f"Sequence length ({len(sequence)}) does not match PAE rows ({len(pae_min_per_residue)}) for {direction_name}."
            )
    # Overall minimum PAE across all residue pairs
    overall_min_pae = min(pae_min_per_residue) if pae_min_per_residue else float('inf')
    
    # Format the sequence based on confidence
    if sequence and len(sequence) == len(pae_min_per_residue):
        if use_rolling:
            # Compute centered rolling average using convolution (zero-padded at edges)
            kernel = np.ones(int(max(1, window_size)), dtype=float) / float(max(1, window_size))
            rolling_avg = np.convolve(np.array(pae_min_per_residue, dtype=float), kernel, mode='same')
            formatted_sequence = "".join(
                aa.upper() if avg < window_avg_threshold else aa.lower()
                for aa, avg in zip(sequence, rolling_avg)
            )
        else:
            formatted_sequence = "".join(
                aa.upper() if pae_val < pae_threshold else aa.lower()
                for aa, pae_val in zip(sequence, pae_min_per_residue)
            )
    else:
        formatted_sequence = "N/A"
    
    # Find the longest high-confidence motif that meets length threshold
    motif_sequence = motif_start = motif_stop = motif_length = "N/A"
    
    # Find all uppercase regions (high-confidence)
    matches = list(re.finditer(r"[A-Z]+", formatted_sequence))
    
    if matches:
        # Filter by minimum length and find the longest
        valid_matches = [m for m in matches if len(m.group(0)) >= motif_length_threshold]
        
        if valid_matches:
            # Get the longest motif
            longest_match = max(valid_matches, key=lambda m: len(m.group(0)))
            motif_sequence = longest_match.group(0)
            motif_start = longest_match.start() + 1  # 1-based indexing
            motif_stop = longest_match.end()
            motif_length = len(motif_sequence)
    
    return {
        "direction": direction_name,
        "pae_min_per_residue": pae_min_per_residue,
        "overall_min_pae": overall_min_pae,
        "formatted_sequence": formatted_sequence,
        "motif_sequence": motif_sequence,
        "motif_start": motif_start,
        "motif_stop": motif_stop,
        "motif_length": motif_length
    }

# ------------------------------------------------------------------------------
# Function: read_scores_from_zip_bidirectional
# Purpose: Extracts PAE scores in both directions for all rank JSONs
# ------------------------------------------------------------------------------

def read_scores_from_zip_bidirectional(zip_path, pae_threshold, motif_length_threshold, verbose=False,
                                       use_rolling=False, window_size=5, window_avg_threshold=5.0):
    """
    Process all ranking JSONs in the AlphaFold result zip.
    
    Returns:
        List of results, one for each rank JSON found
    """
    all_results = []
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Find ALL JSON files with scores (not just rank_001)
        json_candidates = sorted([f for f in zip_ref.namelist() 
                                 if f.endswith(".json") and "scores" in f])
        
        if not json_candidates:
            raise FileNotFoundError("No score JSON files found inside the archive.")
        
        # Get the A3M file
        basename = os.path.basename(zip_path).replace(".result.zip", "")
        expected_a3m_name = f"{basename}.a3m"
        a3m_candidates = [f for f in zip_ref.namelist() if f.endswith(expected_a3m_name)]
        
        if not a3m_candidates:
            raise FileNotFoundError("No A3M file found inside the archive.")
        
        # Read A3M file once (same for all ranks)
        a3m_name = a3m_candidates[0]
        
        with zip_ref.open(a3m_name) as f:
            lines = f.read().decode("utf-8").splitlines()
            a3mhead = lines[0].strip()
            lengths, card = parse_a3m_header(a3mhead)
            if verbose:
                print(f"A3M header: lengths={lengths}, card={card}")
            
            # Determine structure type
            if len(lengths) == 2 and card == "1,1":
                # Heterodimer
                chain_a_len, chain_b_len = lengths
                is_homodimer = False
            elif len(lengths) == 1 and card == "2":
                # Homodimer (two identical chains)
                chain_a_len = chain_b_len = lengths[0]
                is_homodimer = True
            elif len(lengths) == 1 and card == "1":
                raise ValueError("Monomeric structure detected — only dimeric models are supported.")
            else:
                raise ValueError(f"Unsupported structure: lengths={lengths}, cardinalities={card}")

            # Extract sequences (robust):
            # Find the first query marker line (usually ">101 102"), then split the next line by lengths
            chain_a_sequence = None
            chain_b_sequence = None

            query_marker_idx = None
            for i, line in enumerate(lines):
                s = line.strip()
                if s.startswith(">"):
                    # Tokenize after removing leading '>'
                    tokens = re.split(r"\s+", s[1:].strip())
                    # For dimers we expect at least two tokens (e.g., 101 102)
                    if len(tokens) >= 1 and tokens[0] in {"101", "102"}:
                        query_marker_idx = i
                        break

            if query_marker_idx is None or query_marker_idx + 1 >= len(lines):
                raise ValueError("Could not locate query sequence marker (e.g., '>101 102') in A3M file.")

            full_sequence = lines[query_marker_idx + 1].strip().replace("-", "")
            if verbose:
                print(f"Query marker line index: {query_marker_idx}; Query length (no gaps): {len(full_sequence)}")

            expected_total = chain_a_len + chain_b_len
            if len(full_sequence) < expected_total:
                raise ValueError(
                    f"A3M query sequence too short: got {len(full_sequence)}, expected at least {expected_total}"
                )

            # For both homo- and heterodimers, split by given lengths in header
            chain_a_sequence = full_sequence[:chain_a_len]
            chain_b_sequence = full_sequence[chain_a_len:chain_a_len + chain_b_len]
            if verbose:
                print(f"Chain A len/seq: {chain_a_len}/{chain_a_sequence[:10]}...")
                print(f"Chain B len/seq: {chain_b_len}/{chain_b_sequence[:10]}...")
        
        # Process each JSON file
        for json_name in json_candidates:
            # Extract rank number from filename (e.g., "rank_001" -> "001")
            rank_match = re.search(r'rank_(\d+)', json_name)
            rank_id = rank_match.group(1) if rank_match else "unknown"
            
            with zip_ref.open(json_name) as f:
                scores = json.load(f)
            
            # Load PAE matrix
            pae = np.array(scores["pae"])
            total_len = chain_a_len + chain_b_len
            assert pae.shape == (total_len, total_len), f"PAE shape mismatch: {pae.shape} vs ({total_len},{total_len})"
            
            # Analyze both directions
            results = []
            
            # Direction 1: Chain A (IDP) -> Chain B (receptor)
            if chain_a_sequence:
                result_a_to_b = analyze_pae_direction(
                    pae, 0, chain_a_len, chain_a_len, total_len,
                    chain_a_sequence, pae_threshold, motif_length_threshold, "ChainA_to_ChainB",
                    use_rolling=use_rolling, window_size=window_size, window_avg_threshold=window_avg_threshold
                )
                results.append(result_a_to_b)
            
            # Direction 2: Chain B (IDP) -> Chain A (receptor)
            if chain_b_sequence:
                result_b_to_a = analyze_pae_direction(
                    pae, chain_a_len, total_len, 0, chain_a_len,
                    chain_b_sequence, pae_threshold, motif_length_threshold, "ChainB_to_ChainA",
                    use_rolling=use_rolling, window_size=window_size, window_avg_threshold=window_avg_threshold
                )
                results.append(result_b_to_a)
            
            # Select the direction with the shortest motif; tie-breaker = lower mini PAE
            best_result = min(
                results,
                key=lambda x: (_safe_motif_len(x.get("motif_length")), x.get("overall_min_pae", float('inf')))
            )
            alternative_result = next((r for r in results if r is not best_result), best_result)
            
            best_result["rank"] = rank_id
            best_result["alternative_min_pae"] = alternative_result["overall_min_pae"]
            best_result["both_directions_analyzed"] = True
            best_result["selection_strategy"] = "shortest_motif_then_min_miniPAE"
            
            all_results.append(best_result)
    
    return all_results

# ------------------------------------------------------------------------------
# Function: read_scores_from_unzipped_dir
# Purpose: Extracts PAE scores in both directions for an unzipped result folder
# ------------------------------------------------------------------------------

def read_scores_from_unzipped_dir(dir_path, pae_threshold, motif_length_threshold, verbose=False,
                                  use_rolling=False, window_size=5, window_avg_threshold=5.0):
    """
    Process all ranking JSONs in an unzipped AlphaFold result directory.

    Returns:
        List of results, one for each rank JSON found
    """
    all_results = []

    # Find JSON candidates
    json_candidates = sorted([
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith('.json') and 'scores' in f
    ])
    if not json_candidates:
        raise FileNotFoundError("No score JSON files found inside the directory.")

    # Find A3M file
    a3m_candidates = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.a3m')]
    if not a3m_candidates:
        raise FileNotFoundError("No A3M file found inside the directory.")
    a3m_name = a3m_candidates[0]

    # Read A3M
    with open(a3m_name, 'r') as f:
        lines = f.read().splitlines()
        a3mhead = lines[0].strip()
        lengths, card = parse_a3m_header(a3mhead)
        if verbose:
            print(f"A3M header: lengths={lengths}, card={card}")

        # Determine structure type and lengths
        if len(lengths) == 2 and card == "1,1":
            chain_a_len, chain_b_len = lengths
        elif len(lengths) == 1 and card == "2":
            chain_a_len = chain_b_len = lengths[0]
        elif len(lengths) == 1 and card == "1":
            raise ValueError("Monomeric structure detected — only dimeric models are supported.")
        else:
            raise ValueError(f"Unsupported structure: lengths={lengths}, cardinalities={card}")

        # Locate query marker, get full sequence, split by lengths
        query_marker_idx = None
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith(">"):
                tokens = re.split(r"\s+", s[1:].strip())
                if len(tokens) >= 1 and tokens[0] in {"101", "102"}:
                    query_marker_idx = i
                    break
        if query_marker_idx is None or query_marker_idx + 1 >= len(lines):
            raise ValueError("Could not locate query sequence marker (e.g., '>101 102') in A3M file.")

        full_sequence = lines[query_marker_idx + 1].strip().replace("-", "")
        if verbose:
            print(f"Query marker line index: {query_marker_idx}; Query length (no gaps): {len(full_sequence)}")
        expected_total = chain_a_len + chain_b_len
        if len(full_sequence) < expected_total:
            raise ValueError(
                f"A3M query sequence too short: got {len(full_sequence)}, expected at least {expected_total}"
            )

        chain_a_sequence = full_sequence[:chain_a_len]
        chain_b_sequence = full_sequence[chain_a_len:chain_a_len + chain_b_len]
        if verbose:
            print(f"Chain A len/seq: {chain_a_len}/{chain_a_sequence[:10]}...")
            print(f"Chain B len/seq: {chain_b_len}/{chain_b_sequence[:10]}...")

    # Process each JSON
    total_len = chain_a_len + chain_b_len
    for json_path in json_candidates:
        rank_match = re.search(r'rank_(\d+)', os.path.basename(json_path))
        rank_id = rank_match.group(1) if rank_match else "unknown"

        with open(json_path, 'r') as f:
            scores = json.load(f)

        pae = np.array(scores["pae"])
        assert pae.shape == (total_len, total_len), f"PAE shape mismatch: {pae.shape} vs ({total_len},{total_len})"

        results = []
        if chain_a_sequence:
            result_a_to_b = analyze_pae_direction(
                pae, 0, chain_a_len, chain_a_len, total_len,
                chain_a_sequence, pae_threshold, motif_length_threshold, "ChainA_to_ChainB",
                use_rolling=use_rolling, window_size=window_size, window_avg_threshold=window_avg_threshold
            )
            results.append(result_a_to_b)
        if chain_b_sequence:
            result_b_to_a = analyze_pae_direction(
                pae, chain_a_len, total_len, 0, chain_a_len,
                chain_b_sequence, pae_threshold, motif_length_threshold, "ChainB_to_ChainA",
                use_rolling=use_rolling, window_size=window_size, window_avg_threshold=window_avg_threshold
            )
            results.append(result_b_to_a)

        # Select the direction with the shortest motif; tie-breaker = lower mini PAE
        best_result = min(
            results,
            key=lambda x: (_safe_motif_len(x.get("motif_length")), x.get("overall_min_pae", float('inf')))
        )
        # The alternative is the other result (if present)
        alternative_result = next((r for r in results if r is not best_result), best_result)

        best_result["rank"] = rank_id
        best_result["alternative_min_pae"] = alternative_result.get("overall_min_pae", "N/A")
        best_result["both_directions_analyzed"] = True
        best_result["selection_strategy"] = "shortest_motif_then_min_miniPAE"

        all_results.append(best_result)

    return all_results

# ------------------------------------------------------------------------------
# Function: find_result_zip_files
# Purpose: Find all .result.zip files, optionally recursively
# ------------------------------------------------------------------------------

def find_result_zip_files(input_path, recursive=False):
    """
    Find all .result.zip files in the given path.
    
    Args:
        input_path: Path to search
        recursive: Whether to search subdirectories
    
    Returns:
        List of paths to .result.zip files
    """
    files = []
    
    if os.path.isfile(input_path) and input_path.endswith(".result.zip"):
        files = [input_path]
    elif os.path.isdir(input_path):
        if recursive:
            # Use Path.rglob for recursive search
            path = Path(input_path)
            files = sorted([str(p) for p in path.rglob("*.result.zip")])
        else:
            # Just search the top level
            files = sorted([os.path.join(input_path, f) 
                          for f in os.listdir(input_path) 
                          if f.endswith(".result.zip")])
    
    return files

# ------------------------------------------------------------------------------
# Function: find_unzipped_result_dirs
# Purpose: Find unzipped result directories, optionally recursively
# ------------------------------------------------------------------------------

def find_unzipped_result_dirs(input_path, recursive=False):
    """Find folders that look like unzipped AF2 result directories."""
    dirs = []
    if os.path.isdir(input_path):
        if recursive:
            for root, subdirs, _ in os.walk(input_path):
                for d in subdirs:
                    cand = os.path.join(root, d)
                    if _looks_like_unzipped_result_dir(cand):
                        dirs.append(cand)
        else:
            for d in os.listdir(input_path):
                cand = os.path.join(input_path, d)
                if os.path.isdir(cand) and _looks_like_unzipped_result_dir(cand):
                    dirs.append(cand)
    return sorted(dirs)

# ------------------------------------------------------------------------------
# Function: process_files
# Purpose: Process a list of .result.zip file paths
# ------------------------------------------------------------------------------

def process_files(file_paths, pae_threshold, motif_length_threshold, output_file, verbose=False,
                  use_rolling=False, window_size=5, window_avg_threshold=5.0):
    """
    Process a list of .result.zip file paths.
    
    Args:
        file_paths: List of paths to .result.zip files
        pae_threshold: PAE threshold for confidence
        motif_length_threshold: Minimum motif length
        output_file: Path to output CSV file
    
    Returns:
        Number of successfully processed files
    """
    processed_count = 0
    total_ranks_processed = 0
    
    with open(output_file, "w") as csvfile:
        # Write header line (pae_list and formatted_sequence moved to the end)
        csvfile.write(
            "experiment_name,rank,best_direction,miniPAE,alternative_miniPAE,motif_length,"+
            "motif_sequence,motif_start,motif_stop,pae_list,formatted_sequence\n"
        )
        
        for path in file_paths:
            is_zip = isinstance(path, str) and path.endswith(".result.zip")
            filename = os.path.basename(path)
            experiment_name = filename.replace(".result.zip", "") if is_zip else os.path.basename(path)
            
            try:
                # Extract and analyze all ranks in both directions
                if is_zip:
                    all_ranks_results = read_scores_from_zip_bidirectional(
                        path, pae_threshold, motif_length_threshold, verbose=verbose,
                        use_rolling=use_rolling, window_size=window_size, window_avg_threshold=window_avg_threshold
                    )
                else:
                    all_ranks_results = read_scores_from_unzipped_dir(
                        path, pae_threshold, motif_length_threshold, verbose=verbose,
                        use_rolling=use_rolling, window_size=window_size, window_avg_threshold=window_avg_threshold
                    )
                
                for result in all_ranks_results:
                    pae_list = result["pae_min_per_residue"]
                    mini_pae = result["overall_min_pae"]
                    alt_pae = result.get("alternative_min_pae", "N/A")
                    
                    csvfile.write(
                        f"{experiment_name}," +
                        f"rank_{result['rank']}," +
                        f"{result['direction']}," +
                        f"{mini_pae}," +
                        f"{alt_pae}," +
                        f"{result['motif_length']}," +
                        f"{result['motif_sequence']}," +
                        f"{result['motif_start']}," +
                        f"{result['motif_stop']}," +
                        ";".join(map(str, pae_list)) + "," +
                        f"{result['formatted_sequence']}\n"
                    )
                    
                    total_ranks_processed += 1
                    print(f"  ✓ {experiment_name} rank_{result['rank']}: {result['direction']}, miniPAE={mini_pae:.2f}, motif_len={result['motif_length']}")
                
                processed_count += 1
                
            except Exception as e:
                print(f"  ✗ Error in {filename}: {e}")
    
    return processed_count, total_ranks_processed

# ------------------------------------------------------------------------------
# Function: setup_argparse
# Purpose: Configure command-line argument parser
# ------------------------------------------------------------------------------

def setup_argparse():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Process AlphaFold2 PAE scores bidirectionally to find the most confident binding direction.",
        epilog="Example usage:\n"
               "  %(prog)s data/results/                        # Process all .result.zip files in folder\n"
               "  %(prog)s data/results/ -r                     # Recursively process all subdirectories\n"
               "  %(prog)s protein1.result.zip                  # Process single file\n"
               "  %(prog)s data/ -t 5 -m 6 -o output.csv        # Custom thresholds and output\n"
               "  %(prog)s data/ -r -t 3.5                      # Recursive with custom PAE threshold",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "input",
        help="Path to a single .result.zip file, an unzipped result folder, or a directory containing them"
    )
    
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=4.0,
        help="PAE threshold for determining high-confidence interactions (default: 4.0)"
    )
    
    parser.add_argument(
        "-m", "--motif-length",
        type=int,
        default=4,
        help="Minimum length threshold for valid motifs (default: 4)"
    )

    # Rolling-window options
    parser.add_argument(
        "--rolling",
        action="store_true",
        help="Use rolling-window average of PAE to define high-confidence residues"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Window size for rolling average (default: 5)"
    )
    parser.add_argument(
        "--window-avg",
        type=float,
        default=7.0,
        help="Average PAE threshold within the rolling window (default: 7.0)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output CSV file path (default: auto-generated based on input)"
    )
    
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search directories recursively for .result.zip files"
    )

    parser.add_argument(
        "--zip-dir",
        action="store_true",
        help="Treat input as a directory of .result.zip files (use -r to search recursively)"
    )

    parser.add_argument(
        "--unzipped-dir",
        action="store_true",
        help="Treat input as a directory containing unzipped AF2 result folders (use -r to search recursively)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )
    
    return parser

# ------------------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------------------

def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' does not exist.")
        return 1
    
    # Determine inputs to process
    files_to_process = []
    if os.path.isfile(args.input):
        if args.input.endswith('.result.zip'):
            files_to_process = [args.input]
        elif _looks_like_unzipped_result_dir(os.path.dirname(args.input)):
            # If a file is given but its parent looks like a result dir, try parent
            parent = os.path.dirname(args.input)
            if _looks_like_unzipped_result_dir(parent):
                files_to_process = [parent]
        else:
            print(f"Error: Unsupported input file '{args.input}'. Expected a .result.zip or a result directory.")
            return 1
    elif os.path.isdir(args.input):
        # If the directory itself is a result folder, use it directly
        if _looks_like_unzipped_result_dir(args.input):
            files_to_process = [args.input]
        elif args.zip_dir:
            files_to_process = find_result_zip_files(args.input, args.recursive)
        elif args.unzipped_dir:
            files_to_process = find_unzipped_result_dirs(args.input, args.recursive)
        else:
            # auto-detect: collect both types one level (or recursively if -r)
            zips = find_result_zip_files(args.input, args.recursive)
            dirs = find_unzipped_result_dirs(args.input, args.recursive)
            files_to_process = zips + dirs
    else:
        files_to_process = []

    if not files_to_process:
        print(f"Error: No inputs found in '{args.input}'")
        if os.path.isdir(args.input) and not args.recursive:
            print("Tip: Use -r to search subdirectories recursively")
            print("     Or specify --zip-dir (for .result.zip files) or --unzipped-dir (for unzipped result folders)")
        return 1
    
    # Determine output file path
    if args.output:
        output_file = args.output
    else:
        if os.path.isfile(args.input):
            default_output_dir = os.path.dirname(args.input) or "."
            default_output_name = os.path.basename(args.input).replace(".result.zip", "")
        else:
            default_output_dir = args.input
            default_output_name = "combined"
        
        output_file = os.path.join(
            default_output_dir,
            f"{default_output_name}_miniPAE_bidirectional_t{args.threshold}_m{args.motif_length}.csv"
        )
    
    # Display processing parameters
    print(f"\n{'='*60}")
    print(f"🧬 AlphaFold PAE Bidirectional Analysis")
    print(f"{'='*60}")
    
    if os.path.isfile(args.input):
        print(f"📄 Input: Single file - {args.input}")
    else:
        print(f"📁 Input: Directory - {args.input}")
        if args.recursive:
            print(f"   Mode: Recursive (searching all subdirectories)")
        else:
            print(f"   Mode: Top-level only")
    
    print(f"\n⚙️  Settings:")
    print(f"   PAE threshold: {args.threshold}")
    print(f"   Minimum motif length: {args.motif_length}")
    print(f"   Files found: {len(files_to_process)}")
    print(f"   Output file: {output_file}")
    if args.rolling:
        print(f"   Rolling-window: ON (size={args.window_size}, avg<{args.window_avg})")
    
    print(f"\n🔄 Processing {len(files_to_process)} file(s)...\n")
    
    # Process files
    processed_count, total_ranks = process_files(
        files_to_process, args.threshold, args.motif_length, output_file, verbose=args.verbose,
        use_rolling=args.rolling, window_size=args.window_size, window_avg_threshold=args.window_avg
    )
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"✅ Analysis complete!")
    print(f"   Files processed: {processed_count}/{len(files_to_process)}")
    print(f"   Total rank models analyzed: {total_ranks}")
    print(f"   Results saved to: {output_file}")
    print(f"   Analysis: Bidirectional PAE (all ranks, both chain directions)")
    print(f"{'='*60}\n")
    
    return 0 if processed_count > 0 else 1

# ------------------------------------------------------------------------------
# Script entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    exit(main())