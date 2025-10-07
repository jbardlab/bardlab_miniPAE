Forked from https://github.com/martinovein/AF2_MiniPAE#
# AF2_MiniPAE

AF2_MiniPAE is a lightweight tool for analyzing multiple AlphaFold2 multimer predictions (from **ColabFold**) to identify high-confidence motif-like regions between an **intrinsically disordered protein (IDP)** and a **receptor**. It computes the minimal **Predicted Aligned Error (PAE)** between chains and highlights the most confident interaction region.

> Author: Martin Veinstein

You can generate your AlphaFold2 predictions using ColabFold: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb

Or the batch version for multiple predictions: https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/batch/AlphaFold2_batch.ipynb#scrollTo=kOblAo-xetgx

## � Recommended ways to run (Pixi or Singularity)

This repo includes a Pixi environment (pixi.toml + pixi.lock) and a published container image. We recommend either:

- Running locally with Pixi (fast and reproducible on macOS/Linux/Windows)
- Running via Singularity/Apptainer using the published Docker image

### Run locally with Pixi

```bash
# Analyze a single unzipped result folder
pixi run python AF2_MiniPAE_JB_v1.py data/example/G3BP1_HUMAN-PRO_0000227773 \
  --rolling --window-size 5 --window-avg 7 -o data/example/out.csv

# Analyze a directory containing unzipped result folders (top-level only)
pixi run python AF2_MiniPAE_JB_v1.py data/example --unzipped-dir -o data/example/out.csv

# Analyze a directory containing .result.zip files (recursively)
pixi run python AF2_MiniPAE_JB_v1.py data/ --zip-dir -r -o data/out.csv
```

Notes:
- Use `-t/--threshold` for non-rolling mode (default 4.0)
- Use `--rolling --window-size N --window-avg X` for rolling-average confidence
- Results are saved to CSV; columns include miniPAE, motif info, pae_list, and formatted_sequence

### Run with Singularity/Apptainer

You can run directly from the published image on GHCR or pull a `.sif` first.


Pull to SIF, then run:

```bash
singularity run --bind "$PWD/data:/data" bardlab_minipae_v0.1.sif \
  python /app/AF2_MiniPAE_JB_v1.py /data/example/G3BP1_HUMAN-PRO_0000227773 \
  --rolling --window-size 5 --window-avg 7 -o /data/out.csv
```


Tip: On some HPC systems you may prefer `-B` instead of `--bind`, e.g. `-B /path/on/host:/path/in/container`.


## References

This script is adapted from the work of:

Omidi et al.
A. Omidi, M.H. Møller, N. Malhis, J.M. Bui, & J. Gsponer,
AlphaFold-Multimer accurately captures interactions and dynamics of intrinsically disordered protein regions,
Proc. Natl. Acad. Sci. U.S.A. 121 (44) e2406407121,
https://doi.org/10.1073/pnas.2406407121 (2024)

## 📜 License
This project is licensed under the MIT License.

