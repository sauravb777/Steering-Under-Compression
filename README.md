# Steering Under Compression — Supplementary Materials

Supplementary code, data, and audit materials for:

**Steering Under Compression: Dose-Response, Capability Cost, and Failure Asymmetry in Quantized LLMs**

*Saurav Bhandari and Benjamin Wade, 2026*

---

## Contents

### `code/`

* Collection pipeline: `steerquant_phase0_harness.py`, `run_matrix.py`
* Scoring and judging scripts
* GSM8K parser rescore scripts (v2.3 and v2.3.1)
* E* analysis using both as-coded and censored conventions
* Option C REML + Knapp-Hartung pooling analysis
* Length-matrix summaries and failure detectors, including unit tests
* Stimulus generation and figure generation scripts

### `data/results/`

* 105 confirmatory result files
* 45 sentiment cells and 60 length cells
* 4 models × 3 quantization schemes × 5 resamples
* Full per-generation records and environment metadata (`meta.env`)
* One-line run summaries

### `data/vectors/`

* 105 extracted steering vectors
* Unit-norm `.npy` files, one per cell

### `method_notes/`

* Preregistration and dated method notes for every analysis decision
* Parser versions (v2, v2.3, v2.3.1)
* Censored E* convention and Option C pooling notes
* Materials corresponding to the paper's Appendix A deviation ledger

### `gate_c_audit/`

* Complete Gate C rescore audit
* Flip list (1,644 items)
* Rescored sibling results
* Validation outputs

### `tables/`

* E* crossing report using both conventions
* Final threshold-10% table
* Per-alpha length matrix underlying the failure-asymmetry analysis

### `MANIFEST.md`

* SHA256 checksums for every file in the repository

---

## Environment

* **PyTorch:** 2.6.0
* **CUDA:** 12.4
* **Transformers:** 4.57.3
* **Hardware:** NVIDIA RTX 4090 GPUs
* **Python dependencies:** `code/requirements.txt`

Environment information is also recorded per run in the corresponding result files' `meta.env`.

---

## License

All rights reserved.

The code, data, documentation, and other materials in this repository are provided for review and reference purposes only. No part of this repository may be used, reproduced, modified, distributed, or incorporated into other works without prior written permission from the copyright holders.

---

## Citation

If you reference this work, please cite:

**Bhandari, Saurav, and Benjamin Wade.** *Steering Under Compression: Dose-Response, Capability Cost, and Failure Asymmetry in Quantized LLMs.* 2026.

The arXiv link will be added here once the paper is publicly announced.
