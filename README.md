# Steering Under Compression — Supplementary Materials

Supplementary code, data, and audit materials for:

> **Steering Under Compression: Dose-Response, Capability Cost, and Failure
> Asymmetry in Quantized LLMs.** Saurav Bhandari and Benjamin Wade, 2026.
> (arXiv link to be added on announcement.)

## Contents

- `code/` — collection pipeline (`steerquant_phase0_harness.py`, `run_matrix.py`),
  scoring and judging, the v2.3/v2.3.1 GSM8K parser rescore scripts, the E*
  analysis (both conventions: as-coded and censored), the Option C REML +
  Knapp-Hartung pooling analysis, length-matrix summaries, failure detectors
  (with unit tests), stimulus generation, and figure generation.
- `data/results/` — the 105 confirmatory result files (45 sentiment cells,
  60 length cells; 4 models x 3 quantization schemes x 5 resamples), each with
  full per-generation records and environment metadata (`meta.env`), plus
  one-line run summaries.
- `data/vectors/` — the 105 steering vectors (unit-norm, .npy), one per cell.
- `method_notes/` — the preregistration and dated method notes for every
  analysis decision (parser v2/v2.3/v2.3.1, censored-E* convention, Option C
  pooling), matching the paper's Appendix A deviation ledger.
- `gate_c_audit/` — the complete Gate C rescore audit: flip list (1,644 items),
  rescored sibling results, and validation outputs.
- `tables/` — E* crossing report (both conventions), final threshold-10% table,
  and the per-alpha length matrix underlying the failure-asymmetry analysis.
- `MANIFEST.md` — sha256 checksums for every file.

## Environment

PyTorch 2.6.0 (CUDA 12.4), Transformers 4.57.3, on NVIDIA RTX 4090 GPUs
(recorded per run in each result file's `meta.env`). Python dependencies:
`code/requirements.txt`.

## License

Code: MIT. Data and text materials: CC BY 4.0.
