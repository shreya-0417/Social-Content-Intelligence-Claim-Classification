# Changelog

All notable changes to this repository are documented here.
This changelog starts with the presentation migration; the analytical work it
packages predates it and is credited in [README.md](README.md#credits-and-acknowledgements).

## [1.0.0] - 2026-11-26

### Changed
- Re-titled the project to **Social Content Intelligence: Claim Classification**
  across the README, documentation and notebook headers.
- Reorganised the repository into `data/`, `notebooks/`, `docs/`, `assets/` and
  `screenshots/`; the previous flat `Images/` and `Resources/` folders no longer
  exist.
- Renamed notebooks to an explicit analysis order
  (`01_initial_eda` &rarr; `06_model_development`).
- Replaced hard-coded Windows dataset paths in all six notebooks with portable
  `pathlib` paths resolved from the repository root.
- Rewrote `README.md` from scratch with overview, methodology, results,
  installation, execution and attribution sections.

### Added
- `requirements.txt` with pinned minimum versions for every imported library.
- `docs/architecture.mmd` and `docs/architecture.png` describing the notebook
  pipeline.
- `screenshots/dashboard-overview.png`, `dashboard-kpis.png`,
  `dashboard-analysis.png` and `dashboard-details.png`, captured from the
  original Tableau summary dashboard.
- `CONTRIBUTING.md`, `CHANGELOG.md`, `.github/workflows/ci.yml` and an expanded
  `.gitignore`.
- `LICENSE` restating the upstream all-rights-reserved terms together with the
  migration note.

### Removed
- Unreferenced image assets (`placeholder.png`, `readme_image.jpg`,
  `tiktok_neon_colors.jpg`, `Logo_pods_case.jpeg`, `Course5_RegressionAnalysis.png`,
  `Course_6_Content.png`) and the unused `tiktok-claims-classification.Rproj`.
- The inherited personal banner from the README hero position; the asset is
  retained in `assets/images/` for attribution.
