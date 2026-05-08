# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-08

### Added
- **Core API**: Introduced the unified `prepare()`, `fit()`, and `generate()` foundational architecture for all synthesis modules.
- **Tabular Domain**: Implemented `CTGANSynthesizer` and `GaussianCopulaSynthesizer` powered by SDV.
- **Text Domain**: Implemented `MarkovSynthesizer`, `TemplateSynthesizer`, and `TransformerSynthesizer` via Hugging Face.
- **Image Domain**: Implemented `ImageAugmentor` with fast robust augmentations using Albumentations.
- **Time-Series Domain**: Implemented `PARSynthesizer` for deep sequential data generation.
- **Examples**: Shipped an extensive interactive `syntharc_showcase.ipynb` Jupyter Notebook tutorial alongside basic `.py` standalone examples.
- **Documentation**: Deployed comprehensive markdown guides inside `docs/` detailing APIs and getting-started tutorials.
- **CI/CD**: Added robust GitHub Actions workflows for continuous integration (`ci.yml`) matrix-testing Python 3.10 to 3.12, and automated PyPI publishing (`publish.yml`) via Trusted Publishing.
- **Quality Assurance**: Added `pytest` coverage testing and integrated `pre-commit` hooks for `ruff` and `mypy` strict linting.
