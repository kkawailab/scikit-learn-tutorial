# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview
This is a Japanese-language scikit-learn tutorial (scikit-learn完全ガイド) for teaching machine learning from basics to practical applications. The content is entirely in Japanese, including code comments and documentation.

## Development Commands

### Installing Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

### Running Sample Scripts
```bash
python scripts/01_basic_classification.py
python scripts/02_data_preprocessing.py
python scripts/03_regression_analysis.py
python scripts/04_clustering_examples.py
python scripts/05_model_evaluation.py
```

Each script is self-contained and generates visualization outputs (PNG files).

## Code Architecture

### Repository Structure
- **Chapter Files (chapter1-8_*.md)**: Tutorial content covering ML concepts progressively
- **scripts/**: Standalone Python scripts demonstrating key concepts
  - Each script follows the pattern: load data → preprocess → train → evaluate → visualize
  - Scripts generate PNG visualizations saved to the current directory
  - All scripts use Japanese comments and output messages

### Code Conventions
- **Language**: All comments, docstrings, and output messages are in Japanese
- **Style**: Clean, educational code following scikit-learn best practices
- **Imports**: Standard ML stack (numpy, pandas, matplotlib, seaborn, sklearn)
- **Structure**: Each script has a `main()` function with clear sections:
  1. データの読み込み (Data loading)
  2. データの分割 (Data splitting)
  3. データの標準化 (Data standardization)
  4. モデルの学習 (Model training)
  5. 予測 (Prediction)
  6. 評価 (Evaluation)
  7. 可視化 (Visualization)

### Missing Infrastructure
Note: This repository currently lacks:
- No requirements.txt or dependency management
- No test suite or testing framework
- No linting or type checking configuration
- Directories mentioned in README but not present: `exercises/`, `solutions/`, `projects/`

When implementing new features or scripts, follow the existing patterns in the `scripts/` directory and maintain consistency with Japanese language usage throughout.