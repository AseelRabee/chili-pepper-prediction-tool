# 🌶️ Chili Pepper Capsaicinoid Prediction Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flutter](https://img.shields.io/badge/Flutter-3.0+-blue?logo=flutter)](https://flutter.dev)

**A validated, interpretable five-variable predictive equation for capsaicinoid content in pepper hybrids, outperforming complex genomic models with 99.8% fewer inputs.**

---

## 📖 Overview

This repository contains the complete source code and materials for the research paper:

> **"An interpretable five-variable predictive equation for capsaicinoid content in pepper hybrids: A validated decision-support tool outperforming complex genomic models"**

Our model reduces predictors from **3,194 SNP markers** to just **five biologically interpretable variables**, achieving **R² = 0.5966** on the test set—slightly surpassing the original GBLUP-GAUSS model (R² ≈ 0.59) while being **static, interpretable, and deployable** without retraining.

---

## 🚀 Key Features

- ✅ **Static Equation** – No retraining required for new parental combinations
- ✅ **Biologically Interpretable** – Explicit coefficients with clear genetic meaning
- ✅ **Cross-Platform App** – Built with Flutter (Android + Web)
- ✅ **Automated Computation** – Calculates WGD, ND, ME, PE, and Pavg from raw inputs
- ✅ **Offline Capability** – Works in remote field locations without internet
- ✅ **Batch Processing** – Upload CSV files for multiple crosses at once
- ✅ **95% Confidence Intervals** – For all predictions (derived from bootstrap analysis)

---

## 📊 The Five Biological Variables

| Variable | Definition | Biological Meaning |
| :--- | :--- | :--- |
| **WGD** | Weighted Genetic Distance | Normalized Euclidean distance between parental SNP vectors |
| **ND** | Number of Differences | Count of SNP loci where parents carry different alleles |
| **ME** | Maternal Effect | Average rMPH produced by the female parent (GCA) |
| **PE** | Paternal Effect | Average rMPH produced by the male parent (GCA) |
| **Pavg** | Parental Average | Mean TCAPgDW BLUP value of both parents |

---

## 📐 The Predictive Equation

```txt
rMPH = 45.4260 - 82.1698·WGD_scaled + 82.3899·ND_scaled + 22.0811·ME_scaled + 22.4838·PE_scaled - 2.6207·Pavg_scaled

TCAP_pred = Pavg × (1 + rMPH / 100)
```

## 📁 Repository Structure
```txt
chili-pepper-prediction-tool/
├── analysis/                          # Python analysis scripts
│   ├── full_analysis_script.py        # Complete statistical analysis
│   └── requirements.txt               # Python dependencies
├── app/                               # Flutter decision-support app
│   ├── lib/                           # Dart source code
│   └── pubspec.yaml                   # Flutter dependencies
├── data/                              # Sample datasets
│   ├── training_data.csv              # Training set (116 hybrids)
│   ├── test_data.csv                  # Test set (32 hybrids)
│   └── outliers_excluded.csv          # Filtered extreme hybrids
└── figures/                           # Publication-ready figures
    ├── Figure_Actual_vs_Predicted.png
    ├── Figure_Effect_Size_Ranking.png
    ├── Figure_Correlation_Heatmap.png
    ├── Figure_Cumulative_Accuracy.png
    ├── Figure_Bootstrap_R2.png
    ├── Figure_Radar_Comparison.png
    ├── Figure_Sensitivity_Analysis.png
    └── residual_diagnostics.png
```

## 🔧 Quick Start
1. Run the Statistical Analysis (Python)
Clone the repository and install dependencies:

```txt
git clone https://github.com/AseelRabee/chili-pepper-prediction-tool.git
cd chili-pepper-prediction-tool/analysis
pip install -r requirements.txt
python full_analysis_script.py
Input files required: BLUP_value.csv, gt.score.csv, combi.csv (place them in the same directory).
```

This will generate:
All tables and figures from the paper
Trained Ridge regression model (ridge_model_final.pkl)
StandardScaler (standard_scaler_final.pkl)
Prediction results and datasets

2. Build the Flutter App
```txt
cd ../app
flutter pub get
flutter run -d chrome      # Web version
flutter run -d android     # Android version
```

## 📈 Performance Summary
Metric	Value
R² (rMPH)	0.3467
R² (TCAP)	0.5966
RMSE (TCAP)	4940.53 µg/g
MAE (TCAP)	3882.61 µg/g
95% CI for R²	[0.4956, 0.6142]
Accuracy within 15% error	65.6%

Benchmarking Results
Model	R² (TCAP)	Retraining Required	Interpretability
Our Ridge (Static)	0.5966	❌ No	✅ High (Explicit)
Lasso	0.5965	✅ Yes	Moderate
Linear Regression	0.5954	✅ Yes	High
XGBoost	0.5039	✅ Yes	Low
Random Forest	0.5349	✅ Yes	Low
SVR	0.3648	✅ Yes	Low

🧬 Key Biological Insights
Paternal Effect (PE) and Maternal Effect (ME) have practically identical coefficients (22.48 vs. 22.08), confirming that both parents contribute equally to heterosis when controlling for their combining ability.
ND and WGD are the strongest drivers of heterosis (coefficients of +82.39 and -82.17, respectively), highlighting the critical role of optimal genetic divergence.
The apparent dominance of PE in ablation analysis reflects higher variance among fathers in this specific population, not superior biological strength.

📝 Citation
If you use this code or model in your research, please cite:

bibtex
@article{e,
  title={An interpretable five-variable predictive equation for capsaicinoid content in pepper hybrids},
  author={},
  journal={},
  year={},
  doi={...}
}

📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

📧 Contact
For questions or collaborations, please open an issue or contact the corresponding author.

