# SDB Production Suite (QGIS Plugin ID 4946)

**Satellite-Derived Bathymetry (SDB) Generation using Machine Learning & Multispectral Optics**

## 1. Overview & Hydrographic Scope

The **SDB Production Suite** is an open-source QGIS plugin designed for hydrographic mapping, coastal zone management, and shallow-water bathymetric modeling. By integrating ACOLITE-processed Sentinel-2 multispectral reflectance imagery with surveyed ground-truth control points (e.g., single-beam sonar, ICESat-2 ATL13, or leadline soundings), the suite generates high-resolution, Mean Lower Low Water (MLLW)-referenced bathymetric surface grids.

The suite implements a two-stage processing architecture powered by `scikit-learn` Random Forest Regression and `rasterio` chunked spatial predictors, ensuring low memory consumption and high execution stability across large Sentinel-2 tiles (10,000 x 10,000 pixels).

## 2. Key Features

- **Non-Blocking Processing:** Native integration with the QGIS Processing Framework allows background execution without locking the map canvas UI.
- **Low Memory Footprint:** Windowed raster reading/writing (`rasterio.block_windows`) ensures full $10\text{ m}$ satellite scenes are processed safely under 100 MB RAM allocation.
- **Hydrographic Error Metrics:** Out-of-Bag (OOB) cross-validation reports $R^2$, Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and mean vertical bias during model training.
- **IHO S-44 Contextual Reporting:** Automated precision classification provides immediate survey suitability context aligned with International Hydrographic Organization (IHO) Standards for Hydrographic Surveys.
- **Flexible Column Mapping:** Automatic detection of depth fields (`MLLW`, `depth`, or `Z`) supporting both negative elevation and positive bathymetric depth conventions.

## 3. Installation & Prerequisites

### System Requirements

- **QGIS Version:** $3.44 \le \text{QGIS Version} \le 4.5$
- **Operating System:** Windows 10/11, macOS, or Linux

### Python Dependencies

The plugin requires standard scientific Python libraries available within your QGIS Python environment.

To install dependencies via the QGIS OSGeo4W Shell (Windows) or Terminal (Linux/macOS):

Bash

```
# Windows (OSGeo4W Shell as Administrator)
pip install scikit-learn joblib rasterio pandas

# Linux / macOS
qgis-python -m pip install scikit-learn joblib rasterio pandas
```

### Installation Steps

1. Open QGIS and navigate to **Plugins** $\rightarrow$ **Manage and Install Plugins...**
2. Search for **SDB Production Suite** (Plugin ID `4946`).
3. Click **Install Plugin**.
4. Access the tools under **SDB Suite** inside the QGIS Processing Toolbox panel.

## 4. Workflow Overview

```
[ ACOLITE Level-2R Reflectance ] + [ Surveyed Control Points CSV ]
               │                                │
               └────────────────┬───────────────┘
                                │
                                ▼
               [ 1. Train Bathymetry Model (RF) ]
                                │
                                ▼
                 [ Trained Model Payload (.pkl) ]
                                │
                                ▼
               [ 2. Generate Bathymetry Map ]
                                │
                                ▼
                [ Bathymetric GeoTIFF Grid ]
```

## 5. Software Architecture & Module Layout

Plaintext

```
sdb_production_suite/
├── metadata.txt          # Plugin definition, QGIS 3.44–4.5 bounds, version 1.0.2
├── sdb_provider.py       # QGIS Processing Provider registration & S-44 metrics helper
├── sdb_trainer_alg.py    # Algorithm 1: Random Forest Model Trainer with OOB metrics
├── sdb_generator_alg.py  # Algorithm 2: Chunked Raster Predictor & Depth Inversion
├── icon.png              # Plugin toolbox branding icon
├── LICENSE               # GNU General Public License v2.0 or later text
└── README.md             # Technical documentation & open-source compliance
```

## 6. Hydrographic Accuracy & Error Metrics

The model trainer evaluates depth predictions against measured soundings using Out-of-Bag (OOB) samples to prevent resubstitution overfitting bias.

| **Metric**        | **Calculation / Description**                                | **Hydrographic Significance**                                |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **OOB $R^2$**     | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Measures model goodness of fit and spectral depth correlation. |
| **OOB RMSE**      | $\sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$        | Quantifies absolute vertical precision across the surveyed depth band. |
| **OOB MAE**       | $\frac{1}{n} \sum_{i=1}^n \Vert{}y_i - \hat{y}_i\Vert{}$     | Assesses average magnitude of errors without over-weighting outliers. |
| **Vertical Bias** | $\frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)$                 | Identifies systemic overestimation (deep bias) or underestimation (shallow bias). |

## 7. Open Source Compliance & Human-in-the-Loop Statement

### License Terms

The **SDB Production Suite** is released under the **GNU General Public License v2.0 or later (GPL-2.0-or-later)**. In accordance with OSGeo standards and QGIS core licensing guidelines:

- The full source code is un-obfuscated, publicly accessible, and free to modify or distribute under copyleft terms.
- All mathematical transformations, ratio equations, and Random Forest routines are fully transparent, deterministic, and open for academic or commercial verification.

### Human-in-the-Loop (HITL) Development Governance

In alignment with community discussions regarding software maintainability, IP clean provenance, and core codebase integrity (e.g., QEP 428 principles):

- **Deterministic Core Execution:** The SDB Production Suite contains no generative or non-deterministic code execution loops. Depth inversion is governed strictly by statistical machine learning algorithms trained on verified hydrographic data.
- **Human Code Oversight:** Every class, function, parameter mapping, and PyQGIS API call has undergone direct human developer review, verification, and performance profiling.
- **Reproducibility:** All processing pipelines—from ACOLITE L2R input handling to feature extraction and raster writing—are deterministic and reproducible given identical satellite inputs, hyperparameter seeds, and sounding datasets.

## 8. Author & Repository Info

- **Author:** Alvin M. Natividad (OSGeo Charter Member)
- **Email:** AlvinMN2022@gmail.com
- **Repository:** https://github.com/AlvinMN2024/SDB_Suite
- **Bug Tracker:** https://github.com/AlvinMN2024/SDB_Suite/issues
- **License:** GPL-2.0-or-later