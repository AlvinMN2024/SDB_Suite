# SDB Production Suite (v1.0.0)

The **SDB Production Suite** is a specialized QGIS Processing Provider developed for the **2026 Hydrographic Inventory**. It leverages machine learning (Random Forest) and Sentinel-2 MSI data to generate high-resolution Satellite-Derived Bathymetry (SDB) for coastal monitoring and maritime safety.

## 🌊 Provenance: The Ilian Coast Project

This suite was field-tested during the 2026 survey of the **Ilian Coast, Philippines**. By utilizing ACOLITE-processed surface reflectance and a localized **-0.14m bias correction**, the model achieved an R2>0.85 in depths up to 20 meters.

## 🛠 Features

- **Machine Learning Integration:** Uses Scikit-Learn's `Random Forest Regressor` for non-linear depth estimation.
- **ACOLITE Optimized:** Built to ingest `rhow` or `Rrs` bands directly from ACOLITE atmospheric correction outputs.
- **Automated Feature Selection:** Detects and utilizes Blue, Green, Red, and Nechad (Turbidity/SPM) bands.
- **Flexible Calibration:** User-definable depth stretching and vertical bias correction.

## 🚀 Getting Started

### 1. Prerequisites

- **QGIS 4.x** (Qt6)
- Python dependencies: `pandas`, `numpy`, `scikit-learn`, `joblib`, `rasterio`.

### 2. Workflow

1. **Data Prep:** Process your Sentinel-2 image using **ACOLITE** (Dark Spectrum Fitting recommended).
2. **Training:** Use the `1. Train Bathymetry Model` tool with a CSV containing your survey points (MLLW) and corresponding spectral values.
3. **Generation:** Run the `2. Generate Bathymetry Map` tool. Input your multi-band TIF and the `.pkl` model generated in step 2.
4. **Refinement:** Apply a **Bias Correction** (e.g., `-0.14`) to align the output with local chart datum.

## ❓ FAQ

**Q: Why use Random Forest instead of the Stumpf Ratio?** A: Random Forest better accounts for seafloor variability (seagrass vs. sand) by utilizing the full spectral signature rather than just a blue/green ratio.

**Q: What is the optimal depth range?** A: The suite is optimized for the **0–20m** range. Accuracy typically degrades beyond the photic zone.

## 👨‍💻 Author

**Alvin M. Natividad** 

*OSGeo Charter Member #280* 

------