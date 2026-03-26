# 🌊 SDB Production Suite (v1.0.0)

**Advanced Satellite-Derived Bathymetry for QGIS**

The **SDB Production Suite** is a specialized Processing Provider developed for the **2026 Hydrographic Inventory**. It leverages Random Forest Machine Learning and Sentinel-2 MSI data to generate high-resolution bathymetric products for coastal monitoring and maritime safety.

------

## 📍 Provenance: The Ilian Coast Project

This suite was field-tested and validated during the 2026 survey of the **Ilian Coast, Philippines**.

- **Validation:** Achieved R2>0.85 in depths up to 20 meters.
- **Methodology:** Integrated ACOLITE-processed surface reflectance with a localized **-0.14m bias correction**.

------

## 🚀 Installation & Compatibility

### 💻 Supported Versions

- **QGIS:** Compatible with **QGIS 3.28 LTR and above** (including 3.34+ and 4.x).
- **OS:** Linux (Ubuntu/Debian) and Windows (OSGeo4W).

### 🛠 Manual Setup

1. **Download** this repository as a ZIP and extract the `SDB_Suite` folder.
2. **Locate your Plugins folder**:
   - **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   - **Windows**: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
3. **Move** the `SDB_Suite` folder there and **Restart QGIS**.
4. **Enable** the plugin via *Plugins > Manage and Install Plugins*.

------

## 📦 System Requirements

To prevent "Module Not Found" errors, ensure these dependencies are installed in your QGIS Python environment:

| Dependency       | Purpose               | Installation (Terminal/OSGeo4W) |
| ---------------- | --------------------- | ------------------------------- |
| **scikit-learn** | Machine Learning Core | `pip install scikit-learn`      |
| **rasterio**     | Geospatial I/O        | `pip install rasterio`          |
| **pandas**       | Data Handling         | `pip install pandas`            |
| **joblib**       | Model Saving/Loading  | `pip install joblib`            |

Export to Sheets

> **Linux Users:** You can also use `sudo apt install python3-sklearn python3-rasterio python3-pandas python3-joblib`.

------

## 🛠 Core Features

- **Machine Learning Integration:** Uses Scikit-Learn's `Random Forest Regressor` for robust, non-linear depth estimation.
- **ACOLITE Optimized:** Directly ingests `rhow` or `Rrs` bands from ACOLITE atmospheric correction.
- **Automated Feature Selection:** Automatically detects Blue, Green, Red, and Nechad (Turbidity/SPM) bands.
- **Flexible Calibration:** User-definable depth stretching and vertical bias correction tools.

------

## 📖 The Workflow

1. **Atmospheric Correction:** Process Sentinel-2 imagery in **ACOLITE** (Dark Spectrum Fitting).
2. **Model Training:** Run `1. Train Bathymetry Model` using a CSV of survey points (MLLW) and spectral values.
3. **Map Generation:** Run `2. Generate Bathymetry Map` using your multi-band TIF and the `.pkl` model.
4. **Refinement:** Apply a **Bias Correction** (e.g., `-0.14m`) to align with your local chart datum.

------

## ❓ FAQ

**Q: Why Random Forest instead of the Stumpf Ratio?** A: RF accounts for seafloor variability (seagrass vs. sand) by analyzing the full spectral signature rather than just a two-band ratio.

**Q: What is the optimal depth range?** A: Optimized for **0–20m**. Accuracy typically degrades as light attenuation increases beyond the photic zone.

------

## 👨‍💻 Author

**Alvin M. Natividad** *OSGeo Charter Member #280*

------

------