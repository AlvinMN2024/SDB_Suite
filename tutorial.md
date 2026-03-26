## 📘 Tutorial: Satellite-Derived Bathymetry (SDB) Workflow

### From ACOLITE Raw Imagery to QGIS Depth Maps

This tutorial documents the exact parameters used for the **2026 Hydrographic Inventory** of the Ilian Coast, Philippines.

### Phase 1: Atmospheric Correction (ACOLITE)

We use ACOLITE (Python version 20251013.0) to convert Top-of-Atmosphere (TOA) reflectance into Surface Reflectance (rhow).

#### 1. Input & Spatial Clipping

- **Input:** Sentinel-2 L1C `.SAFE` bundle.
- **Polygon Masking:** Use a `.shp` file (e.g., `SeaMaskSmall.shp`) to restrict processing to the coastal zone. This significantly speeds up processing and focuses the DSF algorithm on the water.

#### 2. The "Ilian" Settings (DSF Method)

Navigate to the ACOLITE GUI or edit your settings file with these key parameters:

| Parameter                       | Value                            | Logic                                                      |
| ------------------------------- | -------------------------------- | ---------------------------------------------------------- |
| `atmospheric_correction_method` | `dark_spectrum`                  | Best for complex coastal waters.                           |
| `l2w_parameters`                | `rhow_*, tur_nechad, spm_nechad` | Exports reflectance + turbidity for the RF model.          |
| `l2w_mask_threshold`            | `0.25`                           | Prevents over-masking in very shallow, bright sands.       |
| `s2_target_res`                 | `10`                             | Ensures all bands are at 10m resolution for the SDB Suite. |
| `dsf_aot_estimate`              | `tiled`                          | Allows for aerosol variation across the coast.             |

Export to Sheets

#### 3. Execution

Run the process. ACOLITE will automatically fetch **GMAO MERRA2** ancillary data to correct for:

- **Water Vapor:** (e.g., 4.66 in Ilian)
- **Pressure:** (e.g., 1004.97 hPa)
- **Ozone:** (e.g., 0.25)

**Output:** You are looking for the file ending in `_L2W.nc` or the exported GeoTIFFs.

------

### Phase 2: QGIS SDB Production Suite

Once you have your `_L2W` GeoTIFF, move into QGIS.

#### 1. Model Training

1. Open **1. Train Bathymetry Model (RF)**.
2. Input your CSV containing survey depths (MLLW) and the ACOLITE `rhow` values.
3. Set your depth range (e.g., `-20` to `-1`).
4. Run to generate your `.pkl` model file.

#### 2. Bathymetry Generation

1. Open **2. Generate Bathymetry Map**.
2. Select your ACOLITE TIF and the `.pkl` model.
3. **Crucial:** Apply the **Bias Correction of -0.14m** (found during the Ilian validation) to align the satellite data with the local tide gauge datum.
4. Set **Stretch Factor** to `1.0` (unless you find a systematic depth compression).

------

### 💡 "Gotchas" & Pro-Tips

- **The Log Check:** Always check your `acolite_log`. If you see "Scene too small for tiling," ACOLITE will switch to `fixed` processing—this is fine for small survey areas like Ilian.
- **Negative Reflectance:** We set `l2w_mask_negative_rhos=False`. In very clear water, ACOLITE might slightly over-correct. Allowing negatives ensures the Random Forest model can still see the trend in the deepest pixels.
- **Feature Indices:** Ensure your Band Indices in the QGIS tool match your ACOLITE export order (usually Band 1=443, Band 2=492, etc.).