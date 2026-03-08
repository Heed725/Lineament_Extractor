# PCI LINE-Style Lineament Extraction from DEM

Automated geological lineament extraction from Digital Elevation Model (DEM) raster data, producing georeferenced shapefiles. Designed to replicate the output style of PCI Geomatica's LINE module using open-source Python libraries.

No training data, no GPU, no commercial software required.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Parameter Tuning Guide](#parameter-tuning-guide)
- [Controlling Output Count](#controlling-output-count)
- [Spatial Balance](#spatial-balance)
- [Troubleshooting](#troubleshooting)
- [Comparison with PCI LINE](#comparison-with-pci-line)
- [License](#license)

---

## Overview

This script extracts geological lineaments (faults, fractures, joints, lithological contacts) from a DEM and exports them as a shapefile with attributes including length, azimuth, sinuosity, importance score, and rank.

It was developed for the Dodoma region (Tanzania) but works with any single-band GeoTIFF DEM (SRTM, ASTER, LiDAR, national DEMs, etc.).

---

## How It Works

The extraction pipeline has 9 steps:

1. **Load DEM** — reads the GeoTIFF, handles nodata, smooths with Gaussian filter.

2. **Gradient computation** — computes gradient magnitude and direction (equivalent to PCI's RADI parameter), then applies non-maximum suppression to thin edges to single-pixel width.

3. **Ridge and valley detection** — uses eigenvalues of the Hessian matrix to identify ridges and valleys. This is closer to PCI LINE's internal algorithm than simple edge detection alone.

4. **Multi-azimuth hillshade edge detection** — generates hillshade from 8 sun directions (0°, 45°, 90°, ... 315°) and runs Canny edge detection on each. This ensures lineaments of all orientations are captured regardless of illumination angle.

5. **Edge combination and filtering** — merges all edge sources (gradient, Hessian, hillshade), removes low-density noise using a local density filter, and skeletonizes to single-pixel-wide lines.

6. **Multi-scale line extraction** — runs Probabilistic Hough Transform at 3 scales (fine, medium, regional) to capture both small local features and larger regional structures. Deduplicates overlapping segments.

7. **Segment processing** — removes isolated noise segments, extends endpoints slightly, merges nearby collinear segments (controlled by ATHR and DTHR parameters), and splits any overly long segments.

8. **Importance scoring** — ranks every lineament on a 0–100 scale based on four weighted factors: length (35%), gradient strength (25%), straightness (20%), and local lineament density (20%).

9. **Spatially-balanced export** — if a maximum count is set, divides the map into a grid and selects the top lineaments from each cell, ensuring coverage across the entire DEM rather than clustering in one area.

---

## Requirements

- Python 3.6 or higher
- No GPU needed — runs entirely on CPU
- Sufficient RAM for your DEM size (a 5000×5000 pixel DEM uses roughly 1–2 GB)

### Python Libraries

| Library | Purpose |
|---------|---------|
| numpy | Array operations |
| scipy | Gaussian filtering, spatial indexing (cKDTree) |
| matplotlib | Preview plots and rose diagram |
| scikit-image | Canny edge detection, Hough transform, skeletonization |
| rasterio | Reading GeoTIFF, coordinate transforms |
| geopandas | Building and saving shapefiles |
| shapely | Line geometry |
| fiona | Shapefile I/O backend for geopandas |

---

## Installation

```bash
pip install numpy scipy matplotlib scikit-image rasterio geopandas shapely fiona
```

On Windows, if rasterio or fiona fail to install, use pre-built wheels:

```bash
pip install pipwin
pipwin install rasterio
pipwin install fiona
```

Or use conda:

```bash
conda install -c conda-forge numpy scipy matplotlib scikit-image rasterio geopandas shapely fiona
```

---

## Quick Start

1. Place your DEM file (e.g., `Dodoma_DEM.tif`) in a known folder.

2. Open `lineament_pci_style.py` and edit these two lines:

```python
DEM_PATH = r"C:\Users\user\Documents\Dodoma_DEM.tif"
OUTPUT_DIR = r"C:\Users\user\Documents"
```

3. Run:

```bash
python lineament_pci_style.py
```

4. Open `lineaments.shp` in QGIS or ArcGIS.

---

## Configuration

All parameters are at the top of the script. Here is what each one does:

### Paths

| Parameter | Description |
|-----------|-------------|
| `DEM_PATH` | Full path to your input DEM (.tif) |
| `OUTPUT_DIR` | Folder where output files will be saved |
| `OUTPUT_NAME` | Base name for output files (default: "lineaments") |

### Output Control

| Parameter | Description | Default |
|-----------|-------------|---------|
| `MAX_LINEAMENTS` | How many lineaments to export. 0 = all. | 0 |
| `MAX_LINE_LENGTH` | Maximum segment length in pixels. Longer lines get split. | 120 |

### PCI LINE Equivalent Parameters

| Parameter | PCI Equivalent | Description | Default |
|-----------|---------------|-------------|---------|
| `RADI` | RADI | Filter radius for gradient computation | 10 |
| `GTHR` | GTHR | Gradient threshold (0–255). Lower = more edges. | 50 |
| `LTHR` | LTHR | Minimum line length in pixels | 15 |
| `FTHR` | FTHR | Line fitting error tolerance | 3 |
| `ATHR` | ATHR | Angular tolerance for merging collinear segments (degrees) | 8 |
| `DTHR` | DTHR | Maximum distance for linking segments (pixels) | 10 |

### Multi-Scale Detection

The `SCALES` list defines 3 detection passes:

```python
SCALES = [
    {"sigma": 1.0, "min_length": 15, "line_gap": 2},   # Fine detail
    {"sigma": 2.0, "min_length": 20, "line_gap": 3},   # Medium features
    {"sigma": 3.0, "min_length": 25, "line_gap": 4},   # Regional structures
]
```

### Importance Weights

Control what "important" means. Must sum to 1.0:

| Weight | What it favors | Default |
|--------|---------------|---------|
| `WEIGHT_LENGTH` | Longer lineaments | 0.35 |
| `WEIGHT_GRADIENT` | Stronger terrain gradient along the line | 0.25 |
| `WEIGHT_STRAIGHTNESS` | Straighter (lower sinuosity) lineaments | 0.20 |
| `WEIGHT_DENSITY` | Lines in areas with many other lineaments | 0.20 |

---

## Output Files

| File | Description |
|------|-------------|
| `lineaments.shp` | Georeferenced shapefile with lineament polylines |
| `lineaments.shx` | Shape index file |
| `lineaments.dbf` | Attribute table |
| `lineaments.prj` | Projection/CRS information |
| `lineaments.cpg` | Character encoding |
| `lineaments_preview.png` | Full extent map: red lines on gray hillshade |
| `lineaments_zoom.png` | Zoomed detail of center region |
| `lineaments_rose.png` | Rose diagram of lineament orientations |

### Shapefile Attributes

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Sequential ID |
| `rank` | Integer | Rank by importance (1 = most important) |
| `length` | Float | Line length in CRS units |
| `azimuth` | Float | Orientation 0–180° (0° = North, 90° = East) |
| `sinuosity` | Float | Ratio of line length to straight distance (1.0 = perfectly straight) |
| `importance` | Float | Importance score 0–100 |

---

## Parameter Tuning Guide

### Too few lineaments detected

- Lower `GTHR` (e.g., 30 instead of 50)
- Lower `CANNY_LOW` and `CANNY_HIGH`
- Lower `LTHR` (e.g., 10)
- Add a coarser scale to `SCALES` (e.g., `{"sigma": 5.0, "min_length": 30, "line_gap": 8}`)

### Too many lineaments (noisy)

- Raise `GTHR` (e.g., 80 or 100)
- Raise `LTHR` (e.g., 30 or 40)
- Raise `MIN_LINE_LENGTH` values in `SCALES`

### Lines are too long

- Lower `MAX_LINE_LENGTH` (e.g., 80)
- Lower `DTHR` (e.g., 5) to reduce merging distance
- Lower `ATHR` (e.g., 5) to make merging stricter

### Lines are too short / fragmented

- Raise `DTHR` (e.g., 30–50) for more aggressive merging
- Raise `ATHR` (e.g., 15–20)
- Raise `MAX_LINE_LENGTH` (e.g., 200–300)

### Lines cluster in one area when using MAX_LINEAMENTS

- This should not happen with the spatially-balanced selection. If it does, the grid is working correctly — some areas may genuinely have no lineaments.

### DEM-specific tips

- **SRTM 30m**: RADI=10, GTHR=50, LTHR=15 (default settings work well)
- **SRTM 90m**: RADI=5, GTHR=40, LTHR=10 (coarser DEM needs lower thresholds)
- **LiDAR 1–5m**: RADI=15, GTHR=80, LTHR=30 (high detail needs higher thresholds to avoid noise)
- **ASTER 30m**: Similar to SRTM, may need slightly lower GTHR due to more noise

---

## Controlling Output Count

Set `MAX_LINEAMENTS` at the top of the script:

```python
MAX_LINEAMENTS = 0       # Export ALL lineaments (no limit)
MAX_LINEAMENTS = 1000    # Export only the top 1000
MAX_LINEAMENTS = 500     # Export only the top 500
MAX_LINEAMENTS = 5000    # Export only the top 5000
```

When a limit is applied, lineaments are selected by importance score, and the selection is spatially balanced across the entire DEM.

---

## Spatial Balance

When `MAX_LINEAMENTS` is set, the script does **not** simply take the top N highest-scoring lineaments (which would cluster in one area). Instead:

1. The DEM extent is divided into an auto-sized grid of cells.
2. Each occupied cell receives an equal quota of lineaments.
3. Within each cell, the highest-importance lineaments are selected.
4. Any remaining quota is distributed to cells with leftover lineaments.

This ensures coverage across the entire study area while still prioritizing the most geologically significant features in each sub-region.

---

## Troubleshooting

**"No lineaments found"**
- Your GTHR is too high. Lower it (e.g., GTHR=30).
- Your DEM may be very flat. Lower CANNY_LOW and CANNY_HIGH.

**Script runs out of memory**
- Your DEM is very large. Consider clipping it to a smaller area first.
- Reduce the number of SCALES or AZIMUTHS.

**Shapefile has no CRS / .prj file is empty**
- Your input DEM may not have a CRS defined. Open it in QGIS, assign a CRS, re-save, and re-run.

**Lines don't align with features in QGIS/ArcGIS**
- Ensure the DEM and your basemap use the same CRS.
- The shapefile inherits the CRS from the input DEM.

**Very slow on large DEMs**
- The non-maximum suppression step is the bottleneck (pixel-by-pixel loop). For DEMs larger than ~10,000 x 10,000 pixels, consider downsampling first or clipping to sub-areas.

---

## Comparison with PCI LINE

| Feature | PCI LINE | This Script |
|---------|----------|-------------|
| Edge detection | Proprietary algorithm | Gradient + Hessian + multi-azimuth Canny |
| Ridge/valley detection | Built-in | Hessian eigenvalue method |
| Line extraction | Curve following | Probabilistic Hough Transform |
| Multi-scale | Single pass | 3-scale extraction |
| Segment merging | ATHR + DTHR params | Collinear merging with same params |
| Output format | .pix vector layer | Shapefile (.shp) |
| Importance ranking | Not available | 0–100 score with 4 factors |
| Spatial balancing | Not available | Grid-based balanced selection |
| Requires | PCI Geomatica license | Free (open-source Python) |
| GPU needed | No | No |

---

## License

This script is provided as-is for research and educational use. The author makes no warranty regarding geological accuracy. Always validate extracted lineaments against field data and published geological maps.
