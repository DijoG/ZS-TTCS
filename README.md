# ZS-TTCS ~ ZeroShot-Topological Tree Crown Segmentor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

A **training-free** framework for individual tree crown segmentation from high-resolution imagery (10-20cm). Combines semantic priors (SegFormer) with topological segmentation (Forman gradient/Discrete Morse theory).

## ✨ Features

- 🚀 **Zero-shot** – No training data or annotations needed
- ⚡ **Fast C++ backend** – Processes large drone surveys efficiently
- 🌲 **Accurate crown delineation** – Topologically guaranteed separation of touching crowns
- 📦 **Batch processing** – Process entire directories automatically
- 🗺️ **Geospatial outputs** – GeoJSON, Shapefile, GeoTIFF, CSV
- 🐍 **Python API + CLI** – Easy to use for both developers and practitioners

## 📦 Installation

### With conda (recommended)
```bash
conda create -n zs-ttcs python=3.10 -y
conda activate zs-ttcs
pip install git+https://github.com/DijoG/ZS-TTCS.git
```

### With pip (Python 3.10 only)
```bash
python -m venv zs-ttcs
zs-ttcs\Scripts\activate  # Windows
# source zs-ttcs/bin/activate  # Mac/Linux
pip install git+https://github.com/DijoG/ZS-TTCS.git
```
## 🚀 Quick Start

### Command line (batch processing)
```bash
# Process a single image
zs-ttcs all drone_image.tif --output ./results --viz

# Process entire directory
zs-ttcs all ./drone_survey/ --batch --output ./results --viz
```
### Python API
```python
from zs_ttcs import complete_pipeline

# Run full pipeline
results = complete_pipeline("drone_image.tif", output_dir="./results")

# Access results as GeoDataFrame
crowns = results['crowns']
print(f"Found {len(crowns)} tree crowns")
```
### Step-by-step
```bash
# Step 1: Extract canopy mask
zs-ttcs-step1 image.tif --output ./masks --viz

# Step 2: Compute vegetation index
zs-ttcs-step2 image.tif --mask ./masks/image_mask.npy --output ./indices --viz

# Step 3: Segment crowns
zs-ttcs-step3 ./indices/image_masked_index.tif --output ./crowns --viz --format geojson
```
## 📊 Parameters
| Parameter | Default | Description |
|:---|:---|:---|
| `--persistence` | 0.05 | Critical point filtering (lower = more sensitive) |
| `--min-size` | 50 | Minimum crown size in pixels |
| `--max-size` | 5000 | Maximum crown size in pixels |
| `--index` | exg | Vegetation index (exg, ndvi, vari, gli) |
| `--smooth` | 1.0 | Gaussian smoothing sigma |

## 📁 Output Structure
```text
results/
├── step1_masks/           # Binary canopy masks (.npy, .tif)
├── step2_indices/         # Masked vegetation indices (.tif, .npy)
└── step3_crowns/          # Final crowns (.geojson, .shp, .csv, .tif)
``` 
## 🌲 Example Results

Paste here: Input (RGB) → Canopy Mask → Crown Segmentation

## 📚 Dependencies
-`Python 3.10`
-`PyTorch, Transformers (SegFormer)`
-`NumPy, SciPy, scikit-image`
-`Rasterio, GeoPandas, Shapely`
-`pybind11 (C++ bindings)`

## 📄 License

MIT License - see LICENSE file

## 📧 Contact

Gergo Dioszegi - dijogergo@gmail.com