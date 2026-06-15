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

### Complete pipeline (one command)
```bash
# Process a single image
zs-ttcs all drone_image.tif --output ./results 

# Process entire directory
zs-ttcs all ./drone_survey/ --batch --output ./results 
```

### Step-by-step (for parameter tuning)
```bash
# Step 1: Extract canopy mask
zs-ttcs-step1 image.tif --output ./masks --viz

# Step 2: Compute vegetation index
zs-ttcs-step2 image.tif --mask ./masks/image_mask.npy --output ./indices --viz

# Step 3: Segment crowns
zs-ttcs-step3 ./indices/image_masked_index.tif --output ./crowns --viz
```

### Python API
```python
from zs_ttcs import complete_pipeline

results = complete_pipeline("drone_image.tif", output_dir="./results")
crowns = results['crowns']  # GeoDataFrame
print(f"Found {len(crowns)} tree crowns")
```
## 🎛️ Command Reference

### Step 1: SegFormer (semantic mask)
```bash
zs-ttcs-step1 input.tif --output ./masks --threshold 0.5 --model b5 --viz
```
| Parameter | Default | Description |
|:---|:---|:---|
| input | (required) | Image file or directory |
| --output, -o | ./output | Output directory |
| --threshold | 0.5 | Confidence threshold (0-1, lower = more canopy) |
| --model | b5 | Model size: b0 (fast) to b5 (accurate) |
| --viz | False | Save visualization image |
| --batch | False | Process entire directory |

### Step 2: Vegetation index
```bash
zs-ttcs-step2 input.tif --mask mask.npy --index exg --smooth 1.0 --viz
```
| Parameter | Default | Description |
|:---|:---|:---|
| input | (required) | Image file or directory |
| --mask, -m | (required) | Mask file from Step 1 |
| --output, -o | ./step2_output | Output directory |
| --index | exg | Vegetation index: exg, ndvi, vari, gli |
| --smooth | 1.0 | Gaussian smoothing sigma (0 = no smoothing) |
| --viz | False | Save visualization image |
| --batch | False | Process entire directory |

### Step 3: Forman gradient (crown segmentation)
```bash
zs-ttcs-step3 masked_index.tif --persistence 0.05 --min-size 50 --max-size 5000 --format geojson --viz
```
| Parameter | Default | Description |
|:---|:---|:---|
| input | (required) | Masked index file from Step 2 |
| --output, -o | ./step3_output | Output directory |
| --persistence | 0.05 | Critical point filter (lower = more crowns) |
| --min-size | 50 | Minimum crown size in pixels |
| --max-size | 5000 | Maximum crown size in pixels |
| --smooth | 0.0 | Additional Gaussian smoothing |
| --format | geojson | Output: geojson, shapefile, both |
| --viz | False | Save visualization image |
| --batch | False | Process entire directory |

### Common options (all steps)
| Option | Description |
|:---|:---|
| `--batch` | Process all images in input directory |
| `--resume` | Skip already processed files |
| `--max-images` | Limit number of images to process |

## 📁 Output Structure
```text
results/
├── step1_masks/
│   ├── image_mask.npy             # Binary canopy mask (for Step 2)
│   └── image_viz.png              # Visual verification
├── step2_indices/
│   ├── image_masked_index.tif     # Masked vegetation index (for Step 3)
│   ├── image_masked_index.npy     # Fast-load version
│   └── image_step2_viz.png        # Visual verification
└── step3_crowns/
    ├── image_crowns.gpkg          # ← SINGLE FILE! All crown polygons + attributes
    ├── image_segmentation.tif     # Raster with crown IDs
    ├── image_crown_properties.csv # Crown statistics (area, intensity, etc.)
    └── image_step3_viz.png        # Visual verification
``` 

## 🌲 Example Results

...

## 🔧 Troubleshooting

### ImportError: cannot import name 'SegFormerImageProcessor'
**Solution:** Use the correct class names (lowercase 'f'):
```python
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
```
### C++ module not found (_forman)
```bash
pip install -e . --force-reinstall
```
### Rust compilation error when installing transformers
Solution: Use Python 3.10 with conda:
```bash
conda create -n zs-ttcs python=3.10
conda activate zs-ttcs
pip install transformers==4.28.1
```
### Out of memory error
Solution: Process smaller tiles or use Python fallback:
```bash
zs-ttcs-step3 input.tif --no-cpp
```
### No crowns detected
Solution: Adjust persistence threshold and min-size:
```bash
zs-ttcs-step3 input.tif --persistence 0.03 --min-size 30
```
### Batch processing fails
Solution: Ensure masks have matching filenames:
```text
images/image1.tif  →  masks/image1_mask.npy
images/image2.tif  →  masks/image2_mask.npy
``` 

## 📚 Dependencies
    -Python 3.10
    -PyTorch, Transformers (SegFormer)
    -NumPy, SciPy, scikit-image
    -Rasterio, GeoPandas, Shapely
    -pybind11 (C++ bindings)

## 📄 License

MIT License - see LICENSE file
