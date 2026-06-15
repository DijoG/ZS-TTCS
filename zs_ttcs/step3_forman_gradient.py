#!/usr/bin/env python3
"""
ZS-TTCS - Step 3: Forman Gradient Segmentation (BATCH CAPABLE)
================================================================
ZeroShot-Topological Tree Crown Segmentor
Step 3: Apply discrete Morse theory/Forman gradient to segment individual tree crowns

This module runs THIRD in the pipeline, taking the masked vegetation index
from Step 2 and performing topological segmentation using the Forman gradient.

BATCH PROCESSING: Can process entire directories of masked indices from Step 2.
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, List
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import json
from datetime import datetime
import warnings
from scipy.ndimage import gaussian_filter 
from skimage import measure
import geopandas as gpd
from shapely.geometry import Polygon as ShapelyPolygon
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import C++ module (compiled from cpp/forman.cpp and cpp/bindings.cpp)
try:
    from . import _forman as tsc
    CPP_AVAILABLE = True
    logger.info("✅ C++ module loaded successfully")
except ImportError:
    try:
        import zs_ttcs._forman as tsc  # Absolute import as fallback
        CPP_AVAILABLE = True
        logger.info("✅ C++ module loaded successfully (absolute import)")
    except ImportError:
        CPP_AVAILABLE = False
        warnings.warn(
            "C++ module '_forman' not found. Using Python fallback (slower).\n"
            "To enable fast C++ backend, make sure compilation succeeded:\n"
            "pip install -e ."
        )

class FormanGradientSegmentor:
    """
    Step 3: Forman gradient-based tree crown segmentation.
    
    This class implements discrete Morse theory / Forman gradient
    to segment individual tree crowns from a vegetation index.
    Uses C++ backend for performance with Python fallback.
    
    BATCH CAPABLE: Can process multiple masked indices from Step 2.
    """
    
    # Supported input extensions
    SUPPORTED_EXTENSIONS = ['.tif', '.tiff', '.npy']
    
    def __init__(
        self,
        persistence_threshold: float = 0.05,
        min_crown_size: int = 50,        # Minimum crown size in pixels
        max_crown_size: int = 5000,       # Maximum crown size in pixels
        smoothing_sigma: float = 0.0,      # Additional smoothing (0 = use Step 2 smoothing)
        use_cpp: bool = True,               # Use C++ backend if available
        n_threads: int = -1                  # Number of threads (-1 = all)
    ):
        """
        Initialize Forman gradient segmentor.
        
        Args:
            persistence_threshold: Threshold for critical point filtering
            min_crown_size: Minimum crown size in pixels
            max_crown_size: Maximum crown size in pixels
            smoothing_sigma: Additional Gaussian smoothing
            use_cpp: Use C++ backend if available
            n_threads: Number of threads for parallel processing
            output_format: Vector output format
        """
        self.persistence_threshold = persistence_threshold
        self.min_crown_size = min_crown_size
        self.max_crown_size = max_crown_size
        self.smoothing_sigma = smoothing_sigma
        self.use_cpp = use_cpp and CPP_AVAILABLE
        self.n_threads = n_threads if n_threads > 0 else None
        
        logger.info("=" * 60)
        logger.info("ZS-TTCS - Step 3: Forman Gradient Segmentation")
        logger.info("=" * 60)
        logger.info(f"Persistence threshold: {self.persistence_threshold}")
        logger.info(f"Min crown size: {self.min_crown_size} pixels")
        logger.info(f"Max crown size: {self.max_crown_size} pixels")
        logger.info(f"Using C++ backend: {self.use_cpp}")
        logger.info(f"Output format: {self.output_format}")
        logger.info("=" * 60)
    
    def find_masked_indices_in_directory(
        self, 
        directory: Union[str, Path],
        pattern: Optional[str] = None
    ) -> List[Path]:
        """
        Find all masked index files from Step 2 in a directory.
        
        Args:
            directory: Directory to search
            pattern: Optional pattern (e.g., "*_masked_index.tif")
        
        Returns:
            Sorted list of masked index paths
        """
        directory = Path(directory)
        indices = []
        
        if pattern:
            indices.extend(directory.glob(pattern))
        else:
            # Look for Step 2 output patterns
            for ext in self.SUPPORTED_EXTENSIONS:
                indices.extend(directory.glob(f"*_masked_index{ext}"))
                indices.extend(directory.glob(f"*_masked_index{ext.upper()}"))
        
        # Remove duplicates and sort
        indices = sorted(list(set(indices)))
        
        logger.info(f"Found {len(indices)} masked index files in {directory}")
        return indices
    
    def load_masked_index(
        self, 
        input_path: Union[str, Path]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Load masked vegetation index from Step 2.
        
        Args:
            input_path: Path to masked index (.tif or .npy)
        
        Returns:
            - Masked index array (H, W)
            - Metadata dictionary
        """
        input_path = Path(input_path)
        metadata = {'source': str(input_path)}
        
        if input_path.suffix == '.npy':
            # Load NumPy array
            data = np.load(input_path)
            metadata['format'] = 'numpy'
            metadata['shape'] = data.shape
            metadata['dtype'] = str(data.dtype)
            
            # Try to find corresponding GeoTIFF for metadata
            tif_path = input_path.with_suffix('.tif')
            if tif_path.exists():
                with rasterio.open(tif_path) as src:
                    metadata['crs'] = src.crs
                    metadata['transform'] = src.transform
                    metadata['bounds'] = src.bounds
                    metadata['driver'] = src.driver
            else:
                metadata['crs'] = None
                metadata['transform'] = None
        
        else:
            # Load GeoTIFF
            with rasterio.open(input_path) as src:
                data = src.read(1)
                metadata.update({
                    'format': 'geotiff',
                    'shape': data.shape,
                    'crs': src.crs,
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'driver': src.driver,
                    'nodata': src.nodata
                })
        
        # Handle nodata values
        if metadata.get('nodata') is not None:
            mask = data != metadata['nodata']
            data = data * mask
        else:
            # Assume non-zero is canopy (from Step 2 masking)
            mask = data != 0
        
        metadata['canopy_mask'] = mask
        metadata['canopy_pixels'] = int(np.sum(mask))
        metadata['canopy_percent'] = float(np.mean(mask) * 100)
        
        logger.info(f"Loaded masked index: {data.shape}, range [{data.min():.3f}, {data.max():.3f}]")
        logger.info(f"Canopy pixels: {metadata['canopy_pixels']:,} ({metadata['canopy_percent']:.1f}%)")
        
        return data, metadata
    
    def segment_with_cpp(
        self,
        data: np.ndarray,
        metadata: Dict
    ) -> Dict:
        """
        Segment using C++ Forman gradient backend.
        
        Args:
            data: Masked vegetation index
            metadata: Image metadata
        
        Returns:
            Dictionary with segmentation results
        """
        if not CPP_AVAILABLE:
            raise ImportError("C++ module not available")
        
        h, w = data.shape
        
        # Build cell complex
        logger.info("   Building 2D cell complex...")
        complex = tsc.CellComplex2D(w, h, data.flatten())
        complex.build()
        
        # Compute Forman gradient
        logger.info("   Computing Forman gradient...")
        gradient = tsc.FormanGradient(complex)
        gradient.compute_gradient()
        
        # Filter by persistence
        logger.info(f"   Filtering with persistence threshold {self.persistence_threshold}...")
        gradient.filter_by_persistence(self.persistence_threshold)
        
        # Get critical points (local maxima = tree tops)
        critical_vertices = gradient.get_critical_vertices()
        logger.info(f"   Found {len(critical_vertices)} critical points")
        
        # Extract basins (crowns)
        logger.info("   Extracting crown basins...")
        crowns = []
        crown_masks = []
        
        for v_id in critical_vertices:
            # Get basin pixels
            basin_pixels = gradient.get_basin(v_id)
            
            if len(basin_pixels) < self.min_crown_size:
                continue
            if len(basin_pixels) > self.max_crown_size:
                continue
            
            # Create mask
            mask = np.zeros((h, w), dtype=np.uint8)
            for pixel_id in basin_pixels:
                y = pixel_id // w
                x = pixel_id % w
                mask[y, x] = 1
            
            crown_masks.append(mask)
            
            # Get crown properties
            props = self._measure_crown_properties(mask, data, v_id, (y, x))
            crowns.append(props)
        
        logger.info(f"   Retained {len(crowns)} crowns after size filtering")
        
        return {
            'crowns': crowns,
            'crown_masks': crown_masks,
            'critical_points': critical_vertices,
            'num_critical': len(critical_vertices),
            'num_crowns': len(crowns)
        }
    
    def segment_with_python(
        self,
        data: np.ndarray,
        metadata: Dict
    ) -> Dict:
        """
        Fallback Python implementation (slower but works without C++).
        Uses watershed from local maxima as approximation.
        
        Args:
            data: Masked vegetation index
            metadata: Image metadata
        
        Returns:
            Dictionary with segmentation results
        """
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
        from scipy import ndimage
        
        logger.info("   Using Python fallback (watershed approximation)...")
        
        h, w = data.shape
        canopy_mask = metadata['canopy_mask']
        
        # Smooth if requested
        if self.smoothing_sigma > 0:
            data_smooth = ndimage.gaussian_filter(data, sigma=self.smoothing_sigma)
        else:
            data_smooth = data
        
        # Find local maxima (tree tops)
        # Only consider within canopy and with sufficient prominence
        coordinates = peak_local_max(
            data_smooth,
            min_distance=3,
            threshold_abs=np.percentile(data_smooth[canopy_mask], 50),
            exclude_border=False,
            mask=canopy_mask
        )
        
        logger.info(f"   Found {len(coordinates)} local maxima")
        
        if len(coordinates) == 0:
            return {
                'crowns': [],
                'crown_masks': [],
                'critical_points': [],
                'num_critical': 0,
                'num_crowns': 0
            }
        
        # Create markers for watershed
        markers = np.zeros_like(data, dtype=np.int32)
        for i, (y, x) in enumerate(coordinates, 1):
            markers[y, x] = i
        
        # Apply watershed
        segmentation = watershed(-data_smooth, markers, mask=canopy_mask)
        
        # Extract individual crowns
        crowns = []
        crown_masks = []
        
        for crown_id in range(1, np.max(segmentation) + 1):
            mask = segmentation == crown_id
            size = np.sum(mask)
            
            if size < self.min_crown_size or size > self.max_crown_size:
                continue
            
            crown_masks.append(mask.astype(np.uint8))
            
            # Get crown properties
            crown_center = np.mean(np.argwhere(mask), axis=0)
            props = self._measure_crown_properties(
                mask, data, crown_id, crown_center
            )
            crowns.append(props)
        
        logger.info(f"   Retained {len(crowns)} crowns after size filtering")
        
        return {
            'crowns': crowns,
            'crown_masks': crown_masks,
            'critical_points': coordinates,
            'num_critical': len(coordinates),
            'num_crowns': len(crowns)
        }
    
    def _measure_crown_properties(
        self,
        mask: np.ndarray,
        data: np.ndarray,
        crown_id: int,
        peak_location: Tuple[int, int]
    ) -> Dict:
        """
        Measure properties of a single crown.
        
        Args:
            mask: Binary crown mask
            data: Original vegetation index
            crown_id: Crown identifier
            peak_location: (y, x) of crown peak
        
        Returns:
            Dictionary with crown properties
        """
        y_peak, x_peak = peak_location
        crown_pixels = np.argwhere(mask > 0)
        
        # Basic properties
        area = np.sum(mask)
        mean_intensity = np.mean(data[mask > 0])
        max_intensity = data[y_peak, x_peak]
        
        # Boundary
        from skimage.measure import find_contours
        contours = find_contours(mask, 0.5)
        boundary = contours[0] if contours else None
        
        # Bounding box
        rows, cols = np.where(mask > 0)
        bbox = {
            'min_row': int(np.min(rows)),
            'max_row': int(np.max(rows)),
            'min_col': int(np.min(cols)),
            'max_col': int(np.max(cols))
        }
        
        # Centroid
        centroid_y, centroid_x = np.mean(crown_pixels, axis=0)
        
        # Diameter (approx)
        diameter = np.sqrt(area / np.pi) * 2
        
        return {
            'crown_id': int(crown_id),
            'area_pixels': int(area),
            'peak_y': int(y_peak),
            'peak_x': int(x_peak),
            'centroid_y': float(centroid_y),
            'centroid_x': float(centroid_x),
            'mean_intensity': float(mean_intensity),
            'max_intensity': float(max_intensity),
            'diameter_pixels': float(diameter),
            'boundary': boundary.tolist() if boundary is not None else None,
            'bbox': bbox
        }
    
    def masks_to_polygons(
        self,
        masks: List[np.ndarray],
        properties: List[Dict],
        transform: Optional[rasterio.Affine] = None,
        crs: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Convert crown masks to polygon GeoDataFrame.
        
        Args:
            masks: List of crown masks
            properties: List of crown properties
            transform: Affine transform for georeferencing
            crs: Coordinate reference system
        
        Returns:
            GeoDataFrame with crown polygons
        """
        polygons = []
        
        for mask, props in zip(masks, properties):
            # Find contours
            contours = measure.find_contours(mask, 0.5)
            
            if not contours:
                continue
            
            # Take largest contour
            contour = max(contours, key=len)
            
            # Convert to polygon
            if transform is not None:
                # Transform pixel coordinates to world coordinates
                world_coords = [
                    transform * (col, row)
                    for row, col in contour
                ]
            else:
                world_coords = [(col, row) for row, col in contour]
            
            # Create polygon
            if len(world_coords) >= 3:
                polygon = ShapelyPolygon(world_coords)
                if polygon.is_valid and polygon.area > 0:
                    polygons.append({
                        'geometry': polygon,
                        **props
                    })
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(polygons, crs=crs)
        
        if not gdf.empty:
            # Add area in square meters if georeferenced
            if crs is not None:
                gdf['area_m2'] = gdf.to_crs(gdf.estimate_utm_crs()).area
        
        return gdf
    
    def process_single(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_polygons: bool = True,
        save_raster: bool = True,
        save_viz: bool = False
    ) -> Dict:
        """
        Process a single masked index file.
        
        Args:
            input_path: Path to masked index from Step 2
            output_dir: Directory to save results
            save_polygons: Save vector polygons
            save_raster: Save raster segmentation
            save_viz: Save visualization
        
        Returns:
            Dictionary with results and metadata
        """
        result = {
            'input': str(input_path),
            'status': 'processing',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            logger.info(f"\n📌 Processing: {Path(input_path).name}")
            
            # Load masked index
            data, metadata = self.load_masked_index(input_path)
            result['data_shape'] = metadata['shape']
            result['canopy_pixels'] = metadata['canopy_pixels']
            
            # Apply additional smoothing if requested
            if self.smoothing_sigma > 0: 
                data = gaussian_filter(data, sigma=self.smoothing_sigma)
                logger.info(f"   Applied additional smoothing (sigma={self.smoothing_sigma})")
            
            # Perform segmentation
            if self.use_cpp:
                logger.info("   Using C++ Forman gradient backend")
                seg_results = self.segment_with_cpp(data, metadata)
            else:
                logger.info("   Using Python fallback")
                seg_results = self.segment_with_python(data, metadata)
            
            result['num_critical'] = seg_results['num_critical']
            result['num_crowns'] = seg_results['num_crowns']
            
            logger.info(f"   Found {result['num_crowns']} tree crowns")
            
            # Save results
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                base_name = Path(input_path).stem.replace('_masked_index', '')
                
                # Save crown polygons
                if save_polygons and seg_results['crown_masks']:
                    gdf = self.masks_to_polygons(
                        seg_results['crown_masks'],
                        seg_results['crowns'],
                        transform=metadata.get('transform'),
                        crs=metadata.get('crs')
                    )
                    
                    if not gdf.empty:
                        # Save as GeoPackage (single file, no sidecar files!)
                        gpkg_path = output_dir / f"{base_name}_crowns.gpkg"
                        gdf.to_file(gpkg_path, driver='GPKG')
                        result['output_gpkg'] = str(gpkg_path)
                        
                        logger.info(f"   Saved {len(gdf)} crown polygons to {gpkg_path}")
                
                # Save raster segmentation
                if save_raster and seg_results['crown_masks']:
                    # Create combined raster
                    h, w = data.shape
                    raster_out = np.zeros((h, w), dtype=np.uint16)
                    
                    for i, mask in enumerate(seg_results['crown_masks'], 1):
                        raster_out[mask > 0] = i
                    
                    # Save as GeoTIFF
                    if metadata.get('transform'):
                        profile = {
                            'driver': 'GTiff',
                            'height': h,
                            'width': w,
                            'count': 1,
                            'dtype': rasterio.uint16,
                            'crs': metadata.get('crs'),
                            'transform': metadata.get('transform'),
                            'compress': 'lzw'
                        }
                        
                        raster_path = output_dir / f"{base_name}_segmentation.tif"
                        with rasterio.open(raster_path, 'w', **profile) as dst:
                            dst.write(raster_out, 1)
                        
                        result['output_raster'] = str(raster_path)
                        logger.info(f"   Saved segmentation raster")
                
                # Save visualization
                if save_viz:
                    viz_path = output_dir / f"{base_name}_step3_viz.png"
                    self.visualize_results(
                        data, metadata, seg_results,
                        result, viz_path
                    )
                    result['visualization'] = str(viz_path)
                
                # Save properties as CSV
                if seg_results['crowns']:
                    df = pd.DataFrame(seg_results['crowns'])
                    csv_path = output_dir / f"{base_name}_crown_properties.csv"
                    df.to_csv(csv_path, index=False)
                    result['properties_csv'] = str(csv_path)
            
            result['status'] = 'success'
            logger.info(f"✅ Completed: {Path(input_path).name}")
            
        except Exception as e:
            logger.error(f"❌ Failed: {Path(input_path).name} - {str(e)}")
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result
    
    def visualize_results(
        self,
        data: np.ndarray,
        metadata: Dict,
        seg_results: Dict,
        result: Dict,
        save_path: Union[str, Path]
    ):
        """
        Create visualization of segmentation results.
        
        Args:
            data: Original vegetation index
            metadata: Image metadata
            seg_results: Segmentation results
            result: Result dictionary
            save_path: Where to save visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original masked index
        im1 = axes[0, 0].imshow(data, cmap='viridis')
        axes[0, 0].set_title('Masked Vegetation Index')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        
        # Critical points (tree tops)
        axes[0, 1].imshow(data, cmap='gray', alpha=0.5)
        if seg_results['critical_points']:
            critical = np.array(seg_results['critical_points'])
            if len(critical.shape) == 1:
                # Single point
                axes[0, 1].plot(critical[1], critical[0], 'r.', markersize=10)
            else:
                # Multiple points
                axes[0, 1].plot(critical[:, 1], critical[:, 0], 'r.', markersize=5)
        axes[0, 1].set_title(f'Critical Points\n({seg_results["num_critical"]} maxima)')
        axes[0, 1].axis('off')
        
        # Segmentation (colored crowns)
        if seg_results['crown_masks']:
            # Create colored segmentation
            h, w = data.shape
            seg_rgb = np.zeros((h, w, 3))
            
            for i, mask in enumerate(seg_results['crown_masks']):
                color = plt.cm.tab20(i % 20)[:3]
                seg_rgb[mask > 0] = color
            
            axes[0, 2].imshow(seg_rgb)
        axes[0, 2].set_title(f'Segmented Crowns\n({seg_results["num_crowns"]} crowns)')
        axes[0, 2].axis('off')
        
        # Crown size distribution
        if seg_results['crowns']:
            sizes = [c['area_pixels'] for c in seg_results['crowns']]
            axes[1, 0].hist(sizes, bins=30, alpha=0.7, color='green')
            axes[1, 0].axvline(
                np.mean(sizes), color='red', linestyle='--',
                label=f'Mean: {np.mean(sizes):.0f}'
            )
            axes[1, 0].set_xlabel('Crown Size (pixels)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Crown Size Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Boundaries overlay
        axes[1, 1].imshow(data, cmap='viridis', alpha=0.6)
        if seg_results['crowns']:
            for crown in seg_results['crowns']:
                if crown.get('boundary'):
                    boundary = np.array(crown['boundary'])
                    axes[1, 1].plot(boundary[:, 1], boundary[:, 0], 'w-', linewidth=1)
        axes[1, 1].set_title('Crown Boundaries')
        axes[1, 1].axis('off')
        
        # Summary text
        axes[1, 2].axis('off')
        summary_text = (
            f"Results Summary:\n\n"
            f"Total crowns: {seg_results['num_crowns']}\n"
            f"Critical points: {seg_results['num_critical']}\n"
            f"Canopy coverage: {metadata['canopy_percent']:.1f}%\n"
            f"Mean crown size: {np.mean(sizes):.0f} pixels\n"
            f"Median crown size: {np.median(sizes):.0f} pixels\n"
            f"Min crown size: {np.min(sizes):.0f} pixels\n"
            f"Max crown size: {np.max(sizes):.0f} pixels\n\n"
            f"Persistence threshold: {self.persistence_threshold}\n"
            f"Using C++: {self.use_cpp}"
        )
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(
            f'ZS-TTCS - Step 3: Forman Gradient Segmentation\n'
            f'Input: {Path(result["input"]).name}',
            fontsize=14, y=1.02
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def process_batch(
        self,
        input_source: Union[str, Path],
        output_dir: Union[str, Path] = './step3_output',
        pattern: Optional[str] = None,
        save_polygons: bool = True,
        save_raster: bool = True,
        save_viz: bool = False,
        resume: bool = False,
        max_files: Optional[int] = None
    ) -> List[Dict]:
        """
        Process multiple masked index files in batch.
        
        Args:
            input_source: Directory containing masked indices OR single file
            output_dir: Directory to save results
            pattern: Pattern to match files (e.g., "*_masked_index.tif")
            save_polygons: Save vector polygons
            save_raster: Save raster segmentation
            save_viz: Save visualizations
            resume: Skip already processed files
            max_files: Maximum number of files to process
        
        Returns:
            List of result dictionaries
        """
        input_source = Path(input_source)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine if single file or batch
        if input_source.is_file():
            # Single file mode
            input_files = [input_source]
            logger.info(f"Single file mode: {input_source}")
        else:
            # Batch mode - find all masked indices
            input_files = self.find_masked_indices_in_directory(input_source, pattern)
            logger.info(f"Batch mode: Found {len(input_files)} files in {input_source}")
        
        # Limit if requested
        if max_files:
            input_files = input_files[:max_files]
            logger.info(f"Limited to first {max_files} files")
        
        # Check for existing results if resuming
        if resume:
            existing = set()
            if save_polygons:
                existing.update(output_dir.glob("*_crowns.geojson"))
            if save_raster:
                existing.update(output_dir.glob("*_segmentation.tif"))
            
            existing_stems = {f.stem.replace('_crowns', '').replace('_segmentation', '') 
                            for f in existing}
            
            input_files = [f for f in input_files if f.stem.replace('_masked_index', '') not in existing_stems]
            logger.info(f"Resume mode: {len(input_files)} files remaining")
        
        if not input_files:
            logger.warning("No files to process")
            return []
        
        # Process each file
        results = []
        
        for input_file in tqdm(input_files, desc="Step 3 Batch Processing"):
            # Process single file
            result = self.process_single(
                input_file,
                output_dir=output_dir,
                save_polygons=save_polygons,
                save_raster=save_raster,
                save_viz=save_viz
            )
            
            results.append(result)
            
            # Save intermediate results periodically
            if len(results) % 10 == 0:
                self.save_batch_summary(results, output_dir / "intermediate_summary.json")
        
        # Save final summary
        summary_path = output_dir / "batch_summary.json"
        self.save_batch_summary(results, summary_path)
        
        # Print summary statistics
        self.print_batch_summary(results)
        
        return results
    
    def save_batch_summary(self, results: List[Dict], summary_path: Path):
        """
        Save batch processing summary to JSON.
        
        Args:
            results: List of result dictionaries
            summary_path: Where to save summary
        """
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(results),
            'successful': sum(1 for r in results if r.get('status') == 'success'),
            'failed': sum(1 for r in results if r.get('status') == 'failed'),
            'total_crowns': sum(r.get('num_crowns', 0) for r in results if r.get('status') == 'success'),
            'results': [convert_for_json(r) for r in results]
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=convert_for_json)
        
        logger.info(f"Batch summary saved to {summary_path}")
    
    def print_batch_summary(self, results: List[Dict]):
        """
        Print summary of batch processing.
        
        Args:
            results: List of result dictionaries
        """
        successful = [r for r in results if r.get('status') == 'success']
        failed = [r for r in results if r.get('status') == 'failed']
        
        total_crowns = sum(r.get('num_crowns', 0) for r in successful)
        
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING SUMMARY - STEP 3")
        logger.info("=" * 60)
        logger.info(f"Total files: {len(results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Total crowns detected: {total_crowns}")
        
        if successful:
            avg_crowns = np.mean([r.get('num_crowns', 0) for r in successful])
            logger.info(f"Average crowns per image: {avg_crowns:.1f}")
        
        if failed:
            logger.info("\nFailed files:")
            for r in failed:
                logger.info(f"  - {Path(r['input']).name}: {r.get('error', 'Unknown error')}")
        
        logger.info("=" * 60)


# ======================================================================
# Command-line interface
# ======================================================================

def main():
    """Command-line interface for Step 3 with batch processing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ZS-TTCS Step 3: Forman gradient segmentation (BATCH CAPABLE)'
    )
    
    # Input arguments
    parser.add_argument(
        'input',
        help='Input masked index file or directory from Step 2'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./step3_output',
        help='Output directory (default: ./step3_output)'
    )
    
    # Processing parameters
    parser.add_argument(
        '--persistence',
        type=float,
        default=0.05,
        help='Persistence threshold for critical point filtering (default: 0.05)'
    )
    
    parser.add_argument(
        '--min-size',
        type=int,
        default=50,
        help='Minimum crown size in pixels (default: 50)'
    )
    
    parser.add_argument(
        '--max-size',
        type=int,
        default=5000,
        help='Maximum crown size in pixels (default: 5000)'
    )
    
    parser.add_argument(
        '--smooth',
        type=float,
        default=0.0,
        help='Additional Gaussian smoothing (default: 0.0)'
    )
    
    parser.add_argument(
        '--no-cpp',
        action='store_true',
        help='Disable C++ backend (use Python fallback)'
    )
    
    parser.add_argument(
        '--threads',
        type=int,
        default=-1,
        help='Number of threads for C++ backend (-1 = all)'
    )
    
    # Batch options
    parser.add_argument(
        '--pattern',
        help='File pattern for batch mode (e.g., "*_masked_index.tif")'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum number of files to process'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip already processed files'
    )
    
    parser.add_argument(
        '--no-polygons',
        action='store_true',
        help='Disable vector polygon output'
    )
    
    parser.add_argument(
        '--no-raster',
        action='store_true',
        help='Disable raster segmentation output'
    )
    
    parser.add_argument(
        '--viz',
        action='store_true',
        help='Save visualizations'
    )
    
    args = parser.parse_args()
    
    # Initialize segmentor
    segmentor = FormanGradientSegmentor(
        persistence_threshold=args.persistence,
        min_crown_size=args.min_size,
        max_crown_size=args.max_size,
        smoothing_sigma=args.smooth,
        use_cpp=not args.no_cpp,
        n_threads=args.threads,
        output_format=args.format
    )
    
    # Run batch processing
    results = segmentor.process_batch(
        input_source=args.input,
        output_dir=args.output,
        pattern=args.pattern,
        save_polygons=not args.no_polygons,
        save_raster=not args.no_raster,
        save_viz=args.viz,
        resume=args.resume,
        max_files=args.max_files
    )
    
    # Final message
    successful = [r for r in results if r.get('status') == 'success']
    if successful:
        logger.info(f"\n🎉 Step 3 complete! Successfully processed {len(successful)} files")
        logger.info(f"   Results saved to: {args.output}")
    else:
        logger.warning("\n⚠️ Step 3 completed with no successful results")


if __name__ == "__main__":
    main()