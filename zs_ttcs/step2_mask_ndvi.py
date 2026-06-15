#!/usr/bin/env python3
"""
ZS-TTCS - Step 2: Apply Mask to Vegetation Index (BATCH CAPABLE)
===============================================================
ZeroShot-Topological Tree Crown Segmentor
Step 2: Apply canopy mask from SegFormer to vegetation index

This module runs SECOND in the pipeline, creating a masked vegetation
index that will be used for Forman gradient computation in Step 3.

BATCH PROCESSING: Can process entire directories of images with
corresponding masks from Step 1.
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
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MaskedVegetationIndex:
    """
    Step 2: Apply canopy mask to vegetation index.
    
    This class takes the canopy mask from Step 1 and applies it to
    a vegetation index (NDVI for multispectral, Excess Green for RGB)
    to focus Forman gradient computation on tree regions only.
    
    BATCH CAPABLE: Can process multiple images with corresponding masks.
    """
    
    # Vegetation index methods
    INDEX_NDVI = 'ndvi'      # Requires NIR band
    INDEX_EXG = 'exg'         # Excess Green for RGB
    INDEX_VARI = 'vari'       # Visible Atmospherically Resistant Index
    INDEX_GLI = 'gli'         # Green Leaf Index
    
    # Supported image extensions
    SUPPORTED_EXTENSIONS = ['.tif', '.tiff', '.jpg', '.jpeg', '.png']
    
    def __init__(
        self,
        index_type: str = 'exg',  # Default to Excess Green for RGB
        smooth_sigma: float = 1.0,  # Gaussian smoothing sigma
        normalize: bool = True,     # Normalize to [0, 1] range
    ):
        """
        Initialize masked vegetation index calculator.
        
        Args:
            index_type: 'ndvi', 'exg', 'vari', or 'gli'
            smooth_sigma: Gaussian smoothing sigma (0 = no smoothing)
            normalize: Whether to normalize output to [0, 1]
        """
        self.index_type = index_type.lower()
        self.smooth_sigma = smooth_sigma
        self.normalize = normalize
        
        logger.info("=" * 60)
        logger.info("ZS-TTCS - Step 2: Apply Mask to Vegetation Index")
        logger.info("=" * 60)
        logger.info(f"Vegetation index: {self.index_type}")
        logger.info(f"Smoothing sigma: {self.smooth_sigma}")
        logger.info(f"Normalize output: {self.normalize}")
        logger.info("=" * 60)
    
    def compute_vegetation_index(self, image: np.ndarray) -> np.ndarray:
        """
        Compute vegetation index from image bands.
        
        Args:
            image: Image array (H, W, C) with bands in order:
                  For NDVI: bands [R, G, B, NIR] (NIR band 4)
                  For RGB indices: bands [R, G, B] (any order)
        
        Returns:
            Vegetation index array (H, W)
        """
        h, w, c = image.shape
        
        if self.index_type == self.INDEX_NDVI:
            if c < 4:
                logger.warning(f"NDVI requested but only {c} bands found. Falling back to ExG.")
                return self._compute_excess_green(image)
            return self._compute_ndvi(image)
        
        elif self.index_type == self.INDEX_EXG:
            return self._compute_excess_green(image)
        
        elif self.index_type == self.INDEX_VARI:
            return self._compute_vari(image)
        
        elif self.index_type == self.INDEX_GLI:
            return self._compute_gli(image)
        
        else:
            logger.warning(f"Unknown index {self.index_type}, using ExG")
            return self._compute_excess_green(image)
    
    def _compute_ndvi(self, image: np.ndarray) -> np.ndarray:
        """Compute NDVI: (NIR - R) / (NIR + R)"""
        # Assume bands: R=0, G=1, B=2, NIR=3
        red = image[..., 0].astype(np.float32)
        nir = image[..., 3].astype(np.float32)
        
        # Avoid division by zero
        denominator = nir + red
        denominator[denominator == 0] = 1e-10
        
        ndvi = (nir - red) / denominator
        return ndvi
    
    def _compute_excess_green(self, image: np.ndarray) -> np.ndarray:
        """Compute Excess Green Index: 2G - R - B"""
        # Normalize bands to [0, 1] if needed
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]
        
        exg = 2 * g - r - b
        return exg
    
    def _compute_vari(self, image: np.ndarray) -> np.ndarray:
        """
        Compute Visible Atmospherically Resistant Index:
        (G - R) / (G + R - B)
        """
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]
        
        denominator = g + r - b
        denominator[denominator == 0] = 1e-10
        
        vari = (g - r) / denominator
        return vari
    
    def _compute_gli(self, image: np.ndarray) -> np.ndarray:
        """Compute Green Leaf Index: (2G - R - B) / (2G + R + B)"""
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]
        
        numerator = 2 * g - r - b
        denominator = 2 * g + r + b
        denominator[denominator == 0] = 1e-10
        
        gli = numerator / denominator
        return gli
    
    def smooth_array(self, arr: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian smoothing to array."""
        if sigma <= 0:
            return arr
        
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(arr, sigma=sigma)
    
    def normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range."""
        arr_min = arr.min()
        arr_max = arr.max()
        
        if arr_max - arr_min < 1e-10:
            return np.zeros_like(arr)
        
        return (arr - arr_min) / (arr_max - arr_min)
    
    def find_images_in_directory(self, directory: Union[str, Path]) -> List[Path]:
        """
        Find all supported images in a directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            Sorted list of image paths
        """
        directory = Path(directory)
        images = []
        
        for ext in self.SUPPORTED_EXTENSIONS:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))
        
        # Remove duplicates and sort
        images = sorted(list(set(images)))
        
        logger.info(f"Found {len(images)} images in {directory}")
        return images
    
    def find_matching_mask(
        self,
        image_path: Path,
        mask_dir: Optional[Path] = None,
        mask_suffix: str = "_mask"
    ) -> Optional[Path]:
        """
        Find matching mask for an image.
        
        Args:
            image_path: Path to image
            mask_dir: Directory containing masks (if None, same as image)
            mask_suffix: Suffix used in Step 1 for masks
        
        Returns:
            Path to mask if found, None otherwise
        """
        # Determine mask directory
        if mask_dir is None:
            mask_dir = image_path.parent
        
        # Try different possible mask paths
        possible_masks = [
            mask_dir / f"{image_path.stem}{mask_suffix}.npy",
            mask_dir / f"{image_path.stem}_mask.npy",
            mask_dir / f"{image_path.stem}{mask_suffix}.tif",
            mask_dir / f"{image_path.stem}_mask.tif",
            image_path.parent / f"{image_path.stem}_mask.npy",
            Path("./masks") / f"{image_path.stem}_mask.npy",
        ]
        
        for mask_path in possible_masks:
            if mask_path.exists():
                logger.debug(f"Found mask for {image_path.name}: {mask_path}")
                return mask_path
        
        logger.warning(f"No mask found for {image_path.name}")
        return None
    
    def load_image(self, image_path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
        """
        Load image and its metadata.
        
        Args:
            image_path: Path to image
            
        Returns:
            - Image array (H, W, C)
            - Metadata dictionary
        """
        image_path = Path(image_path)
        
        with rasterio.open(image_path) as src:
            # Read all bands
            bands = []
            for i in range(1, src.count + 1):
                band = src.read(i)
                bands.append(band)
            
            image = np.stack(bands, axis=-1)
            metadata = src.profile.copy()
            
            logger.debug(f"Loaded image {image_path.name}: {image.shape} with {src.count} bands")
        
        return image, metadata
    
    def load_mask(self, mask_path: Union[str, Path], target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Load mask and ensure it matches target shape.
        
        Args:
            mask_path: Path to mask (.npy or .tif)
            target_shape: Desired shape (H, W)
        
        Returns:
            Mask array (H, W)
        """
        mask_path = Path(mask_path)
        
        # Load mask
        if mask_path.suffix == '.npy':
            mask = np.load(mask_path)
        else:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
        
        # Ensure mask matches image dimensions
        if mask.shape != target_shape:
            logger.warning(f"Mask shape {mask.shape} != target shape {target_shape}. Resizing...")
            from skimage.transform import resize
            mask = resize(mask, target_shape, preserve_range=True, order=0)
            mask = (mask > 0.5).astype(np.float32)  # Ensure binary
        
        return mask
    
    def process_single(
        self,
        image_path: Union[str, Path],
        mask_path: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        return_masked: bool = True,
        save_viz: bool = False
    ) -> Dict:
        """
        Process a single image with its mask.
        
        Args:
            image_path: Path to original image
            mask_path: Path to mask from Step 1
            output_path: If provided, save masked index
            return_masked: If True, return masked index; else return unmasked index
            save_viz: Whether to save visualization
        
        Returns:
            Dictionary with results and metadata
        """
        result = {
            'image': str(image_path),
            'status': 'processing',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            logger.info(f"\n📌 Processing: {Path(image_path).name}")
            
            # Load image
            image, metadata = self.load_image(image_path)
            h, w = image.shape[:2]
            result['image_shape'] = (h, w)
            result['num_bands'] = image.shape[-1]
            
            # Find or load mask
            if mask_path is None:
                mask_path = self.find_matching_mask(Path(image_path))
            
            if mask_path is None:
                raise FileNotFoundError(f"No mask found for {image_path}")
            
            result['mask'] = str(mask_path)
            
            # Load mask
            mask = self.load_mask(mask_path, (h, w))
            result['canopy_pixels'] = int(np.sum(mask > 0))
            result['canopy_percent'] = float(np.mean(mask > 0) * 100)
            
            logger.info(f"   Mask: {result['canopy_pixels']:,} canopy pixels ({result['canopy_percent']:.1f}%)")
            
            # Compute vegetation index
            logger.info(f"   Computing {self.index_type} index...")
            veg_index = self.compute_vegetation_index(image)
            result['index_range_raw'] = [float(veg_index.min()), float(veg_index.max())]
            
            # Apply mask
            if return_masked:
                masked_index = veg_index * mask
                logger.info(f"   Applied mask")
            else:
                masked_index = veg_index
            
            # Smooth if requested
            if self.smooth_sigma > 0:
                logger.info(f"   Smoothing with sigma={self.smooth_sigma}...")
                masked_index = self.smooth_array(masked_index, self.smooth_sigma)
            
            # Normalize if requested
            if self.normalize:
                logger.info(f"   Normalizing to [0, 1]...")
                masked_index = self.normalize_array(masked_index)
                result['index_range_normalized'] = [float(masked_index.min()), float(masked_index.max())]
            
            # Save if output path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save as GeoTIFF
                tif_path = output_path.with_suffix('.tif')
                self.save_as_geotiff(masked_index, tif_path, metadata, mask > 0)
                result['output_tif'] = str(tif_path)
                
                # Save as .npy for fast loading in Step 3
                npy_path = output_path.with_suffix('.npy')
                np.save(npy_path, masked_index)
                result['output_npy'] = str(npy_path)
                
                logger.info(f"   Saved to: {tif_path}")
            
            # Save visualization
            if save_viz and output_path:
                viz_path = output_path.parent / f"{Path(image_path).stem}_step2_viz.png"
                self.visualize_and_save(
                    image, mask, veg_index, masked_index,
                    result, viz_path
                )
                result['visualization'] = str(viz_path)
            
            result['status'] = 'success'
            logger.info(f"✅ Completed: {Path(image_path).name}")
            
        except Exception as e:
            logger.error(f"❌ Failed: {Path(image_path).name} - {str(e)}")
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result
    
    def save_as_geotiff(
        self,
        data: np.ndarray,
        output_path: Union[str, Path],
        reference_metadata: Dict,
        canopy_mask: np.ndarray
    ):
        """
        Save array as GeoTIFF with nodata outside canopy.
        
        Args:
            data: Array to save
            output_path: Output file path
            reference_metadata: Metadata from original image
            canopy_mask: Boolean mask of canopy pixels
        """
        profile = reference_metadata.copy()
        profile.update({
            'dtype': rasterio.float32,
            'count': 1,
            'compress': 'lzw',
            'nodata': -9999
        })
        
        # Set nodata outside canopy
        output_data = data.copy().astype(np.float32)
        output_data[~canopy_mask] = -9999
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(output_data, 1)
    
    def visualize_and_save(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        veg_index: np.ndarray,
        masked_index: np.ndarray,
        result: Dict,
        save_path: Union[str, Path]
    ):
        """
        Create and save visualization of Step 2 results.
        
        Args:
            image: Original RGB image
            mask: Canopy mask
            veg_index: Unmasked vegetation index
            masked_index: Masked vegetation index
            result: Result dictionary with metadata
            save_path: Where to save visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Original and mask
        # Original RGB
        if image.shape[-1] >= 3:
            rgb_display = image[..., :3].copy()
            if rgb_display.max() > 1.0:
                rgb_display = rgb_display / 255.0
            axes[0, 0].imshow(rgb_display)
        else:
            axes[0, 0].imshow(image[..., 0], cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Canopy mask
        axes[0, 1].imshow(mask, cmap='Greens')
        axes[0, 1].set_title(
            f'Canopy Mask\n'
            f'{result["canopy_pixels"]:,} pixels ({result["canopy_percent"]:.1f}%)'
        )
        axes[0, 1].axis('off')
        
        # Overlay
        if image.shape[-1] >= 3:
            overlay = rgb_display.copy()
            overlay_mask = np.stack([mask, mask, mask], axis=-1)
            overlay = np.where(
                overlay_mask > 0,
                overlay * 0.7 + np.array([0, 0.3, 0]),
                overlay
            )
            axes[0, 2].imshow(overlay)
            axes[0, 2].set_title('Mask Overlay')
            axes[0, 2].axis('off')
        
        # Row 2: Vegetation indices
        # Unmasked index
        im1 = axes[1, 0].imshow(veg_index, cmap='viridis')
        axes[1, 0].set_title(f'{self.index_type.upper()} (Unmasked)')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
        
        # Masked index (our output)
        masked_display = masked_index.copy()
        masked_display[~mask.astype(bool)] = np.nan
        im2 = axes[1, 1].imshow(masked_display, cmap='viridis')
        axes[1, 1].set_title(f'{self.index_type.upper()} (Masked)\nOutput for Step 3')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
        
        # Histogram
        canopy_values = masked_index[mask > 0]
        if len(canopy_values) > 0:
            axes[1, 2].hist(canopy_values.flatten(), bins=50, alpha=0.7, color='green')
            axes[1, 2].axvline(
                canopy_values.mean(), color='red', linestyle='--',
                label=f'Mean: {canopy_values.mean():.3f}'
            )
            axes[1, 2].legend()
        axes[1, 2].set_xlabel(f'{self.index_type.upper()} Value')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Distribution (within canopy)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle(
            f'ZS-TTCS - Step 2: {self.index_type.upper()} with Mask\n'
            f'Image: {Path(result["image"]).name}',
            fontsize=14, y=1.02
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def process_batch(
        self,
        input_source: Union[str, Path],
        mask_dir: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = './step2_output',
        image_pattern: str = None,
        save_viz: bool = False,
        resume: bool = False,
        max_images: Optional[int] = None
    ) -> List[Dict]:
        """
        Process multiple images in batch.
        
        This is the main batch processing function that mirrors Step 1's
        batch capability.
        
        Args:
            input_source: Directory containing images OR single image path
            mask_dir: Directory containing masks from Step 1
            output_dir: Directory to save results
            image_pattern: Pattern to match images (e.g., "*.tif")
            save_viz: Whether to save visualizations
            resume: Skip already processed images
            max_images: Maximum number of images to process (for testing)
        
        Returns:
            List of result dictionaries for each processed image
        """
        input_source = Path(input_source)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine if input is single file or directory
        if input_source.is_file():
            # Single image mode
            image_paths = [input_source]
            logger.info(f"Single image mode: {input_source}")
        else:
            # Batch mode - find all images
            if image_pattern:
                image_paths = sorted(list(input_source.glob(image_pattern)))
            else:
                image_paths = self.find_images_in_directory(input_source)
            
            logger.info(f"Batch mode: Found {len(image_paths)} images in {input_source}")
        
        # Limit if requested
        if max_images:
            image_paths = image_paths[:max_images]
            logger.info(f"Limited to first {max_images} images")
        
        # Setup mask directory
        if mask_dir:
            mask_dir = Path(mask_dir)
        else:
            mask_dir = input_source if input_source.is_dir() else input_source.parent
        
        # Check for existing results if resuming
        if resume:
            existing = set(output_dir.glob("*_masked_index.npy"))
            existing_stems = {f.stem.replace('_masked_index', '') for f in existing}
            
            image_paths = [p for p in image_paths if p.stem not in existing_stems]
            logger.info(f"Resume mode: {len(image_paths)} images remaining")
        
        if not image_paths:
            logger.warning("No images to process")
            return []
        
        # Process each image
        results = []
        
        for img_path in tqdm(image_paths, desc="Step 2 Batch Processing"):
            # Determine output path for this image
            out_path = output_dir / f"{img_path.stem}_masked_index"
            
            # Find mask
            mask_path = self.find_matching_mask(img_path, mask_dir)
            
            # Process single image
            result = self.process_single(
                img_path,
                mask_path=mask_path,
                output_path=out_path,
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
        # Convert numpy types to Python native for JSON
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
            'total_images': len(results),
            'successful': sum(1 for r in results if r.get('status') == 'success'),
            'failed': sum(1 for r in results if r.get('status') == 'failed'),
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
        
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total images: {len(results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        
        if successful:
            # Calculate average canopy coverage
            canopy_percents = [r.get('canopy_percent', 0) for r in successful]
            avg_canopy = np.mean(canopy_percents)
            logger.info(f"Average canopy coverage: {avg_canopy:.1f}%")
        
        if failed:
            logger.info("\nFailed images:")
            for r in failed:
                logger.info(f"  - {Path(r['image']).name}: {r.get('error', 'Unknown error')}")
        
        logger.info("=" * 60)


# ======================================================================
# Command-line interface
# ======================================================================

def main():
    """Command-line interface for Step 2 with batch processing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ZS-TTCS Step 2: Apply canopy mask to vegetation index (BATCH CAPABLE)'
    )
    
    # Input arguments
    parser.add_argument(
        'input',
        help='Input image or directory of images'
    )
    
    parser.add_argument(
        '--mask', '-m',
        help='Mask file or directory containing masks from Step 1'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./step2_output',
        help='Output directory (default: ./step2_output)'
    )
    
    # Processing options
    parser.add_argument(
        '--index',
        choices=['exg', 'ndvi', 'vari', 'gli'],
        default='exg',
        help='Vegetation index type (default: exg for RGB)'
    )
    
    parser.add_argument(
        '--smooth',
        type=float,
        default=1.0,
        help='Gaussian smoothing sigma (default: 1.0, 0=no smoothing)'
    )
    
    parser.add_argument(
        '--no-norm',
        action='store_true',
        help='Disable normalization to [0,1]'
    )
    
    # Batch options
    parser.add_argument(
        '--pattern',
        help='Image pattern for batch mode (e.g., "*.tif")'
    )
    
    parser.add_argument(
        '--max-images',
        type=int,
        help='Maximum number of images to process (for testing)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip already processed images'
    )
    
    # Output options
    parser.add_argument(
        '--viz',
        action='store_true',
        help='Save visualization for each image'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Process without saving (dry run)'
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = MaskedVegetationIndex(
        index_type=args.index,
        smooth_sigma=args.smooth,
        normalize=not args.no_norm
    )
    
    # Determine if batch or single
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image mode
        logger.info("Running in SINGLE IMAGE mode")
        
        # Find mask if not provided
        mask_path = args.mask
        if mask_path:
            mask_path = Path(mask_path)
        
        # Determine output path
        if args.no_save:
            output_path = None
        else:
            output_path = Path(args.output) / f"{input_path.stem}_masked_index"
            Path(args.output).mkdir(parents=True, exist_ok=True)
        
        # Process single image
        result = processor.process_single(
            input_path,
            mask_path=mask_path,
            output_path=output_path,
            save_viz=args.viz
        )
        
        # Print result
        if result['status'] == 'success':
            logger.info(f"\n✅ Processing complete!")
            logger.info(f"   Image: {result['image']}")
            logger.info(f"   Canopy pixels: {result['canopy_pixels']:,} ({result['canopy_percent']:.1f}%)")
            if 'output_tif' in result:
                logger.info(f"   Output: {result['output_tif']}")
        else:
            logger.error(f"\n❌ Processing failed: {result.get('error', 'Unknown error')}")
    
    else:
        # Batch mode
        logger.info("Running in BATCH mode")
        
        results = processor.process_batch(
            input_source=input_path,
            mask_dir=args.mask,
            output_dir=args.output,
            image_pattern=args.pattern,
            save_viz=args.viz,
            resume=args.resume,
            max_images=args.max_images
        )
        
        logger.info(f"\n✅ Batch processing complete!")
        logger.info(f"   Summary saved to: {Path(args.output) / 'batch_summary.json'}")


if __name__ == "__main__":
    main()