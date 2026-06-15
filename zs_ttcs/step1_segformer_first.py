#!/usr/bin/env python3
"""
ZS-TTCS - Step 1: SegFormer First
==================================
ZeroShot-Topological Tree Crown Segmentor
Step 1: Semantic prior extraction using SegFormer (exactly as in ZS-TreeSeg)

This module runs FIRST in the pipeline, creating a canopy mask that will
be used to constrain the Forman gradient computation.

Reference: ZS-TreeSeg paper (https://arxiv.org/html/2602.00470v1)
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import rasterio
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SegFormerFirst:
    """
    Step 1: Semantic prior extraction using SegFormer.
    
    This class implements the exact semantic segmentation approach from
    ZS-TreeSeg paper, using a pretrained SegFormer model to create a
    binary canopy mask.
    
    Attributes:
        device: torch device (cuda/cpu)
        confidence_threshold: threshold for binary mask (default: 0.5)
        tree_class_ids: list of ADE20k class IDs for vegetation
    """
    
    # ADE20k dataset class IDs for vegetation (from ZS-TreeSeg paper)
    DEFAULT_TREE_CLASSES = [
        4,   # tree
        13,  # palm
        77,  # plant
        80,  # grass
        94,  # field
        108, # forest
        129, # bush
        130, # flower
        131, # vegetation
        132, # crop
    ]
    
    # Available SegFormer models (from best to good)
    AVAILABLE_MODELS = {
        'b5': "nvidia/segformer-b5-finetuned-ade-640-640",  # Best (used in paper)
        'b4': "nvidia/segformer-b4-finetuned-ade-512-512",  # Good balance
        'b3': "nvidia/segformer-b3-finetuned-ade-512-512",  # Faster
        'b2': "nvidia/segformer-b2-finetuned-ade-512-512",  # Even faster
        'b1': "nvidia/segformer-b1-finetuned-ade-512-512",  # Fastest
        'b0': "nvidia/segformer-b0-finetuned-ade-512-512",  # Tiny
    }
    
    def __init__(
        self,
        model_size: str = 'b5',
        confidence_threshold: float = 0.5,
        tree_class_ids: Optional[List[int]] = None,
        device: Optional[str] = None,
        use_amp: bool = True,  # Use automatic mixed precision
        batch_size: int = 1,    # For future multi-image support
    ):
        """
        Initialize SegFormer for semantic prior extraction.
        
        Args:
            model_size: 'b5' (best), 'b4', 'b3', 'b2', 'b1', 'b0' (fastest)
            confidence_threshold: Threshold for binary mask [0,1]
            tree_class_ids: Custom list of class IDs for vegetation
            device: 'cuda' or 'cpu' (auto-detected if None)
            use_amp: Use automatic mixed precision for faster inference
            batch_size: Batch size for processing (currently 1)
        
        Raises:
            ValueError: If model_size is invalid
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.tree_class_ids = tree_class_ids or self.DEFAULT_TREE_CLASSES
        self.use_amp = use_amp and torch.cuda.is_available()
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Validate model size
        if model_size not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model size '{model_size}' not recognized. "
                f"Choose from: {list(self.AVAILABLE_MODELS.keys())}"
            )
        
        self._load_model()
        self._log_init_info()
    
    def _load_model(self):
        """Load SegFormer model and processor."""
        model_name = self.AVAILABLE_MODELS[self.model_size]
        
        logger.info(f"Loading SegFormer-{self.model_size} from {model_name}")
        
        try:
            # Load processor and model
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.use_amp else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Get model info
            self.num_classes = self.model.config.num_labels
            self.input_size = self.model.config.image_size
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"   Input size: {self.input_size}")
            logger.info(f"   Number of classes: {self.num_classes}")
            logger.info(f"   Using device: {self.device}")
            logger.info(f"   Using AMP: {self.use_amp}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _log_init_info(self):
        """Log initialization information."""
        logger.info("=" * 60)
        logger.info("ZS-TTCS - Step 1: SegFormer First")
        logger.info("=" * 60)
        logger.info(f"Model: SegFormer-{self.model_size}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info(f"Monitoring {len(self.tree_class_ids)} vegetation classes")
        logger.info(f"Class IDs: {self.tree_class_ids}")
        logger.info("=" * 60)
    
    def load_image(self, image_source: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        """
        Load image from various sources.
        
        Args:
            image_source: Path to image or numpy array
            
        Returns:
            - RGB image array (H, W, 3) with values in [0, 255]
            - Metadata dictionary
        """
        metadata = {'source_type': 'unknown'}
        
        # Case 1: Already numpy array
        if isinstance(image_source, np.ndarray):
            img_array = image_source
            metadata['source_type'] = 'numpy'
            metadata['original_shape'] = img_array.shape
            
            # Ensure RGB
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[-1] > 3:
                img_array = img_array[..., :3]
            
            # Ensure uint8
            if img_array.dtype != np.uint8:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
            
            return img_array, metadata
        
        # Case 2: Path to image
        image_path = Path(image_source)
        metadata['source_type'] = 'file'
        metadata['filename'] = str(image_path)
        
        # Try rasterio first (for GeoTIFF)
        try:
            with rasterio.open(image_path) as src:
                # Read first 3 bands (RGB)
                if src.count >= 3:
                    r = src.read(1)
                    g = src.read(2)
                    b = src.read(3)
                    img_array = np.stack([r, g, b], axis=-1)
                    
                    # Normalize to 0-255 if needed
                    if img_array.max() > 255:
                        img_array = (img_array / img_array.max() * 255).astype(np.uint8)
                    elif img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                    
                    metadata.update({
                        'driver': src.driver,
                        'crs': src.crs,
                        'transform': src.transform,
                        'bounds': src.bounds,
                    })
                    
                    logger.info(f"   Loaded GeoTIFF: {img_array.shape}")
                    return img_array, metadata
        except Exception as e:
            logger.debug(f"Rasterio failed, trying PIL: {e}")
        
        # Fall back to PIL
        try:
            pil_image = Image.open(image_path).convert('RGB')
            img_array = np.array(pil_image)
            metadata.update({
                'driver': 'PIL',
                'format': pil_image.format,
                'mode': pil_image.mode,
                'size': pil_image.size
            })
            logger.info(f"   Loaded image with PIL: {img_array.shape}")
            return img_array, metadata
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise
    
    def preprocess(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Preprocess image for SegFormer.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Dictionary with 'pixel_values' tensor
        """
        # Convert to PIL Image (SegFormer processor expects this)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
        
        # Process with SegFormer's processor
        inputs = self.processor(images=pil_image, return_tensors="pt")
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    @torch.no_grad()
    def get_canopy_mask(
        self,
        image_source: Union[str, Path, np.ndarray],
        return_probability: bool = False,
        return_metadata: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        STEP 1: Get tree canopy mask from SegFormer.
        
        This is the main function that implements the semantic prior
        extraction exactly as described in ZS-TreeSeg paper.
        
        Args:
            image_source: Path to image or numpy array
            return_probability: If True, return probability map instead of binary mask
            return_metadata: If True, return metadata along with mask
            
        Returns:
            - Binary canopy mask (H, W) where 1 = tree, 0 = background
            - OR probability map (H, W) with values in [0,1]
            - Optional metadata dictionary
        """
        logger.info("📌 STEP 1.1: Running SegFormer inference...")
        
        # Load image
        image, metadata = self.load_image(image_source)
        original_h, original_w = image.shape[:2]
        logger.info(f"   Original image size: {original_h} x {original_w}")
        
        # Preprocess
        inputs = self.preprocess(image)
        
        # Run inference
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model(**inputs)
            logits = outputs.logits  # (1, num_classes, H/4, W/4)
        
        logger.info(f"   SegFormer output shape: {logits.shape}")
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)  # (1, num_classes, H/4, W/4)
        
        # Aggregate tree class probabilities
        tree_probs = torch.zeros_like(probs[0, 0])
        valid_classes = []
        
        for class_id in self.tree_class_ids:
            if class_id < probs.shape[1]:
                tree_probs += probs[0, class_id]
                valid_classes.append(class_id)
            else:
                logger.warning(f"Class {class_id} not in model (max class {probs.shape[1]-1})")
        
        logger.info(f"   Using {len(valid_classes)} valid vegetation classes")
        
        # Upsample to original size
        tree_probs = F.interpolate(
            tree_probs.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
            size=(original_h, original_w),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()
        
        # Apply threshold if binary mask requested
        if not return_probability:
            result = (tree_probs > self.confidence_threshold).astype(np.float32)
            logger.info(f"📌 STEP 1.2: Binary canopy mask created")
        else:
            result = tree_probs
            logger.info(f"📌 STEP 1.2: Probability map created")
        
        # Log statistics
        canopy_pixels = np.sum(result > 0) if not return_probability else np.sum(result > 0.5)
        canopy_percent = 100 * canopy_pixels / (original_h * original_w)
        logger.info(f"   Mask shape: {result.shape}")
        logger.info(f"   Canopy pixels: {canopy_pixels:,} ({canopy_percent:.1f}% of image)")
        logger.info(f"   Mean probability: {tree_probs.mean():.3f}")
        logger.info(f"   Max probability: {tree_probs.max():.3f}")
        logger.info("=" * 60)
        
        # Update metadata
        metadata.update({
            'original_size': (original_h, original_w),
            'canopy_percent': canopy_percent,
            'confidence_threshold': self.confidence_threshold,
            'tree_class_ids_used': valid_classes,
            'step1_complete': True
        })
        
        if return_metadata:
            return result, metadata
        return result
    
    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        **kwargs
    ) -> List[Dict]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of paths to images
            output_dir: Directory to save masks
            **kwargs: Additional arguments for get_canopy_mask
            
        Returns:
            List of metadata dictionaries for each image
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for img_path in tqdm(image_paths, desc="Processing images"):
            img_path = Path(img_path)
            
            try:
                # Get mask
                mask, metadata = self.get_canopy_mask(
                    img_path,
                    return_metadata=True,
                    **kwargs
                )
                
                # Save mask
                mask_path = output_dir / f"{img_path.stem}_canopy_mask.npy"
                np.save(mask_path, mask)
                
                # Save visualization
                viz_path = output_dir / f"{img_path.stem}_mask_viz.png"
                self.visualize_mask(
                    image_source=img_path,
                    mask=mask,
                    save_path=viz_path
                )
                
                metadata['mask_path'] = str(mask_path)
                metadata['viz_path'] = str(viz_path)
                results.append(metadata)
                
                logger.info(f"✅ Processed {img_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                results.append({'error': str(e), 'path': str(img_path)})
        
        # Save summary
        import json
        summary_path = output_dir / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Batch processing complete. Summary saved to {summary_path}")
        
        return results
    
    def visualize_mask(
        self,
        image_source: Union[str, Path, np.ndarray],
        mask: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = False,
        figsize: Tuple[int, int] = (15, 5)
    ):
        """
        Visualize the canopy mask from STEP 1.
        
        Args:
            image_source: Original image
            mask: Canopy mask (binary or probability)
            save_path: Path to save visualization
            show: Whether to display the plot
            figsize: Figure size
        """
        # Load original image
        if isinstance(image_source, (str, Path)):
            image, _ = self.load_image(image_source)
        else:
            image = image_source
        
        # Ensure image is uint8 for display
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original RGB Image', fontsize=12)
        axes[0].axis('off')
        
        # Mask type depends on input
        if mask.dtype == np.bool_ or mask.max() <= 1.0:
            # Binary mask
            im = axes[1].imshow(mask, cmap='Greens', alpha=0.7)
            axes[1].set_title(f'Binary Canopy Mask\n(threshold={self.confidence_threshold})', fontsize=12)
            axes[1].axis('off')
            
            # Overlay
            overlay = image.copy()
            overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay on Original', fontsize=12)
            axes[2].axis('off')
        else:
            # Probability map
            im = axes[1].imshow(mask, cmap='viridis', vmin=0, vmax=1)
            axes[1].set_title('Canopy Probability Map', fontsize=12)
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046)
            
            # Thresholded overlay
            binary = mask > self.confidence_threshold
            overlay = image.copy()
            overlay[binary] = overlay[binary] * 0.5 + np.array([0, 255, 0]) * 0.5
            axes[2].imshow(overlay)
            axes[2].set_title(f'Overlay (threshold={self.confidence_threshold})', fontsize=12)
            axes[2].axis('off')
        
        plt.suptitle('ZS-TTCS - Step 1: SegFormer First (Semantic Prior)', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_mask(
        self,
        mask: np.ndarray,
        output_path: Union[str, Path],
        reference_image: Optional[Union[str, Path]] = None,
        compress: bool = True
    ):
        """
        Save canopy mask as GeoTIFF (if reference available) or NPY.
        
        Args:
            mask: Canopy mask array
            output_path: Where to save
            reference_image: Reference image for georeferencing
            compress: Whether to compress GeoTIFF
        """
        output_path = Path(output_path)
        
        # If reference image provided, save as GeoTIFF
        if reference_image is not None:
            try:
                with rasterio.open(reference_image) as src:
                    profile = src.profile.copy()
                    
                    # Update for mask
                    profile.update({
                        'dtype': rasterio.uint8,
                        'count': 1,
                        'compress': 'lzw' if compress else None,
                        'nodata': 0
                    })
                    
                    with rasterio.open(output_path.with_suffix('.tif'), 'w', **profile) as dst:
                        dst.write(mask.astype(np.uint8), 1)
                    
                    logger.info(f"Mask saved as GeoTIFF: {output_path}.tif")
                    return
                    
            except Exception as e:
                logger.warning(f"Could not save as GeoTIFF: {e}")
        
        # Fall back to NPY
        np.save(output_path.with_suffix('.npy'), mask)
        logger.info(f"Mask saved as NPY: {output_path}.npy")


# ======================================================================
# Command-line interface
# ======================================================================

def main():
    """Command-line interface for Step 1."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ZS-TTCS Step 1: Extract canopy mask using SegFormer'
    )
    
    parser.add_argument(
        'input',
        help='Input image or directory of images'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./output',
        help='Output directory (default: ./output)'
    )
    
    parser.add_argument(
        '--model',
        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
        default='b5',
        help='SegFormer model size (default: b5)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for binary mask (default: 0.5)'
    )
    
    parser.add_argument(
        '--probability',
        action='store_true',
        help='Output probability map instead of binary mask'
    )
    
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        help='Device to use (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process multiple images (input should be directory)'
    )
    
    parser.add_argument(
        '--viz',
        action='store_true',
        help='Save visualization'
    )
    
    parser.add_argument(
        '--no-amp',
        action='store_true',
        help='Disable automatic mixed precision'
    )
    
    args = parser.parse_args()
    
    # Initialize segmentor
    segmentor = SegFormerFirst(
        model_size=args.model,
        confidence_threshold=args.threshold,
        device=args.device,
        use_amp=not args.no_amp
    )
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Batch processing
    if args.batch and input_path.is_dir():
        image_paths = list(input_path.glob('*.tif')) + \
                     list(input_path.glob('*.tiff')) + \
                     list(input_path.glob('*.jpg')) + \
                     list(input_path.glob('*.png'))
        
        logger.info(f"Found {len(image_paths)} images in {input_path}")
        
        segmentor.process_batch(
            image_paths,
            output_dir,
            return_probability=args.probability
        )
    
    # Single image processing
    else:
        # Get mask
        result, metadata = segmentor.get_canopy_mask(
            args.input,
            return_probability=args.probability,
            return_metadata=True
        )
        
        # Save mask
        output_stem = output_dir / Path(args.input).stem
        segmentor.save_mask(
            result,
            output_stem,
            reference_image=args.input if not args.probability else None
        )
        
        # Save visualization
        if args.viz:
            segmentor.visualize_mask(
                args.input,
                result,
                save_path=output_dir / f"{Path(args.input).stem}_viz.png"
            )
        
        logger.info(f"✅ Step 1 complete! Output saved to {output_dir}")


if __name__ == "__main__":
    main()