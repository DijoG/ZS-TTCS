# zs_ttcs/pipeline.py
from pathlib import Path
from .step1_segformer_first import SegFormerFirst
from .step2_mask_ndvi import MaskedVegetationIndex
from .step3_forman_gradient import FormanGradientSegmentor

def run_pipeline(image_path, output_dir="./output"):
    """Run complete segmentation pipeline"""
    output_dir = Path(output_dir)
    
    # Step 1
    step1 = SegFormerFirst()
    mask = step1.process_single(image_path, output_dir / "step1")
    
    # Step 2
    step2 = MaskedVegetationIndex()
    masked_ndvi = step2.process_single(image_path, mask, output_dir / "step2")
    
    # Step 3
    step3 = FormanGradientSegmentor(use_cpp=True)
    crowns = step3.process_single(masked_ndvi, output_dir / "step3")
    
    return crowns