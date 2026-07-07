"""Day 4: duplicates scenes for under-represented classes with shadow stress. Used by main_day4_optimize.py."""
import cv2
import numpy as np
import random
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("Day4Augment")


class EnvironmentalStressAugmentor:
    """Applies illumination stress to simulate overhead retail environment shadows."""
    
    @staticmethod
    def apply_shadow_overlay(image: np.ndarray, alpha: float = 0.75, beta: int = -35) -> np.ndarray:
        """Modifies contrast scales to evaluate low-performing packaging lines under store shadow profiles."""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def optimize_target_classes(self, img_dir: Path, lbl_dir: Path, weak_class_id: int, duplication_rate: int = 2):
        valid_exts = {'.jpg', '.jpeg', '.png'}
        images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in valid_exts])
        augmented_count = 0
        
        for img_path in images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists(): continue
            
            with open(lbl_path, "r") as f:
                lines = f.readlines()
            
            # Check if scene contains target class needing optimization
            has_weak_target = any(int(line.split()[0]) == weak_class_id for line in lines if line.strip())
            
            if has_weak_target:
                img = cv2.imread(str(img_path))
                if img is None: continue
                
                for step in range(duplication_rate):
                    stressed_img = self.apply_shadow_overlay(
                        img, 
                        alpha=random.uniform(0.65, 0.85), 
                        beta=random.randint(-45, -20)
                    )
                    augmented_stem = f"aug_stress_{step}_{img_path.stem}"
                    
                    # Save enhanced variants while mirroring the original bounding box annotations
                    cv2.imwrite(str(img_dir / f"{augmented_stem}.jpg"), stressed_img)
                    with open(lbl_dir / f"{augmented_stem}.txt", "w") as lf:
                        lf.writelines(lines)
                augmented_count += 1
                
        logger.info(f"✨ Custom augmentations complete. Added {augmented_count * duplication_rate} variant fields.")