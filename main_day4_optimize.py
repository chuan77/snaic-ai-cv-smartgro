from pathlib import Path
from src.data.gallery import GroceryDatasetIndexer
from src.models.optimizer import EnvironmentalStressAugmentor

if __name__ == "__main__":
    print("Executing Day 4: Targeted Class Illumination Stress Balancing")
    cmap = GroceryDatasetIndexer(Path("./dataset/GroceryStoreDataset/dataset/train")).build_class_map()
    augmentor = EnvironmentalStressAugmentor()
    augmentor.optimize_target_classes(
         img_dir=Path("./synthetic_dataset/train/images"),
         lbl_dir=Path("./synthetic_dataset/train/labels"),
         weak_class_id=cmap["Snacks/Chocolate-Bar"]
     )