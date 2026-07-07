from pathlib import Path
from src.data.gallery import UnifiedProductGallery
from src.data.gallery import GroceryDatasetIndexer
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("Day1Gallery")

# Day 1 Execution Strategy
if __name__ == "__main__":
    root_path = Path("./dataset/GroceryStoreDataset/dataset/train")
    if not root_path.exists():
        logger.error(f"Dataset root source missing at {root_path}. Please clone the repository first.")
    else:
        indexer = GroceryDatasetIndexer(root_path)
        cmap = indexer.build_class_map()
        gallery = UnifiedProductGallery(cmap, Path("./artifacts"))
        gallery.compile_gallery(root_path)