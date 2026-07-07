"""Imports Label Studio-annotated product photos into the GroceryStoreDataset tree
so Day 1's GroceryDatasetIndexer discovers them as new classes."""
from pathlib import Path

from src.data.annotation_import import import_label_studio_export

if __name__ == "__main__":
    noodle_written = import_label_studio_export(
        Path("./dataset/raw_photos/Instant-Noodles/labelstudio_export"),
        Path("./dataset/GroceryStoreDataset/dataset/train/Ready-To-Eat/Instant-Noodles"),
    )
    choc_written = import_label_studio_export(
        Path("./dataset/raw_photos/Chocolate-Bar/labelstudio_export"),
        Path("./dataset/GroceryStoreDataset/dataset/train/Snacks/Chocolate-Bar"),
    )
    print(f"Imported {len(noodle_written)} Instant-Noodles crops, {len(choc_written)} Chocolate-Bar crops.")
