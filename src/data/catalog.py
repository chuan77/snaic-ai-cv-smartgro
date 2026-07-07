"""Price-catalog persistence. `HardenedProductCatalog` is what `DeterministicGalleryBuilder`
(src/data/gallery.py) writes through; `ProductCatalog` is a simpler alternate not currently
wired into any pipeline entrypoint."""
import logging
from pathlib import Path
import pandas as pd
import torch
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Day1DataPipeline")


class ProductCatalog:
    """Encapsulates data storage boundaries for price registries."""
    def __init__(self, storage_path: str = "catalog_prices.csv"):
        self._storage_path = storage_path
        self._catalog: Dict[str, float] = {}

    def register_product(self, name: str, default_price: float = 1.50) -> None:
        self._catalog[name] = default_price

    def export_catalog(self) -> None:
        df = pd.DataFrame(list(self._catalog.items()), columns=['product_name', 'price_usd'])
        df.to_csv(self._storage_path, index=False)

class HardenedProductCatalog:
    """Manages store inventory datasets with transactional sorting and persistence."""
    def __init__(self, export_path: Path):
        self.export_path = export_path
        self.export_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: List[Dict[str, Any]] = []

    def register_product(self, class_id: int, name: str, base_price: float = 1.50) -> None:
        self._records.append({"class_id": class_id, "product_name": name, "price_usd": base_price})

    def serialize(self) -> None:
        df = pd.DataFrame(self._records).sort_values(by="class_id").reset_index(drop=True)
        df.to_csv(self.export_path, index=False)
        logger.info(f"💾 Price catalog saved securely to: {self.export_path}")