"""FastAPI backend serving real YOLO detections and catalog data to the frontend."""


def leaf_display_name(product_name: str) -> str:
    """Derives a human-readable label from a dataset-relative class path, e.g.
    'Fruit/Apple/Royal-Gala' -> 'Royal Gala'."""
    leaf = product_name.split("/")[-1]
    return leaf.replace("-", " ").title()


def xyxy_to_fractional_xywh(
    box: list[float], img_width: int, img_height: int
) -> tuple[float, float, float, float]:
    """Converts a pixel xyxy box to a fractional top-left-origin (x, y, w, h) box."""
    x_min, y_min, x_max, y_max = box
    return (
        x_min / img_width,
        y_min / img_height,
        (x_max - x_min) / img_width,
        (y_max - y_min) / img_height,
    )
