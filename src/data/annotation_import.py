"""Imports Label Studio YOLO-format exports into the GroceryStoreDataset leaf-class tree."""
from pathlib import Path

from PIL import Image


def parse_yolo_bbox_line(line: str, img_width: int, img_height: int) -> tuple[int, int, int, int]:
    """Converts one YOLO-format label line ('class_id cx cy w h', normalized) to pixel xyxy."""
    _, cx, cy, w, h = line.split()
    cx, cy, w, h = float(cx), float(cy), float(w), float(h)
    x_min = int((cx - w / 2) * img_width)
    y_min = int((cy - h / 2) * img_height)
    x_max = int((cx + w / 2) * img_width)
    y_max = int((cy + h / 2) * img_height)
    return (
        max(0, x_min),
        max(0, y_min),
        min(img_width, x_max),
        min(img_height, y_max),
    )


def crop_with_padding(
    image: Image.Image, box: tuple[int, int, int, int], padding_ratio: float = 0.05
) -> Image.Image:
    """Crops `image` to `box`, expanded by `padding_ratio` of the box's own width/height."""
    x_min, y_min, x_max, y_max = box
    pad_x = int((x_max - x_min) * padding_ratio)
    pad_y = int((y_max - y_min) * padding_ratio)
    padded_box = (
        max(0, x_min - pad_x),
        max(0, y_min - pad_y),
        min(image.width, x_max + pad_x),
        min(image.height, y_max + pad_y),
    )
    return image.crop(padded_box)


def load_class_names(export_dir: Path) -> dict[int, str]:
    """Reads Label Studio's classes.txt (one class name per line, indexed by line number = class_id)."""
    lines = (export_dir / "classes.txt").read_text().splitlines()
    return {idx: name for idx, name in enumerate(lines)}


def classify_category(class_name: str, category_keywords: dict[str, Path]) -> tuple[str, Path] | None:
    """Matches a class name against category keywords (substring match), returning the
    (matched keyword, destination directory) for the longest matching keyword — so a more
    specific keyword (e.g. an exact class path) wins over a shorter one that happens to be
    one of its prefixes (e.g. 'Myojo/Chicken' vs. 'Myojo/ChickenAbalone') — or None if no
    keyword matches."""
    matches = [(keyword, dest_dir) for keyword, dest_dir in category_keywords.items() if keyword in class_name]
    if not matches:
        return None
    return max(matches, key=lambda match: len(match[0]))


def variant_subpath(class_name: str, keyword: str) -> Path:
    """Path segments of class_name that follow the segment containing `keyword`, preserving
    any Brand/Variant nesting (e.g. 'Instant-Noodles/Nissin/328-KATONG-LAKSA' -> 'Nissin/328-KATONG-LAKSA').
    Falls back to the last segment when the keyword's segment has no remainder."""
    segments = class_name.split("/")
    match_index = next((i for i, seg in enumerate(segments) if keyword in seg), None)
    remainder = segments[match_index + 1:] if match_index is not None else []
    return Path(*remainder) if remainder else Path(segments[-1])


def import_label_studio_export(export_dir: Path, category_keywords: dict[str, Path]) -> list[Path]:
    """Reads a Label Studio YOLO-format export (shared images/labels/classes.txt covering
    multiple categories) and crops every bounding box in every label file, routing each
    crop into a variant subdirectory (preserving Brand/Variant nesting) under the
    destination directory whose keyword matches that box's class name."""
    class_names = load_class_names(export_dir)
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"
    written: list[Path] = []

    for label_path in sorted(labels_dir.glob("*.txt")):
        lines = label_path.read_text().strip().splitlines()
        if not lines:
            continue

        image_path = next(
            (candidate for ext in (".jpg", ".jpeg", ".png", ".webp")
             if (candidate := images_dir / f"{label_path.stem}{ext}").exists()),
            None,
        )
        if image_path is None:
            continue

        with Image.open(image_path).convert("RGB") as img:
            for box_index, line in enumerate(lines):
                class_name = class_names.get(int(line.split()[0]))
                if class_name is None:
                    continue

                match = classify_category(class_name, category_keywords)
                if match is None:
                    continue
                keyword, dest_dir = match

                variant_dir = dest_dir / variant_subpath(class_name, keyword)
                variant_dir.mkdir(parents=True, exist_ok=True)

                box = parse_yolo_bbox_line(line, img.width, img.height)
                cropped = crop_with_padding(img, box)
                dest_path = variant_dir / f"{image_path.stem}_{box_index}{image_path.suffix}"
                cropped.save(dest_path)
                written.append(dest_path)

    return written
