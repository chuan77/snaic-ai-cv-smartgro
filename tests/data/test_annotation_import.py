from pathlib import Path

import pytest
from PIL import Image

from src.data.annotation_import import (
    parse_yolo_bbox_line,
    crop_with_padding,
    load_class_names,
    classify_category,
    variant_subpath,
    import_label_studio_export,
)


def test_parse_yolo_bbox_line_converts_normalized_center_to_pixel_xyxy():
    box = parse_yolo_bbox_line("0 0.5 0.5 0.4 0.2", img_width=100, img_height=100)

    assert box == (30, 40, 70, 60)


def test_parse_yolo_bbox_line_clamps_to_image_bounds():
    box = parse_yolo_bbox_line("0 0.05 0.05 0.5 0.5", img_width=100, img_height=100)

    assert box == (0, 0, 30, 30)


def test_crop_with_padding_expands_box_by_padding_ratio():
    image = Image.new("RGB", (100, 100))

    cropped = crop_with_padding(image, box=(40, 40, 60, 60), padding_ratio=0.5)

    assert cropped.size == (40, 40)


def test_crop_with_padding_clamps_to_image_bounds():
    image = Image.new("RGB", (100, 100))

    cropped = crop_with_padding(image, box=(0, 0, 10, 10), padding_ratio=1.0)

    assert cropped.size == (20, 20)


def test_load_class_names_indexes_by_line_number(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "classes.txt").write_text("Instant-Noodles/Maggi/Curry\nChocolate/Cadbury/Dairy-Milk\n")

    names = load_class_names(export_dir)

    assert names == {0: "Instant-Noodles/Maggi/Curry", 1: "Chocolate/Cadbury/Dairy-Milk"}


@pytest.mark.parametrize("class_name,expected_key", [
    ("Instant-Noodles/Nissin/CHU-QIAN-YI-DING-SESAME", "Instant-Noodles"),
    ("Ready-To-Eat/Instant-Noodles/Maggi/2-Minute-Curry", "Instant-Noodles"),
    ("Snacks/Chocolate/Cadbury/Dairy-Milk", "Chocolate"),
    ("Snacks/Chocolate-Bar/Cadbury/Dairy-Milk", "Chocolate"),
    ("Fruit/Apple/Royal-Gala", None),
])
def test_classify_category_matches_by_keyword(class_name, expected_key):
    category_keywords = {"Instant-Noodles": Path("/noodles"), "Chocolate": Path("/choc")}

    result = classify_category(class_name, category_keywords)

    expected = (expected_key, category_keywords[expected_key]) if expected_key else None
    assert result == expected


def test_classify_category_prefers_the_longest_matching_keyword():
    category_keywords = {
        "Ready-To-Eat/Instant-Noodles/Myojo/Chicken": Path("/chicken"),
        "Ready-To-Eat/Instant-Noodles/Myojo/ChickenAbalone": Path("/chicken-abalone"),
    }

    result = classify_category("Ready-To-Eat/Instant-Noodles/Myojo/ChickenAbalone", category_keywords)

    assert result == ("Ready-To-Eat/Instant-Noodles/Myojo/ChickenAbalone", Path("/chicken-abalone"))


@pytest.mark.parametrize("class_name,keyword,expected", [
    ("Ready-To-Eat/Instant-Noodles/Nissin-328-KATONG-LAKSA", "Instant-Noodles", Path("Nissin-328-KATONG-LAKSA")),
    ("Instant-Noodles/Nissin/328-KATONG-LAKSA", "Instant-Noodles", Path("Nissin/328-KATONG-LAKSA")),
    ("Snacks/Chocolate-Bar/Cadbury-RoastAlmond", "Chocolate", Path("Cadbury-RoastAlmond")),
    ("Snacks/Chocolate", "Chocolate", Path("Chocolate")),
])
def test_variant_subpath_preserves_segments_after_matched_keyword(class_name, keyword, expected):
    assert variant_subpath(class_name, keyword) == expected


def test_import_label_studio_export_nests_crops_under_the_variant_subpath(tmp_path):
    export_dir = tmp_path / "export"
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (export_dir / "classes.txt").write_text("Instant-Noodles/Maggi/Curry\n")
    Image.new("RGB", (100, 100), color=(255, 0, 0)).save(images_dir / "sample_001.jpg")
    (labels_dir / "sample_001.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    dest_dir = tmp_path / "dest"

    written = import_label_studio_export(export_dir, {"Instant-Noodles": dest_dir})

    expected_path = dest_dir / "Maggi" / "Curry" / "sample_001_0.jpg"
    assert written == [expected_path]
    with Image.open(expected_path) as cropped:
        assert cropped.size == (44, 44)


def test_import_label_studio_export_crops_every_box_in_a_multi_item_label_file(tmp_path):
    export_dir = tmp_path / "export"
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (export_dir / "classes.txt").write_text("Instant-Noodles/Maggi/Curry\nChocolate/Cadbury/Dairy-Milk\n")
    Image.new("RGB", (200, 200)).save(images_dir / "flatlay_001.jpg")
    (labels_dir / "flatlay_001.txt").write_text("0 0.25 0.25 0.2 0.2\n1 0.75 0.75 0.2 0.2\n")
    dest_dirs = {
        "Instant-Noodles": tmp_path / "dest" / "noodles",
        "Chocolate": tmp_path / "dest" / "chocolate",
    }

    written = import_label_studio_export(export_dir, dest_dirs)

    assert written == [
        dest_dirs["Instant-Noodles"] / "Maggi" / "Curry" / "flatlay_001_0.jpg",
        dest_dirs["Chocolate"] / "Cadbury" / "Dairy-Milk" / "flatlay_001_1.jpg",
    ]


def test_import_label_studio_export_skips_boxes_with_unmatched_category(tmp_path):
    export_dir = tmp_path / "export"
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (export_dir / "classes.txt").write_text("Fruit/Apple\n")
    Image.new("RGB", (100, 100)).save(images_dir / "sample_001.jpg")
    (labels_dir / "sample_001.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    written = import_label_studio_export(export_dir, {"Instant-Noodles": tmp_path / "dest"})

    assert written == []


def test_import_label_studio_export_skips_label_files_without_matching_image(tmp_path):
    export_dir = tmp_path / "export"
    (export_dir / "images").mkdir(parents=True)
    labels_dir = export_dir / "labels"
    labels_dir.mkdir(parents=True)
    (export_dir / "classes.txt").write_text("Instant-Noodles/Maggi/Curry\n")
    (labels_dir / "orphan.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    written = import_label_studio_export(export_dir, {"Instant-Noodles": tmp_path / "dest"})

    assert written == []


def test_import_label_studio_export_skips_empty_label_files(tmp_path):
    export_dir = tmp_path / "export"
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (export_dir / "classes.txt").write_text("Instant-Noodles/Maggi/Curry\n")
    Image.new("RGB", (50, 50)).save(images_dir / "empty_001.jpg")
    (labels_dir / "empty_001.txt").write_text("")

    written = import_label_studio_export(export_dir, {"Instant-Noodles": tmp_path / "dest"})

    assert written == []
