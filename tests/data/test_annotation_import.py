from PIL import Image

from src.data.annotation_import import (
    parse_yolo_bbox_line,
    crop_with_padding,
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


def test_import_label_studio_export_writes_cropped_images_and_returns_paths(tmp_path):
    export_dir = tmp_path / "export"
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    Image.new("RGB", (100, 100), color=(255, 0, 0)).save(images_dir / "sample_001.jpg")
    (labels_dir / "sample_001.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    dest_dir = tmp_path / "dest"

    written = import_label_studio_export(export_dir, dest_dir)

    assert written == [dest_dir / "sample_001.jpg"]
    with Image.open(dest_dir / "sample_001.jpg") as cropped:
        assert cropped.size == (44, 44)


def test_import_label_studio_export_skips_label_files_without_matching_image(tmp_path):
    export_dir = tmp_path / "export"
    (export_dir / "images").mkdir(parents=True)
    labels_dir = export_dir / "labels"
    labels_dir.mkdir(parents=True)
    (labels_dir / "orphan.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    written = import_label_studio_export(export_dir, tmp_path / "dest")

    assert written == []


def test_import_label_studio_export_skips_empty_label_files(tmp_path):
    export_dir = tmp_path / "export"
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    Image.new("RGB", (50, 50)).save(images_dir / "empty_001.jpg")
    (labels_dir / "empty_001.txt").write_text("")

    written = import_label_studio_export(export_dir, tmp_path / "dest")

    assert written == []
