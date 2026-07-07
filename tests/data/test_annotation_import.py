from PIL import Image

from src.data.annotation_import import parse_yolo_bbox_line, crop_with_padding


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
