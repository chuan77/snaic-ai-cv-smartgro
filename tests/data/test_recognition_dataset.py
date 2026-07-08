from pathlib import Path

from PIL import Image

from src.data.recognition_dataset import (
    build_recognition_dataframe,
    load_synthetic_val_ground_truth,
)


def test_build_recognition_dataframe_returns_one_row_per_image(tmp_path):
    class_a = tmp_path / "Fruit" / "Apple"
    class_a.mkdir(parents=True)
    (class_a / "img1.jpg").touch()
    (class_a / "img2.png").touch()
    class_b = tmp_path / "Vegetables" / "Carrot"
    class_b.mkdir(parents=True)
    (class_b / "img1.jpg").touch()

    df = build_recognition_dataframe(tmp_path)

    assert set(df.columns) == {"crop_path", "fine"}
    assert len(df) == 3
    assert set(df["fine"]) == {"Fruit/Apple", "Vegetables/Carrot"}


def test_build_recognition_dataframe_filters_non_image_files(tmp_path):
    class_dir = tmp_path / "Fruit" / "Apple"
    class_dir.mkdir(parents=True)
    (class_dir / "img1.jpg").touch()
    (class_dir / "notes.txt").touch()

    df = build_recognition_dataframe(tmp_path)

    assert len(df) == 1


def test_build_recognition_dataframe_crop_path_is_resolvable(tmp_path):
    class_dir = tmp_path / "Fruit" / "Apple"
    class_dir.mkdir(parents=True)
    (class_dir / "img1.jpg").touch()

    df = build_recognition_dataframe(tmp_path)

    assert Path(df.iloc[0]["crop_path"]).exists()


def test_build_recognition_dataframe_respects_provided_class_map_subset(tmp_path):
    class_a = tmp_path / "Fruit" / "Apple"
    class_a.mkdir(parents=True)
    (class_a / "img1.jpg").touch()
    class_b = tmp_path / "Vegetables" / "Carrot"
    class_b.mkdir(parents=True)
    (class_b / "img1.jpg").touch()

    df = build_recognition_dataframe(tmp_path, class_map={"Fruit/Apple": 0})

    assert set(df["fine"]) == {"Fruit/Apple"}


def test_load_synthetic_val_ground_truth_returns_one_row_per_box(tmp_path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("names:\n  0: Fruit/Apple\n  1: Vegetables/Carrot\nnc: 2\n")
    Image.new("RGB", (100, 100)).save(images_dir / "scene_001.jpg")
    (labels_dir / "scene_001.txt").write_text("0 0.5 0.5 0.4 0.2\n1 0.2 0.2 0.1 0.1\n")

    df = load_synthetic_val_ground_truth(data_yaml, images_dir, labels_dir)

    assert list(df.columns) == ["image_path", "bbox", "fine"]
    assert len(df) == 2
    assert df.iloc[0]["fine"] == "Fruit/Apple"
    assert df.iloc[0]["bbox"] == (30, 40, 70, 60)
    assert df.iloc[1]["fine"] == "Vegetables/Carrot"
    assert Path(df.iloc[0]["image_path"]) == images_dir / "scene_001.jpg"
