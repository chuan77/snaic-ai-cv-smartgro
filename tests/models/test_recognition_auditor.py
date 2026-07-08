import numpy as np
import pandas as pd

from src.models.recognition_auditor import RecognitionAuditor


def test_analyze_returns_confusion_matrix_matching_known_counts(tmp_path):
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 2])
    auditor = RecognitionAuditor(class_names=["apple", "banana", "carrot"], output_dir=tmp_path)

    result = auditor.analyze(y_true, y_pred)

    expected_cm = np.array([
        [2, 1, 0],
        [0, 2, 0],
        [0, 0, 1],
    ])
    np.testing.assert_array_equal(result["confusion_matrix"], expected_cm)


def test_analyze_writes_per_class_report_csv_with_support(tmp_path):
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 2])
    auditor = RecognitionAuditor(class_names=["apple", "banana", "carrot"], output_dir=tmp_path)

    auditor.analyze(y_true, y_pred)

    df = pd.read_csv(tmp_path / "per_class_report.csv")
    assert list(df["class"]) == ["apple", "banana", "carrot"]
    assert list(df["support"]) == [3, 2, 1]


def test_analyze_writes_confusion_matrix_png(tmp_path):
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    auditor = RecognitionAuditor(class_names=["apple", "banana"], output_dir=tmp_path)

    result = auditor.analyze(y_true, y_pred)

    assert result["heatmap_path"] == tmp_path / "confusion_matrix.png"
    assert result["heatmap_path"].exists()
    assert result["heatmap_path"].stat().st_size > 0


def test_analyze_handles_zero_support_class_without_raising(tmp_path):
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    auditor = RecognitionAuditor(class_names=["apple", "banana", "carrot"], output_dir=tmp_path)

    auditor.analyze(y_true, y_pred)

    df = pd.read_csv(tmp_path / "per_class_report.csv")
    carrot_row = df[df["class"] == "carrot"].iloc[0]
    assert carrot_row["support"] == 0
    assert carrot_row["accuracy"] == 0


def test_analyze_writes_classification_report_text(tmp_path):
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    auditor = RecognitionAuditor(class_names=["apple", "banana"], output_dir=tmp_path)

    auditor.analyze(y_true, y_pred)

    report_path = tmp_path / "classification_report.txt"
    assert report_path.exists()
    assert "apple" in report_path.read_text()
