from pathlib import Path
from unittest.mock import MagicMock, patch

from src.models.auditor import CheckoutModelAuditor


def test_perform_validation_audit_returns_map50_and_map():
    with patch("src.models.auditor.YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        metrics = MagicMock()
        metrics.box.map50 = 0.75
        metrics.box.map = 0.55
        mock_model.val.return_value = metrics
        mock_yolo_cls.return_value = mock_model

        auditor = CheckoutModelAuditor(Path("fake.pt"))
        result = auditor.perform_validation_audit(Path("fake_data.yaml"))

    assert result == (0.75, 0.55)
    assert all(isinstance(v, float) for v in result)
