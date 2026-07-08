import numpy as np

from src.models.recognizer import LinearHead


def _linearly_separable_dataset():
    rng = np.random.default_rng(0)
    class_0 = rng.normal(loc=[-5, -5], scale=0.5, size=(20, 2))
    class_1 = rng.normal(loc=[5, 5], scale=0.5, size=(20, 2))
    X = np.concatenate([class_0, class_1]).astype(np.float32)
    y = np.concatenate([np.zeros(20), np.ones(20)]).astype(np.int64)
    return X, y


def test_fit_predict_recovers_linearly_separable_labels():
    X, y = _linearly_separable_dataset()
    head = LinearHead(in_dim=2, n_classes=2).fit(X, y, epochs=200)

    predictions = head.predict(X)

    assert (predictions == y).mean() == 1.0


def test_fit_records_train_accuracy():
    X, y = _linearly_separable_dataset()
    head = LinearHead(in_dim=2, n_classes=2).fit(X, y, epochs=200)

    assert head.train_accuracy_ == (head.predict(X) == y).mean()


def test_same_seed_gives_identical_predictions():
    X, y = _linearly_separable_dataset()
    head_a = LinearHead(in_dim=2, n_classes=2, seed=42).fit(X, y, epochs=50)
    head_b = LinearHead(in_dim=2, n_classes=2, seed=42).fit(X, y, epochs=50)

    np.testing.assert_array_equal(head_a.predict(X), head_b.predict(X))


def test_save_load_round_trip_preserves_predictions(tmp_path):
    X, y = _linearly_separable_dataset()
    head = LinearHead(in_dim=2, n_classes=2).fit(X, y, epochs=200)
    save_path = tmp_path / "linear_head.pt"

    head.save(save_path)
    loaded = LinearHead.load(save_path)

    np.testing.assert_array_equal(loaded.predict(X), head.predict(X))
