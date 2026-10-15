import numpy as np
import pytest

from src.svm import PegasosSVM


def test_to_pm1_maps_zero_one_labels_and_stores_classes():
    model = PegasosSVM()

    encoded = model._to_pm1(np.array([0, 1, 0, 1]))

    np.testing.assert_array_equal(encoded, [-1, 1, -1, 1])
    np.testing.assert_array_equal(model.classes_, [0, 1])


@pytest.mark.parametrize(
    "labels",
    [
        np.array([]),
        np.array([0, 1, 2]),
        np.array([[0, 1], [0, 1]]),
        np.array([0, np.nan]),
    ],
)
def test_to_pm1_rejects_invalid_labels(labels):
    with pytest.raises(ValueError):
        PegasosSVM()._to_pm1(labels)


def test_fit_learns_separable_data_deterministically():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    first = PegasosSVM(
        lambda_=0.01,
        n_epochs=1000,
        random_state=7,
    ).fit(X, y)

    second = PegasosSVM(
        lambda_=0.01,
        n_epochs=1000,
        random_state=7,
    ).fit(X, y)

    assert first.w.shape == (first.n_components,)
    np.testing.assert_array_equal(first.w, second.w)
    assert first.b == second.b


def test_predict_returns_original_labels():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    model = PegasosSVM(
        lambda_=0.01,
        n_epochs=1000,
        random_state=7,
    ).fit(X, y)

    np.testing.assert_array_equal(model.predict(X), y)
    assert model.decision_function(X).shape == (X.shape[0],)


def test_prediction_requires_fitted_model():
    with pytest.raises(RuntimeError):
        PegasosSVM().predict(np.array([[0.0]]))


def test_prediction_rejects_wrong_feature_count():
    model = PegasosSVM(n_epochs=10).fit(
        np.array([[-1.0], [1.0]]),
        np.array([0, 1]),
    )

    with pytest.raises(ValueError):
        model.predict(np.array([[1.0, 2.0]]))

def test_rff_svm_learns_xor_deterministically():
    X = np.array([
        [-1.0, -1.0],
        [-1.0,  1.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0],
    ])
    y = np.array([0, 1, 1, 0])

    options = {
        "lambda_": 0.01,
        "n_epochs": 10000,
        "random_state": 12,
        "n_components": 256,
        "gamma": 1.0,
    }

    first = PegasosSVM(**options).fit(X, y)
    second = PegasosSVM(**options).fit(X, y)

    np.testing.assert_array_equal(first.predict(X), y)
    np.testing.assert_array_equal(
        first.decision_function(X),
        second.decision_function(X),
    )

@pytest.mark.parametrize("n_components", [0, -1, 2.5, True])
def test_constructor_rejects_invalid_n_components(n_components):
    with pytest.raises(ValueError, match="n_components"):
        PegasosSVM(n_components=n_components)


@pytest.mark.parametrize("gamma", [0, -1, np.inf, np.nan])
def test_constructor_rejects_invalid_gamma(gamma):
    with pytest.raises(ValueError, match="gamma"):
        PegasosSVM(gamma=gamma)

def test_objective_history_is_recorded():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    model = PegasosSVM(
        lambda_=0.01,
        n_epochs=20,
        learning_rate_scale=1.0,
        random_state=7,
    ).fit(X, y)

    assert len(model.objective_history_) == model.n_epochs + 1
    assert np.all(np.isfinite(model.objective_history_))