import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import make_classification, make_regression

from src.decision_tree import DecisionTree, Node


def test_node_is_leaf():
    node = Node()
    assert node.is_leaf() is True
    node.left = Node()
    assert node.is_leaf() is False


def test_impurity_classification_gini():
    tree = DecisionTree(task="classification", criterion="gini")
    # Pure
    assert tree._impurity([0, 0, 0]) == 0.0
    assert tree._impurity([1, 1]) == 0.0
    # Balanced 2-class: 1 - (0.5^2 + 0.5^2) = 0.5
    assert pytest.approx(tree._impurity([0, 1])) == 0.5
    # Empty
    assert tree._impurity([]) == 0.0


def test_impurity_classification_entropy():
    tree = DecisionTree(task="classification", criterion="entropy")
    # Pure
    assert tree._impurity([0, 0, 0]) == 0.0
    # Balanced 2-class: - (0.5 * log2(0.5) * 2) = 1.0
    assert pytest.approx(tree._impurity([0, 1])) == 1.0


def test_impurity_regression_mse():
    tree = DecisionTree(task="regression", criterion="mse")
    # Pure
    assert tree._impurity([5.0, 5.0, 5.0]) == 0.0
    # MSE is variance
    y = np.array([1.0, 2.0, 3.0, 4.0])
    expected_mse = np.var(y)
    assert pytest.approx(tree._impurity(y)) == expected_mse


def test_invalid_parameters():
    with pytest.raises(ValueError, match="Unknown task"):
        DecisionTree(task="clustering")

    with pytest.raises(ValueError, match="Classification criterion"):
        DecisionTree(task="classification", criterion="mse")

    with pytest.raises(ValueError, match="Regression criterion"):
        DecisionTree(task="regression", criterion="entropy")

    with pytest.raises(ValueError, match="max_depth"):
        DecisionTree(max_depth=-1)

    with pytest.raises(ValueError, match="min_samples_split"):
        DecisionTree(min_samples_split=1)

    with pytest.raises(ValueError, match="min_samples_leaf"):
        DecisionTree(min_samples_leaf=0)

    with pytest.raises(ValueError, match="min_impurity_decrease"):
        DecisionTree(min_impurity_decrease=-0.1)


def test_classification_toy_separable():
    # Simple 2D separable data: x0 > 0 -> class 1, x0 <= 0 -> class 0
    X = np.array([[-2.0, 1.0], [-1.0, 2.0], [1.0, -1.0], [2.0, 0.0]])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTree(task="classification", criterion="gini", max_depth=2)
    tree.fit(X, y)

    preds = tree.predict(X)
    np.testing.assert_array_equal(preds, y)

    proba = tree.predict_proba(X)
    assert proba.shape == (4, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(4))
    np.testing.assert_allclose(proba[:, 1], [0.0, 0.0, 1.0, 1.0])


def test_classification_entropy_criterion():
    X = np.array([[-3.0], [-1.0], [1.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    tree = DecisionTree(task="classification", criterion="entropy", max_depth=1)
    tree.fit(X, y)

    preds = tree.predict(X)
    np.testing.assert_array_equal(preds, y)


def test_regression_toy_step():
    # Step function: x < 0 -> 10.0, x >= 0 -> 50.0
    X = np.array([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])
    y = np.array([10.0, 10.0, 10.0, 50.0, 50.0, 50.0])

    tree = DecisionTree(task="regression", criterion="mse", max_depth=1)
    tree.fit(X, y)

    preds = tree.predict(X)
    np.testing.assert_allclose(preds, y)


def test_max_depth_constraint():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 4))
    y = rng.integers(0, 2, size=50)

    # depth 0 tree should only have root
    tree0 = DecisionTree(task="classification", max_depth=0).fit(X, y)
    assert tree0.root.is_leaf() is True

    # depth 1 tree should have root split and leaf children
    tree1 = DecisionTree(task="classification", max_depth=1).fit(X, y)
    if not tree1.root.is_leaf():
        assert tree1.root.left.is_leaf() is True
        assert tree1.root.right.is_leaf() is True


def test_min_samples_leaf_constraint():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] > 0).astype(int)

    min_leaf = 10
    tree = DecisionTree(
        task="classification",
        min_samples_leaf=min_leaf,
        max_depth=4,
    ).fit(X, y)

    def check_leaf_sizes(node, X_node):
        if node.is_leaf():
            assert len(X_node) >= min_leaf
            return
        left_mask = X_node[:, node.feature] <= node.threshold
        check_leaf_sizes(node.left, X_node[left_mask])
        check_leaf_sizes(node.right, X_node[~left_mask])

    check_leaf_sizes(tree.root, X)


def test_predict_proba_not_available_for_regression():
    X = np.array([[1.0], [2.0]])
    y = np.array([1.0, 2.0])
    tree = DecisionTree(task="regression").fit(X, y)
    with pytest.raises(ValueError, match="predict_proba is only available"):
        tree.predict_proba(X)


def test_predict_before_fit():
    tree = DecisionTree(task="classification")
    with pytest.raises(RuntimeError, match="fit must be called"):
        tree.predict([[1.0, 2.0]])


def test_benchmark_against_sklearn_classification():
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )

    custom_tree = DecisionTree(
        task="classification",
        criterion="gini",
        max_depth=4,
        min_samples_leaf=5,
    ).fit(X, y)

    sk_tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=4,
        min_samples_leaf=5,
        random_state=42,
    ).fit(X, y)

    custom_acc = np.mean(custom_tree.predict(X) == y)
    sk_acc = np.mean(sk_tree.predict(X) == y)

    # Both should perform solidly (> 80% accuracy) on this dataset
    assert custom_acc >= 0.80
    assert abs(custom_acc - sk_acc) <= 0.08


def test_benchmark_against_sklearn_regression():
    X, y = make_regression(
        n_samples=200,
        n_features=6,
        n_informative=4,
        noise=0.1,
        random_state=42,
    )

    custom_tree = DecisionTree(
        task="regression",
        criterion="mse",
        max_depth=4,
        min_samples_leaf=5,
    ).fit(X, y)

    sk_tree = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=4,
        min_samples_leaf=5,
        random_state=42,
    ).fit(X, y)

    custom_r2 = 1.0 - (np.var(y - custom_tree.predict(X)) / np.var(y))
    sk_r2 = 1.0 - (np.var(y - sk_tree.predict(X)) / np.var(y))

    assert custom_r2 >= 0.65
    assert abs(custom_r2 - sk_r2) <= 0.10
