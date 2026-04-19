from src.data.load import load_raw_dataset, prepare_base_dataset
from src.features.engineering import add_engineered_features, apply_unknown_strategy, split_features_and_target
from src.utils.config import IDENTIFIER_COLUMN, NAIVE_BAYES_PREFIX, TARGET_NAME


def test_raw_dataset_shape_and_columns():
    df = load_raw_dataset()
    assert df.shape == (10127, 23)
    assert any(column.startswith(NAIVE_BAYES_PREFIX) for column in df.columns)


def test_prepare_base_dataset_drops_leaky_columns_and_maps_target():
    df = prepare_base_dataset(load_raw_dataset())
    assert all(not column.startswith(NAIVE_BAYES_PREFIX) for column in df.columns)
    assert TARGET_NAME in df.columns
    assert set(df[TARGET_NAME].unique()) == {0, 1}


def test_identifier_isolated_from_predictors():
    df = prepare_base_dataset(load_raw_dataset())
    featured = add_engineered_features(df)
    featured = apply_unknown_strategy(featured, "keep")
    X, y, customer_ids = split_features_and_target(featured)
    assert IDENTIFIER_COLUMN not in X.columns
    assert len(y) == len(customer_ids) == len(X)
