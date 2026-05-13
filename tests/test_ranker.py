"""
Unit tests for SmartphoneRanker class.
Run with: pytest tests/
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smartphone_ranker import SmartphoneRanker


@pytest.fixture
def sample_df():
    """Minimal sample dataset for testing."""
    return pd.DataFrame({
        'SMARTPHONENAME': ['Phone A', 'Phone B', 'Phone C'],
        'BATTERY':    [5000, 4000, 3000],
        'CAMERA':     [64,   48,   32],
        'STORAGE':    [256,  128,  64],
        'PROCESSOR':  [3.0,  2.5,  2.0],
        'PRICE':      [20000, 15000, 10000]
    })


@pytest.fixture
def features():
    return ['BATTERY', 'CAMERA', 'STORAGE', 'PROCESSOR', 'PRICE']


@pytest.fixture
def weights():
    return {
        'BATTERY':   0.20,
        'CAMERA':    0.25,
        'STORAGE':   0.15,
        'PROCESSOR': 0.25,
        'PRICE':     0.15
    }


class TestNormalizeData:
    def test_returns_dataframe(self, sample_df, features):
        ranker = SmartphoneRanker(sample_df)
        result = ranker.normalize_data(features)
        assert isinstance(result, pd.DataFrame)

    def test_normalized_values_are_between_0_and_1(self, sample_df, features):
        ranker = SmartphoneRanker(sample_df)
        result = ranker.normalize_data(features)
        for f in features:
            assert result[f].max() <= 1.0 + 1e-9
            assert result[f].min() >= 0.0 - 1e-9

    def test_original_data_unchanged(self, sample_df, features):
        original_battery = sample_df['BATTERY'].tolist()
        ranker = SmartphoneRanker(sample_df)
        ranker.normalize_data(features)
        assert ranker.data['BATTERY'].tolist() == original_battery


class TestApplyWeights:
    def test_raises_if_not_normalized_first(self, sample_df, features, weights):
        ranker = SmartphoneRanker(sample_df)
        with pytest.raises(ValueError):
            ranker.apply_weights(features, weights)

    def test_returns_dataframe(self, sample_df, features, weights):
        ranker = SmartphoneRanker(sample_df)
        ranker.normalize_data(features)
        result = ranker.apply_weights(features, weights)
        assert isinstance(result, pd.DataFrame)

    def test_weights_applied_correctly(self, sample_df, features, weights):
        ranker = SmartphoneRanker(sample_df)
        norm = ranker.normalize_data(features)
        weighted = ranker.apply_weights(features, weights)
        for f in features:
            expected = norm[f] * weights[f]
            pd.testing.assert_series_equal(weighted[f], expected)


class TestCalculateTopsis:
    def test_raises_if_weights_not_applied(self, sample_df, features):
        ranker = SmartphoneRanker(sample_df)
        ranker.normalize_data(features)
        with pytest.raises(ValueError):
            ranker.calculate_topsis(features)

    def test_scores_between_0_and_1(self, sample_df, features, weights):
        ranker = SmartphoneRanker(sample_df)
        ranker.normalize_data(features)
        ranker.apply_weights(features, weights)
        result = ranker.calculate_topsis(features)
        assert result['TOPSIS SCORE'].between(0, 1).all()

    def test_ranks_are_unique(self, sample_df, features, weights):
        ranker = SmartphoneRanker(sample_df)
        ranker.normalize_data(features)
        ranker.apply_weights(features, weights)
        result = ranker.calculate_topsis(features)
        assert result['RANK'].nunique() == len(sample_df)

    def test_result_sorted_by_rank(self, sample_df, features, weights):
        ranker = SmartphoneRanker(sample_df)
        ranker.normalize_data(features)
        ranker.apply_weights(features, weights)
        result = ranker.calculate_topsis(features)
        assert result['RANK'].tolist() == sorted(result['RANK'].tolist())