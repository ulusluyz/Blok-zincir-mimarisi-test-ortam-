from src.benchmark import calculate_metrics

def test_top3_counts_unique_categories_not_duplicate_blocks():
    rows = [("B", ["A", "A", "A", "B"])]
    result = calculate_metrics(rows)
    assert result["top_1"] == 0.0
    assert result["top_3"] == 1.0

def test_binary_error_rates_not_mislabeled():
    result = calculate_metrics([("A", ["B"])])
    assert result["fp_rate"] is None
    assert result["fn_rate"] is None

def test_empty_predictions():
    result = calculate_metrics([])
    assert result["top_1"] == 0.0
    assert result["fp_rate"] is None
