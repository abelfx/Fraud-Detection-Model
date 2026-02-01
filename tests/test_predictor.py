import numpy as np
import pandas as pd
from unittest.mock import Mock

from src.predictor import FraudPredictor


def test_predictor():
    predictor = FraudPredictor()
    mock_trainer = Mock()
    mock_trainer.feature_names = ["f1", "f2"]
    mock_trainer.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
    predictor.model_trainer = mock_trainer

    X = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    result = predictor.predict(X, return_proba=True, threshold=0.5)

    assert result["predictions"].tolist() == [1, 0]
    assert np.allclose(result["fraud_probability"], [0.8, 0.3])
    mock_trainer.predict_proba.assert_called_once()
