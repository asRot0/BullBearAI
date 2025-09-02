import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Evaluator:
    """
    Evaluation utility class for regression models.
    Provides MAE, RMSE, R², and additional residual analysis.
    """

    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

    def mae(self):
        return mean_absolute_error(self.y_true, self.y_pred)

    def rmse(self):
        return np.sqrt(mean_squared_error(self.y_true, self.y_pred))

    def r2(self):
        return r2_score(self.y_true, self.y_pred)

    def residuals(self):
        return self.y_true - self.y_pred

    def summary(self):
        """Return all metrics as a dictionary."""
        return {
            "MAE": self.mae(),
            "RMSE": self.rmse(),
            "R2": self.r2()
        }
