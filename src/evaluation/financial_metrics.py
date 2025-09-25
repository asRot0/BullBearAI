import numpy as np
import pandas as pd

class FinancialMetrics:
    def __init__(self, risk_free_rate: float = 0.01):
        """
        Initialize FinancialMetrics calculator.

        Parameters:
        - risk_free_rate: Annual risk-free rate (default 1%)
        """
        self.risk_free_rate = risk_free_rate

    def directional_accuracy(self, y_true, y_pred) -> float:
        """
        Calculate Directional Accuracy (%).
        Measures how often the model predicts the correct direction of movement.

        Parameters:
        - y_true: np.ndarray or pd.Series (actual values)
        - y_pred: np.ndarray or pd.Series (predicted values)

        Returns:
        - float: Directional accuracy percentage
        """
        true_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        correct = np.sum(true_dir == pred_dir)
        return (correct / len(true_dir)) * 100

    def cumulative_return(self, returns: pd.Series) -> float:
        """
        Calculate cumulative return (%).

        Parameters:
        - returns: Series of daily/period returns

        Returns:
        - float: Cumulative return percentage
        """
        cumulative = (1 + returns).prod() - 1
        return cumulative * 100

    def sharpe_ratio(self, returns: pd.Series, freq: int = 252) -> float:
        """
        Calculate annualized Sharpe Ratio.

        Parameters:
        - returns: Series of daily/period returns
        - freq: Number of periods in a year (252 trading days default)

        Returns:
        - float: Sharpe Ratio
        """
        excess_returns = returns - (self.risk_free_rate / freq)
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(freq)

    def max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown (%).

        Parameters:
        - returns: Series of daily/period returns

        Returns:
        - float: Maximum drawdown percentage
        """
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min() * 100

    def evaluate_all(self, y_true, y_pred, returns: pd.Series) -> dict:
        """
        Run all financial metrics for model comparison.

        Parameters:
        - y_true: Actual stock prices
        - y_pred: Predicted stock prices
        - returns: Series of model returns

        Returns:
        - dict with DA, Cumulative Return, Sharpe Ratio, Max Drawdown
        """
        return {
            "Directional Accuracy (%)": round(self.directional_accuracy(y_true, y_pred), 2),
            "Cumulative Return (%)": round(self.cumulative_return(returns), 2),
            "Sharpe Ratio": round(self.sharpe_ratio(returns), 2),
            "Max Drawdown (%)": round(self.max_drawdown(returns), 2),
        }

'''
# Generate returns (simple daily % change from predictions)
returns = y_pred.pct_change().dropna()

fm = FinancialMetrics(risk_free_rate=0.01)

results = fm.evaluate_all(y_true, y_pred, returns)
'''