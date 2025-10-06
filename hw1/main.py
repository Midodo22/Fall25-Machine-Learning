"""
1. Complete the implementation for the `...` part
2. Feel free to take strategies to make faster convergence
3. You can add additional params to the Class/Function as you need. But the key print out should be kept.
4. Traps in the code. Fix common semantic/stylistic problems to pass the linting
"""

from loguru import logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        """Question1
        Complete this function
        """
        # add column of ones
        X_intercept = np.hstack([np.ones((X.shape[0], 1)), X])

        X_tp = np.transpose(X_intercept)
        beta = np.linalg.solve(np.dot(X_tp, X_intercept), np.dot(X_tp, y))

        self.beta = beta
        self.intercept = beta[0]
        self.weights = beta[1:]

    def predict(self, X):
        """Question4
        Complete this function
        """
        prediction = np.dot(X, self.weights) + self.intercept
        return prediction


class LinearRegressionGradientdescent:
    def fit(
        self,
        X,
        y,
        epochs: float,
        learning_rate
    ):
        """Question2
        Complete this function
        """
        # Ensure X is always 2D
        X = np.atleast_2d(X)
        if X.shape[0] < X.shape[1]:
            X = X.T

        y = np.ravel(y)

        # Normalize data
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / X_std

        y_mean = np.mean(y)
        y_std = np.std(y)
        y_norm = (y - y_mean) / y_std

        # Init weights and intercept
        samples, features = X_norm.shape
        # self.weights = np.random.randn(features) * 0.01
        self.weights = np.zeros(features)
        self.intercept = 0.0
        n = len(X_norm)
        lr = learning_rate

        losses, lr_history = [], []
        for epoch in range(epochs):

            y_pred = np.dot(X_norm, self.weights) + self.intercept
            error = y_norm - y_pred
            loss = compute_mse(y_pred, y_norm)

            X_tp = np.transpose(X_norm)
            weight_derivative = -(2 / n) * np.dot(X_tp, error)
            bias_derivative = -(2 / n) * np.sum(error)

            weight_derivative = np.ravel(weight_derivative)
            self.weights = np.ravel(self.weights)

            self.weights -= lr * weight_derivative
            self.intercept -= lr * bias_derivative

            losses.append(loss)
            lr_history.append(lr)

            # Update learn rate
            # if epoch % 100:
            #     lr *= 0.65

            if epoch % 1000 == 0:
                logger.info(f'EPOCH {epoch}, {loss=:.4f}, {lr=:.4f}')

        self.weights = (y_std / X_std) * self.weights
        self.intercept = y_mean - np.dot(X_mean, self.weights) + y_std * self.intercept

        return losses, lr_history

    def predict(self, X):
        """Question4
        Complete this
        """
        prediction = np.dot(X, self.weights) + self.intercept
        return prediction


def compute_mse(prediction, ground_truth):
    mse = np.mean((prediction - ground_truth) ** 2)
    return mse


def main():
    train_df = pd.read_csv('./train.csv')  # Load training data
    test_df = pd.read_csv('./test.csv')  # Load test data
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)

    """This is the print out of question1"""
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    epoches = 6000
    losses, lr_history = LR_GD.fit(train_x, train_y, epoches, learning_rate=1e-3)

    """
    This is the print out of question2
    Note: You need to screenshot your hyper-parameters as well.
    """
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    """
    Question3: Plot the learning curve.
    Implement here
    """
    plt.plot(range(len(losses)), losses, label='Train MSE Loss')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.legend()
    filename = "loss.png"
    plt.savefig(filename)
    plt.show()

    """Question4"""
    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).mean()
    logger.info(f'Prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = (np.abs(mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()
