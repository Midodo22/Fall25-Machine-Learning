import typing as t
import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import sklearn


class LogisticRegression:
    def __init__(self, learning_rate: float, num_iterations: int):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[np.float64],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        m, n = inputs.shape
        self.weights = np.zeros(n)
        self.intercept = 0

        for _ in range(self.num_iterations):
            z = np.dot(inputs, self.weights) + self.intercept
            h = self.sigmoid(z)

            dw = (1 / m) * np.dot(np.transpose(inputs), (h - targets))
            db = (1 / m) * np.sum(h - targets)

            self.weights -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db

    def predict(self, x) -> t.Tuple[t.Sequence[np.float_], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        probability = self.sigmoid(np.dot(x, self.weights) + self.intercept)
        pred_class = (probability >= 0.5).astype(int)
        return probability, pred_class

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))


class FLD:
    """Implement FLD
    You can add arguments as you need,
    but don't modify those already exist variables.
    """
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(
        self,
        inputs: npt.NDArray[np.float64],
        targets: t.Sequence[int],
    ) -> None:
        class0 = []
        class1 = []
        for i in range(0, targets.size):
            if targets[i] == 0:
                class0.append(inputs[i])
            else:
                class1.append(inputs[i])

        class0 = np.array(class0)
        class1 = np.array(class1)

        # mean
        self.m0 = np.mean(class0, axis=0)
        self.m1 = np.mean(class1, axis=0)

        # subtract mean from data
        class0_mc = class0 - self.m0
        class1_mc = class1 - self.m1

        # between class covariance
        diff = (self.m1 - self.m0).reshape(-1, 1)
        self.sb = diff @ np.transpose(diff)

        # covariance
        class0_cov = np.transpose(class0_mc) @ class0_mc
        class1_cov = np.transpose(class1_mc) @ class1_mc

        # within class covariance
        self.sw = class0_cov + class1_cov

        self.w = np.dot(np.linalg.inv(self.sw), (self.m1 - self.m0))
        self.w = self.w / np.linalg.norm(self.w)

        # calculate slope and intercept
        self.threshold = 0.5 * (self.w @ (self.m0 + self.m1))
        self.slope = -self.w[0] / self.w[1]
        self.intercept = self.threshold / self.w[1]

    def predict(self, x) -> t.Sequence[t.Union[int, bool]]:
        prediction = x @ self.w
        pred_class = (prediction >= self.threshold).astype(int)
        return prediction, pred_class

    def plot_projection(self, x, y_true, y_pred, y_pred_class):
        # plot dots
        colors = np.where((y_pred_class == y_true), 'g', 'r')
        markers = np.where(y_pred_class == 0, 'o', '^')
        plt.figure(figsize=(8, 6))
        for i in range(len(x)):
            plt.scatter(
                x[i, 0], x[i, 1],
                c=colors[i],
                marker=markers[i]
            )

        # data projections
        center = np.mean(x, axis=0)
        w_norm = self.w / np.linalg.norm(self.w)
        projections = ((x - center) @ w_norm)[:, None] * w_norm + center
        for i in range(len(x)):
            plt.plot([x[i, 0], projections[i, 0]], [x[i, 1], projections[i, 1]], color='gray', alpha=0.5)

        # plot lines
        # projection line
        # mid_x = np.mean(x)
        # mid_y = np.mean(y_pred)
        # proj_x = np.array([mid_x - self.w[0], mid_x + self.w[0]])
        # proj_y = np.array([mid_y - self.w[1], mid_y + self.w[1]])
        # plt.axline(proj_x, proj_y, color='gray', label='Projection line')
        slope_proj = w_norm[1] / w_norm[0]
        plt.axline(center, slope=slope_proj, color='gray', label='Projection line')

        # decision line
        plt.axline((0, self.intercept), slope=self.slope, color='blue', linestyle='--', label='Decision boundary')

        plt.title(f"Projection onto FLD axis (slope={self.slope}, intercept={self.intercept})")
        # plt.grid(True)
        plt.legend()
        filename = "projection.png"
        plt.savefig(filename)
        plt.show()


def compute_auc(y_trues, y_preds):
    return sklearn.metrics.roc_auc_score(y_trues, y_preds)


def accuracy_score(y_trues, y_preds):
    return 1 - np.abs(y_trues - y_preds).mean()


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )
    print(y_train.shape)

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=0.16,  # You can modify the parameters as you want
        num_iterations=1000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercept: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['27', '30']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    """
    (TODO): Implement your code to
    1) Fit the FLD model
    2) Make prediction
    3) Compute the evaluation metrics

    Please also take care of the variables you used.
    """
    FLD_.fit(x_train, y_train)
    y_fld_pred, y_fld_pred_class = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_fld_pred_class)
    auc_score = compute_auc(y_test, y_fld_pred)

    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1} of {cols=}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    """
    (TODO): Implement your code below to plot the projection
    """
    FLD_.plot_projection(x_test, y_test, y_fld_pred, y_fld_pred_class)


if __name__ == '__main__':
    main()
