import warnings

import torch

from transformers.modeling_outputs import CausalLMOutput
from sklearn.linear_model import Lasso


class NNModel:

    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, ids=None, **kwargs):
        if ids is None:
            ids = range(ys.shape[1])
        else:
            if max(ids) >= ys.shape[1] or min(ids) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in ids:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i:i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return CausalLMOutput(logits=torch.stack(preds, dim=1))


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:

    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, ids=None, **kwargs):
        xs, ys = xs.cpu(), ys.cpu()
        if ids is None:
            ids = range(ys.shape[1])
        else:
            if max(ids) >= ys.shape[1] or min(ids) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in ids:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i:i + 1]

            ws, _, _, _ = torch.linalg.lstsq(train_xs, train_ys.unsqueeze(2), driver=self.driver)

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return CausalLMOutput(logits=torch.stack(preds, dim=1))


class AveragingModel:

    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, ids=None, **kwargs):
        if ids is None:
            ids = range(ys.shape[1])
        else:
            if max(ids) >= ys.shape[1] or min(ids) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in ids:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i:i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return CausalLMOutput(logits=torch.stack(preds, dim=1))


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:

    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None, **kwargs):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter)

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i:i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return CausalLMOutput(logits=torch.stack(preds, dim=1))


# This is a baseline model that predicts the same value for all points
class NullClassifier:

    def __init__(self):
        self.name = "nullclassifier"

    def __call__(self, xs, ys, **kwargs):
        preds = torch.zeros_like(ys) - 1
        return CausalLMOutput(logits=preds)
