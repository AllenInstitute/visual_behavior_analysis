import seaborn as sns
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin

# Natural cubic spline fitting algorithm adapted from:
# https://github.com/madrury/basis-expansions/blob/master/basis_expansions/basis_expansions.py

class NaturalCubicSpline(BaseEstimator, TransformerMixin):
    """Apply a natural cubic basis expansion to an array.
    The features created with this basis expansion can be used to fit a
    piecewise cubic function under the constraint that the fitted curve is
    linear *outside* the range of the knots..  The fitted curve is continuously
    differentiable to the second order at all of the knots.
    This transformer is created by specifying only the number of knots.
    The knots are equally spaced within the *interior* of (max, min).  
    That is, the endpoints are *not* included as knots.
    Parameters
    ----------
    n_knots: positive integer
        The number of knots to create.
    """

    def __init__(self, n_knots):
        self.n_knots = n_knots

    def fit(self, X, *args, **kwargs):
        self.min = X.min()
        self.max = X.max()
        self.knots = np.linspace(self.min, self.max, num=(self.n_knots + 2))[1:-1]
        return self

    @property
    def n_params(self):
        return self.n_knots - 1

    def transform(self, X, **transform_params):
        if isinstance(X, pd.DataFrame):
            assert X.shape[1] == 1
            X = X.iloc[:, 0]
        X_spl = self._transform_array(X)
        if isinstance(X, pd.Series):
            col_names = self._make_names(X)
            X_spl = pd.DataFrame(X_spl, columns=col_names, index=X.index)
        return X_spl

    def _make_names(self, X):
        first_name = "{}_spline_linear".format(X.name)
        rest_names = ["{}_spline_{}".format(X.name, idx)
                      for idx in range(self.n_knots - 2)]
        return [first_name] + rest_names

    def _transform_array(self, X, **transform_params):
        X = np.asarray(X).reshape(-1)
        try:
            X_spl = np.zeros((X.shape[0], self.n_knots - 1))
        except IndexError:
            X_spl = np.zeros((1, self.n_knots - 1))
        X_spl[:, 0] = X

        def d(knot_idx, x):
            def positive_part(t): return np.maximum(0, t)
            def cube(t): return t**3
            numerator = (cube(positive_part(x - self.knots[knot_idx])) - cube(positive_part(x - self.knots[self.n_knots - 1])))
            denominator = self.knots[self.n_knots - 1] - self.knots[knot_idx]
            return numerator / denominator

        for i in range(0, self.n_knots - 2):
            X_spl[:, i + 1] = (d(i, X) - d(self.n_knots - 2, X))
        return X_spl


def perform_natural_cubic_regression(n_knots):
    '''use the scikit-learn Pipeline class to perform the regression'''
    return Pipeline([
        ('nat_cubic', NaturalCubicSpline(n_knots=n_knots)),
        ('regression', LinearRegression(fit_intercept=True))
    ])


def spline_regression(df, col_to_smooth, n_knots, time_column='time'):
    '''
    a convenience function for applying the fit.
    '''
    # make sure n_knots is an integer
    n_knots = int(n_knots)

    # reshape the time vector
    t = df_sample[time_column].values.reshape(-1, 1)

    # instantiate the regression Pipeline
    regression = perform_natural_cubic_regression(n_knots=n_knots)

    # fill any NaNs in the y-vector
    y = df_sample[col_to_smooth].fillna(method='bfill').values

    # perform the fit
    regression.fit(t, y)

    # return the prediction (same as the fit, since we're not splitting data)
    return regression.predict(t)
