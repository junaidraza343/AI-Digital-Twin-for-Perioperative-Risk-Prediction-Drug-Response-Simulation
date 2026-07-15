"""Baseline IOH predictors: MAP-only logistic, GBDT, random forest, elastic-net."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import twin.config as c

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:  # pragma: no cover - environment dependent
    from sklearn.ensemble import HistGradientBoostingClassifier
    _HAS_LGBM = False

MAP_FEATURES = ["map_last", "map_slope"]


class MapOnlyModel:
    """Logistic regression on current MAP + MAP slope only."""

    def __init__(self):
        self.pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        )

    def fit(self, X, y):
        self.pipe.fit(X[MAP_FEATURES], y)
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X[MAP_FEATURES])[:, 1]


class GbdtModel:
    """Gradient-boosted trees on the full feature set."""

    def __init__(self):
        if _HAS_LGBM:
            self.model = LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=31,
                random_state=c.SEED, n_jobs=-1, verbose=-1,
            )
        else:  # pragma: no cover
            self.model = HistGradientBoostingClassifier(random_state=c.SEED)
        self.columns = None

    def fit(self, X, y):
        self.columns = [col for col in X.columns if X[col].dtype != object]
        self.model.fit(X[self.columns], y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.columns])[:, 1]


def _numeric_cols(X):
    return [col for col in X.columns if X[col].dtype != object]


class RandomForestModel:
    """Random forest on the full numeric feature set (no scaling needed)."""

    def __init__(self):
        self.model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_leaf=5,
                random_state=c.SEED, n_jobs=-1,
            ),
        )
        self.columns = None

    def fit(self, X, y):
        self.columns = _numeric_cols(X)
        self.model.fit(X[self.columns], y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.columns])[:, 1]


class ElasticLogisticModel:
    """Elastic-net logistic regression on the FULL feature set.

    The linear counterpart to MAP-only: uses every covariate + trend, so the gap
    to MAP-only isolates how much the extra features add (vs the nonlinear GBDT).
    """

    def __init__(self):
        self.pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                               C=1.0, max_iter=2000, random_state=c.SEED),
        )
        self.columns = None

    def fit(self, X, y):
        self.columns = _numeric_cols(X)
        self.pipe.fit(X[self.columns], y)
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X[self.columns])[:, 1]
