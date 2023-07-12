from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import Ridge, RidgeClassifier

from esce.base_models import ClassifierModel, RegressionModel

MODELS = {
    "majority-classifier": ClassifierModel(
        lambda **args: DummyClassifier(strategy="most_frequent", **args),
        "majority classifier",
    ),
    "mean-regressor": RegressionModel(
        lambda **args: DummyRegressor(strategy="mean", **args), "mean regressor"
    ),
    "ridge-cls": ClassifierModel(
        lambda **args: RidgeClassifier(**args), "ridge classifier"
    ),
    "ridge-reg": RegressionModel(lambda **args: Ridge(**args), "ridge regressor"),
}
