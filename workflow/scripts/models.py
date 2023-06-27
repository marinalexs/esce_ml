from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from base_models import ClassifierModel, RegressionModel
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Ridge, RidgeClassifier

MODELS = {
    "majority-classifier": ClassifierModel(
        lambda **args: DummyClassifier(strategy="most_frequent", **args),
        "majority classifier",
    ),
    "ridge-cls": ClassifierModel(
        lambda **args: RidgeClassifier(**args), "ridge classifier"
    ),
    "ridge-reg": RegressionModel(lambda **args: Ridge(**args), "ridge regressor"),
}
