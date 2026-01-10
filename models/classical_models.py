from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

def train_models(X, y):
    models = {
        "lr": LogisticRegression(max_iter=1000),
        "svm": SVC(kernel="rbf", probability=True),
        "gb": GradientBoostingClassifier()
    }

    for m in models.values():
        m.fit(X, y)

    return models
