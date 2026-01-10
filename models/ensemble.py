import numpy as np

def ensemble_predict(models, X):
    probs = []
    for model in models.values():
        probs.append(model.predict_proba(X))
    return np.mean(probs, axis=0)
