import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

class PredictDisease():
    def __init__(self, disease):
        model_path = "diabetes.pkl"
        if disease == None:
            model_path = None

        self.model = pickle.load(open(model_path, "rb"))

    def probability(disease, features):
        return self.model.predict_proba(features)

    def goToDoc(self, features, threshold):
        features = np.array([features])
        return self.probability(disease, features) > threshold

def main():
    p = PredictDisease("diabetes")
    p.goToDoc(, .5)



if __name__ == "__main__":
    main()