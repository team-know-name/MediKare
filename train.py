import pandas as pd
import numpy as np
import sys
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score


class RandomForestTrainer():
    def __init__(self, data_path, target_name):
        self.data = pd.read_csv(data_path)
        print(self.data)
        self.X, self.y = self.data.drop([target_name], axis = 1), self.data[target_name]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42, stratify = self.y)

    def train(self):
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000,  num = 10)]
        max_features = ["auto", "sqrt"]
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        min_samples_split = [2,5,10]
        min_samples_leaf = [1,2,4]
        bootstrap = [True, False]

        random_grid = {"n_estimators":n_estimators,
                       "max_features":max_features,
                       "max_depth":max_depth,
                       "min_samples_split":min_samples_split,
                       "min_samples_leaf":min_samples_leaf,
                       "bootstrap":bootstrap
                      }

        rf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose = 2, random_state = 42, n_jobs = -1)

        rf_random.fit(self.X_train, self.y_train)
        self.model = RandomForestClassifier()
        self.model = RandomForestClassifier(**rf_random.best_params_)
        self.model.fit(self.X_train, self.y_train)

    def test(self):
        print("Accuracy =", accuracy_score(self.model.predict(self.X_test), self.y_test))
        print("F1 Score =", f1_score(self.model.predict(self.X_test), self.y_test))

    def save(self, model_path):
        pickle.dump(self.model, open(model_path, "wb"))

    def predict(self, model_path):
        """

        :param model_path:
        :return:
        """
        return None

def main():
    data_path = sys.argv[1]
    target_name = sys.argv[2]
    model_path = sys.argv[3]
    model = RandomForestTrainer(data_path, target_name)
    model.train()
    model.test()
    model.save(model_path)




if __name__ == "__main__":
    main()
