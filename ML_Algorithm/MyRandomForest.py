from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from ML_Algorithm.Util import Util


class MyRandomForest(Util):
    def __init__(self):
        self.__n_estimators_begin = 1
        self.__n_estimators_end = 500
        self.__max_depth_begin = 1
        self.__max_depth_end = 100
        self.__cv = 10

    def _run_rf_bayesian(self, training_feature_array, training_label_array, testing_feature_array, testing_label_array) \
            -> tuple:
        rf = RandomForestClassifier(random_state=42)

        search_spaces = {
            'n_estimators': Integer(self.__n_estimators_begin, self.__n_estimators_end),  # 트리의 수
            'max_features': Categorical([None, 'sqrt', 'log2']),  # 각 트리에서 고려할 최대 피처 수
            'max_depth': Integer(self.__max_depth_begin, self.__n_estimators_end),  # 트리의 최대 깊이
            'criterion': Categorical(['gini', 'entropy'])  # 분할 기준
        }

        # BayesSearchCV 객체 생성
        bayes_search = BayesSearchCV(rf, search_spaces, n_iter=64, cv=10, verbose=1, scoring='f1_weighted')

        bayes_search.fit(training_feature_array, training_label_array)

        predictions = bayes_search.predict(testing_feature_array)

        report = classification_report(testing_label_array, predictions, zero_division=0, output_dict=True)

        return report, bayes_search.best_params_, bayes_search.best_score_

    def _run_rf_all(self, training_feature_array, training_label_array, testing_feature_array, testing_label_array) \
            -> tuple:
        rf = RandomForestClassifier(random_state=42)

        param_grid = {
            'n_estimators': self._generate_values(self.__n_estimators_begin, self.__n_estimators_end, 5, True),
            'max_features': [None, 'sqrt', 'log2'],
            'max_depth': self._generate_values(self.__max_depth_begin, self.__max_depth_end, 5, True),
            'criterion': ['gini', 'entropy']
        }

        # GridSearchCV 객체 생성
        grid_search = GridSearchCV(rf, param_grid, cv=self.__cv, scoring='f1_weighted', verbose=1)

        grid_search.fit(training_feature_array, training_label_array)

        predictions = grid_search.predict(testing_feature_array)

        report = classification_report(testing_label_array, predictions, zero_division=0, output_dict=True)

        return report, grid_search.best_params_, grid_search.best_score_