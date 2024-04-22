import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from skopt import BayesSearchCV
from skopt.space import Real
from ML_Algorithm.Util import Util


class GaussianNaiveBayes(Util):
    def __init__(self):
        self.__var_smoothing_begin = 1e-9
        self.__var_smoothing_end = 1.0
        self.__cv = 10

    def _run_gnb_bayesian(self, training_feature_array, training_label_array,
                          testing_feature_array, testing_label_array) -> tuple:
        # 베이지안 최적화를 위한 파라미터 스페이스
        search_spaces = {
            'var_smoothing': Real(self.__var_smoothing_begin, self.__var_smoothing_end, prior='log-uniform')
        }

        gnb = GaussianNB()

        # BayesSearchCV 객체 생성
        bayes_search = BayesSearchCV(gnb, search_spaces, n_iter=64, cv=self.__cv, scoring='f1_weighted', verbose=1)

        bayes_search.fit(training_feature_array, training_label_array)

        y_pred = bayes_search.predict(testing_feature_array)
        result = classification_report(testing_label_array, y_pred, zero_division=0, output_dict=True)

        return result, bayes_search.best_params_, bayes_search.best_score_

    def _run_gnb_all(self, training_feature_array, training_label_array, testing_feature_array, testing_label_array) \
            -> tuple:
        # 파라미터 그리드 정의
        param_grid = {
            'var_smoothing': self._generate_values(self.__var_smoothing_begin, self.__var_smoothing_end, 100, False)
        }

        gnb = GaussianNB()

        # GridSearchCV 객체 생성
        grid_search = GridSearchCV(gnb, param_grid, cv=self.__cv, scoring='f1_weighted', verbose=1)

        grid_search.fit(training_feature_array, training_label_array)

        y_pred = grid_search.predict(testing_feature_array)
        result = classification_report(testing_label_array, y_pred, zero_division=0, output_dict=True)

        return result, grid_search.best_params_, grid_search.best_score_