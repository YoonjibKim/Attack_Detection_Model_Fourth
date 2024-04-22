from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Integer, Real
from ML_Algorithm.Util import Util


class MyAdaBoost(Util):
    def __init__(self):
        self.__n_estimators_begin = 1
        self.__n_estimators_end = 1000
        self.__learning_rate_begin = 0.0001
        self.__learning_rate_end = 1
        self.__cv = 10

    def _run_ab_bayesian(self, training_feature_array, training_label_array, testing_feature_array, testing_label_array) \
            -> tuple:
        ada_boost = AdaBoostClassifier(random_state=42, algorithm='SAMME')

        # 베이지안 최적화를 위한 파라미터 스페이스 정의
        search_spaces = {
            'n_estimators': Integer(self.__n_estimators_begin, self.__n_estimators_end),
            'learning_rate': Real(self.__learning_rate_begin, self.__learning_rate_end, prior='log-uniform')
        }

        # BayesSearchCV 객체 생성
        bayes_search = BayesSearchCV(ada_boost, search_spaces, n_iter=64, cv=self.__cv, scoring='f1_weighted',
                                     verbose=1, random_state=42)

        bayes_search.fit(training_feature_array, training_label_array.ravel())

        y_pred = bayes_search.predict(testing_feature_array)

        result_dict = classification_report(testing_label_array, y_pred, zero_division=0, output_dict=True)

        return result_dict, bayes_search.best_params_, bayes_search.best_score_

    def _run_ab_all(self, training_feature_array, training_label_array, testing_feature_array, testing_label_array) \
            -> tuple:
        ada_boost = AdaBoostClassifier(random_state=42, algorithm='SAMME')

        param_grid = {
            'n_estimators': self._generate_values(self.__n_estimators_begin, self.__n_estimators_end, 10, True),
            'learning_rate': self._generate_values(self.__learning_rate_begin, self.__learning_rate_end, 10, False)
        }

        # GridSearchCV 객체 생성
        grid_search = GridSearchCV(ada_boost, param_grid, cv=self.__cv, scoring='f1_weighted', verbose=1)

        grid_search.fit(training_feature_array, training_label_array.ravel())

        y_pred = grid_search.predict(testing_feature_array)

        result_dict = classification_report(testing_label_array, y_pred, zero_division=0, output_dict=True)

        return result_dict, grid_search.best_params_, grid_search.best_score_