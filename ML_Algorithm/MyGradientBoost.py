from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from ML_Algorithm.Util import Util


class MyGradientBoost(Util):
    def __init__(self):
        self.__learning_rate_begin = 0.0001
        self.__learning_rate_end = 1.0
        self.__n_estimators_begin = 1
        self.__n_estimators_end = 800
        self.__max_depth_begin = 1
        self.__max_depth_end = 9
        self.__min_samples_split_begin = 2
        self.__min_samples_split_end = 9
        self.__min_samples_leaf_begin = 1
        self.__min_samples_leaf_end = 9
        self.__subsample_begin = 0.01
        self.__subsample_end = 1.0
        self.__cv = 10

    def _run_gb_bayesian(self, training_feature_array, training_label_array, testing_feature_array, testing_label_array) \
            -> tuple:
        # 베이지안 최적화를 위한 파라미터 스페이스 정의
        search_spaces = {
            'learning_rate': Real(self.__learning_rate_begin, self.__learning_rate_end, prior='log-uniform'),
            'n_estimators': Integer(self.__n_estimators_begin, self.__n_estimators_end),
            'max_depth': Integer(self.__max_depth_begin, self.__max_depth_end),
            'min_samples_split': Integer(self.__min_samples_split_begin, self.__min_samples_split_end),
            'min_samples_leaf': Integer(self.__min_samples_leaf_begin, self.__min_samples_leaf_end),
            'subsample': Real(self.__subsample_begin, self.__subsample_end, prior='uniform')
        }

        gbc = GradientBoostingClassifier(random_state=42)

        # BayesSearchCV 객체 생성
        bayes_search = BayesSearchCV(gbc, search_spaces, n_iter=64, cv=self.__cv, scoring='f1_weighted', verbose=1)

        bayes_search.fit(training_feature_array, training_label_array)

        y_pred = bayes_search.predict(testing_feature_array)
        result = classification_report(testing_label_array, y_pred, zero_division=0, output_dict=True)

        return result, bayes_search.best_params_, bayes_search.best_score_

    def _run_gb_all(self, training_feature_array, training_label_array, testing_feature_array, testing_label_array) \
            -> tuple:
        # 파라미터 그리드 정의
        param_grid = {
            'learning_rate': self._generate_values(self.__learning_rate_begin, self.__learning_rate_end, 3, False),
            'n_estimators': self._generate_values(self.__n_estimators_begin, self.__n_estimators_end, 3, True),
            'max_depth': self._generate_values(self.__max_depth_begin, self.__max_depth_end, 3, True),
            'min_samples_split': self._generate_values(self.__min_samples_split_begin, self.__min_samples_split_end,
                                                       3, True),
            'min_samples_leaf': self._generate_values(self.__min_samples_leaf_begin, self.__min_samples_leaf_end,
                                                      3, True),
            'subsample': self._generate_values(self.__subsample_begin, self.__subsample_end, 3, False)
        }

        gbc = GradientBoostingClassifier(random_state=42)

        # GridSearchCV 객체 생성
        grid_search = GridSearchCV(gbc, param_grid, cv=self.__cv, scoring='f1_weighted', verbose=1)

        grid_search.fit(training_feature_array, training_label_array)

        y_pred = grid_search.predict(testing_feature_array)
        result = classification_report(testing_label_array, y_pred, zero_division=0, output_dict=True)

        return result, grid_search.best_params_, grid_search.best_score_
