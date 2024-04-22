from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from skopt import BayesSearchCV  # BayesSearchCV 임포트
from skopt.space import Real, Categorical, Integer

from ML_Algorithm.Util import Util


class MyDecisionTree(Util):
    def __init__(self):
        self.__max_depth_begin = 2
        self.__max_depth_end = 100
        self.__min_samples_split_begin = 2
        self.__min_samples_split_end = 100
        self.__min_samples_leaf_begin = 1
        self.__min_samples_leaf_end = 100
        self.__max_leaf_nodes_begin = 2
        self.__max_leaf_nodes_end = 100
        self.__min_impurity_decrease_begin = 0.0
        self.__min_impurity_decrease_end = 1.0
        self.__cv = 10

    def _run_dt_bayesian(self, training_feature_array, training_label_array,
                         testing_feature_array, testing_label_array) -> tuple:
        # 파라미터 스페이스를 정의
        search_spaces = {
            'criterion': Categorical(['gini', 'entropy']),
            'max_depth': Integer(self.__max_depth_begin, self.__max_depth_end, prior='uniform'),
            'min_samples_split': Integer(self.__min_samples_split_begin, self.__min_samples_split_end, prior='uniform'),
            'min_samples_leaf': Integer(self.__min_samples_leaf_begin, self.__min_samples_leaf_end, prior='uniform'),
            'max_features': Categorical([None, 'sqrt', 'log2']),
            'max_leaf_nodes': Integer(self.__max_leaf_nodes_begin, self.__max_leaf_nodes_end, prior='uniform'),
            'min_impurity_decrease': Real(self.__min_impurity_decrease_begin,
                                          self.__min_impurity_decrease_end, prior='uniform')
        }

        dt = DecisionTreeClassifier(random_state=42)

        # BayesSearchCV 객체 생성
        bayes_search \
            = BayesSearchCV(dt, search_spaces, n_iter=64, scoring='f1_weighted', cv=self.__cv, verbose=1, random_state=42)

        bayes_search.fit(training_feature_array, training_label_array)

        y_pred = bayes_search.predict(testing_feature_array)
        result = classification_report(testing_label_array, y_pred, zero_division=0, output_dict=True)

        return result, bayes_search.best_params_, bayes_search.best_score_

    def _run_dt_all(self, training_feature_array, training_label_array, testing_feature_array, testing_label_array) \
            -> tuple:
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': self._generate_values(self.__max_depth_begin, self.__max_depth_end, 5, True),
            'min_samples_split': self._generate_values(self.__min_samples_split_begin, self.__min_samples_split_end,
                                                       5, True),
            'min_samples_leaf': self._generate_values(self.__min_samples_leaf_begin, self.__min_samples_leaf_end,
                                                      5, True),
            'max_features': [None, 'sqrt', 'log2'],
            'max_leaf_nodes': self._generate_values(self.__max_leaf_nodes_begin, self.__max_leaf_nodes_end,
                                                    5, True),
            'min_impurity_decrease': self._generate_values(self.__min_impurity_decrease_begin,
                                                           self.__min_impurity_decrease_end, 5, False)
        }

        dt = DecisionTreeClassifier(random_state=42)

        # GridSearchCV 객체 생성
        grid_search = GridSearchCV(dt, param_grid, scoring='f1_weighted', cv=self.__cv, verbose=1)

        grid_search.fit(training_feature_array, training_label_array)

        y_pred = grid_search.predict(testing_feature_array)
        result = classification_report(testing_label_array, y_pred, zero_division=0, output_dict=True)

        return result, grid_search.best_params_, grid_search.best_score_
