import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from ML_Algorithm.Util import Util


class MyKMeans(Util):
    def __init__(self):
        self.__n_clusters_begin = 2
        self.__n_clusters_end = 3
        self.__n_init_begin = 5
        self.__n_init_end = 25
        self.__max_iter_begin = 100
        self.__max_iter_end = 500
        self.__tol_begin = 1e-5
        self.__tol_end = 1e-1

    def _run_km_bayesian(self, testing_feature_array, testing_label_array) -> tuple:
        # 베이지안 최적화를 위한 파라미터 스페이스 정의
        space = [
            Integer(self.__n_clusters_begin, self.__n_clusters_end, name='n_clusters'),  # 클러스터 수 범위
            Integer(self.__n_init_begin, self.__n_init_end, name='n_init'),  # 초기화 횟수 범위
            Integer(self.__max_iter_begin, self.__max_iter_end, name='max_iter'),  # 최대 반복 횟수 범위
            Real(self.__tol_begin, self.__tol_end, prior='log-uniform', name='tol')  # 수렴 허용오차
        ]

        # 최적화 목적 함수 정의
        @use_named_args(space)
        def objective(n_clusters, n_init, max_iter, tol):
            model = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, tol=tol, random_state=42)
            model.fit(testing_feature_array)
            labels = model.labels_
            # 실루엣 스코어를 계산
            score = silhouette_score(testing_feature_array, labels)
            return -score  # 최대화를 위해 부호 반전

        # 베이지안 최적화 실행
        result = gp_minimize(objective, space, n_calls=64, random_state=42)

        # 최적의 파라미터로 모델 재구성 및 클러스터링 실행
        best_model \
            = KMeans(n_clusters=result.x[0], n_init=result.x[1], max_iter=result.x[2], tol=result.x[3], random_state=42)
        best_model.fit(testing_feature_array)
        best_labels = best_model.labels_

        # 최종 보고서 생성
        report = classification_report(testing_label_array, best_labels, zero_division=0, output_dict=True)
        best_params = {'n_clusters': result.x[0], 'n_init': result.x[1], 'max_iter': result.x[2], 'tol': result.x[3]}
        best_score = -result.fun

        return report, best_params, best_score

    def _run_km_all(self, testing_feature_array, testing_label_array) -> tuple:
        n_clusters_range = self._generate_values(self.__n_clusters_begin, self.__n_clusters_end, 2, True)
        n_init_range = self._generate_values(self.__n_init_begin, self.__n_init_end, 5, True)
        max_iter_range = self._generate_values(self.__max_iter_begin, self.__max_iter_end, 5, True)
        tol_range = self._generate_values(self.__tol_begin, self.__tol_end, 5, False)

        best_score = -1
        best_params = {}
        best_labels = None

        # 그리드 서치 수행
        for n_clusters in n_clusters_range:
            for n_init in n_init_range:
                for max_iter in max_iter_range:
                    for tol in tol_range:
                        model \
                            = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, tol=tol, random_state=42)
                        model.fit(testing_feature_array)
                        labels = model.labels_
                        score = silhouette_score(testing_feature_array, labels)

                        if score > best_score:
                            best_score = score
                            best_params = {
                                'n_clusters': n_clusters,
                                'n_init': n_init,
                                'max_iter': max_iter,
                                'tol': tol
                            }
                            best_labels = labels

        best_model = KMeans(**best_params, random_state=42)
        best_model.fit(testing_feature_array)

        report = classification_report(testing_label_array, best_labels, zero_division=0, output_dict=True)

        return report, best_params, best_score