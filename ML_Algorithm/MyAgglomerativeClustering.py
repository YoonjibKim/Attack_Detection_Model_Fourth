import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import classification_report, silhouette_score
from skopt import gp_minimize
from skopt.space import Categorical
from skopt.utils import use_named_args


class MyAgglomerativeClustering:
    def __init__(self):
        self.__categorical_list = ['ward', 'complete', 'average', 'single']

    def _run_ac_bayesian(self, testing_feature_array, testing_label_array) -> tuple:
        # 베이지안 최적화를 위한 파라미터 공간 정의
        space = [
            Categorical(self.__categorical_list, name='linkage')
        ]

        # 목적 함수 정의
        @use_named_args(space)
        def objective(linkage):
            # 클러스터의 수를 2로 설정하여 모델 생성 및 피팅
            model = AgglomerativeClustering(n_clusters=2, linkage=linkage)
            model.fit(testing_feature_array)

            # 실루엣 점수 계산
            score = silhouette_score(testing_feature_array, model.labels_)

            # 최대화를 위해 점수의 음수를 반환
            return -score

        # 베이지안 최적화 실행
        res_gp = gp_minimize(objective, space, n_calls=64, random_state=0, verbose=1)

        # 최적의 파라미터로 모델 피팅
        best_linkage = res_gp.x[0]
        best_model = AgglomerativeClustering(n_clusters=2, linkage=best_linkage)
        best_model.fit(testing_feature_array)

        # 최적의 실루엣 점수
        best_silhouette_score = -res_gp.fun

        # 클러스터 레이블과 실제 레이블을 사용하여 Classification Report 생성
        cluster_labels = best_model.labels_
        report = classification_report(testing_label_array, cluster_labels, zero_division=0, output_dict=True)

        return report, best_linkage, best_silhouette_score

    def _run_ac_all(self, testing_feature_array, testing_label_array) -> tuple:
        # 가능한 링크 타입들
        linkages = self.__categorical_list

        # 각 링크 타입에 대한 실루엣 점수 저장할 배열
        silhouette_scores = []

        # 그리드 서치 수행
        for linkage in linkages:
            model = AgglomerativeClustering(n_clusters=2, linkage=linkage)
            model.fit(testing_feature_array)
            score = silhouette_score(testing_feature_array, model.labels_)
            silhouette_scores.append(score)

        # 최대 실루엣 점수를 가진 링크 타입 찾기
        best_index = np.argmax(silhouette_scores)
        best_linkage = linkages[best_index]
        best_silhouette_score = silhouette_scores[best_index]

        # 최적의 파라미터로 모델 다시 피팅
        best_model = AgglomerativeClustering(n_clusters=2, linkage=best_linkage)
        best_model.fit(testing_feature_array)

        # 클러스터 레이블과 실제 레이블을 사용하여 Classification Report 생성
        cluster_labels = best_model.labels_
        report = classification_report(testing_label_array, cluster_labels, zero_division=0, output_dict=True)

        return report, best_linkage, best_silhouette_score
