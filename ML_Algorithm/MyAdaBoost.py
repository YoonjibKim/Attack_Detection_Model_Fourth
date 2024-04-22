from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


class MyAdaBoost:
    def __init__(self):
        pass

    def _run_ab(self, training_feature_array, training_label_array, testing_feature_array, testing_label_array):
        base_estimator = DecisionTreeClassifier(max_depth=1)
        ada_boost = AdaBoostClassifier(base_estimator=base_estimator, random_state=42)

        # 그리드 서치를 위한 파라미터 그리드 설정
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0]
        }

        # 그리드 서치 객체 생성
        grid_search = GridSearchCV(ada_boost, param_grid, cv=3, scoring='accuracy')

        # 그리드 서치 수행
        grid_search.fit(training_feature_array, training_feature_array)

        # 최적의 파라미터와 최고 점수 출력
        print("Best parameters:", grid_search.best_params_)
        print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

        # 테스트 데이터에 대한 예측 수행
        y_pred = grid_search.predict(testing_feature_array)

        # 성능 평가
        print("Test set accuracy: {:.2f}".format(accuracy_score(testing_label_array, y_pred)))
        print("Classification report:")
        print(classification_report(testing_label_array, y_pred, zero_division=True, output_dict=True))
