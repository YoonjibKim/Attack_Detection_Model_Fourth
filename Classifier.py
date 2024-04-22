import numpy as np
import Constant
from sklearn.preprocessing import StandardScaler
from ML_Algorithm.MyAdaBoost import MyAdaBoost
from ML_Algorithm.MyAgglomerativeClustering import MyAgglomerativeClustering
from ML_Algorithm.MyDNN import MyDNN
from ML_Algorithm.MyDecisionTree import MyDecisionTree
from ML_Algorithm.MyGaussianNaiveBayes import GaussianNaiveBayes
from ML_Algorithm.MyGradientBoost import MyGradientBoost
from ML_Algorithm.MyKMeans import MyKMeans
from ML_Algorithm.MyRandomForest import MyRandomForest
from sklearnex import patch_sklearn

patch_sklearn()


class Classifier(MyAdaBoost, MyAgglomerativeClustering, MyDecisionTree, MyDNN,
                 GaussianNaiveBayes, MyGradientBoost, MyKMeans, MyRandomForest):
    def __init__(self):
        MyAdaBoost.__init__(self)
        MyAgglomerativeClustering.__init__(self)
        MyDecisionTree.__init__(self)
        MyDNN.__init__(self)
        GaussianNaiveBayes.__init__(self)
        MyGradientBoost.__init__(self)
        MyKMeans.__init__(self)
        MyRandomForest.__init__(self)

        self.__scaler = StandardScaler()

    def run_bayesian(self, dataset_dict, ml_type) -> tuple:
        print('ML: ' + ml_type)
        training_feature_array, training_label_array, testing_feature_array, testing_label_array \
            = self.__get_separate_dataset_array(dataset_dict)

        if ml_type == Constant.GridSearch.AB:
            result_dict, grid_params, grid_score \
                = self._run_ab_bayesian(training_feature_array, training_label_array, testing_feature_array,
                                        testing_label_array)
        elif ml_type == Constant.GridSearch.AC:
            result_dict, grid_params, grid_score \
                = self._run_ac_bayesian(testing_feature_array, testing_label_array)
        elif ml_type == Constant.GridSearch.DT:
            result_dict, grid_params, grid_score \
                = self._run_dt_bayesian(training_feature_array, training_label_array, testing_feature_array,
                                        testing_label_array)
        elif ml_type == Constant.GridSearch.GNB:
            result_dict, grid_params, grid_score \
                = self._run_gnb_bayesian(training_feature_array, training_label_array,
                                         testing_feature_array, testing_label_array)
        elif ml_type == Constant.GridSearch.GB:
            result_dict, grid_params, grid_score \
                = self._run_gb_bayesian(training_feature_array, training_label_array, testing_feature_array,
                                        testing_label_array)
        elif ml_type == Constant.GridSearch.KM:
            result_dict, grid_params, grid_score = self._run_km_bayesian(testing_feature_array, testing_label_array)
        elif ml_type == Constant.GridSearch.RF:
            result_dict, grid_params, grid_score \
                = self._run_rf_bayesian(training_feature_array, training_label_array, testing_feature_array,
                                        testing_label_array)
        elif ml_type == Constant.DNN.DNN:  # dnn
            result_dict, grid_params, grid_score = self._run_dnn(training_feature_array, training_label_array,
                                                                 testing_feature_array, testing_label_array)
            grid_params = None
            grid_score = None
        else:
            result_dict = None
            grid_params = None
            grid_score = None

        return result_dict, grid_params, grid_score

    def run_all(self, dataset_dict, ml_type) -> tuple:
        print('ML: ' + ml_type)
        training_feature_array, training_label_array, testing_feature_array, testing_label_array \
            = self.__get_separate_dataset_array(dataset_dict)

        if ml_type == Constant.GridSearch.AB:
            result_dict, grid_params, grid_score = self._run_ab_all(training_feature_array, training_label_array,
                                                                    testing_feature_array, testing_label_array)
        elif ml_type == Constant.GridSearch.AC:
            result_dict, grid_params, grid_score = self._run_ac_all(testing_feature_array, testing_label_array)
        elif ml_type == Constant.GridSearch.DT:
            result_dict, grid_params, grid_score = self._run_dt_all(training_feature_array, training_label_array,
                                                                    testing_feature_array, testing_label_array)
        elif ml_type == Constant.GridSearch.GNB:
            result_dict, grid_params, grid_score = self._run_gnb_all(training_feature_array, training_label_array,
                                                                     testing_feature_array, testing_label_array)
        elif ml_type == Constant.GridSearch.GB:
            result_dict, grid_params, grid_score = self._run_gb_all(training_feature_array, training_label_array,
                                                                    testing_feature_array, testing_label_array)
        elif ml_type == Constant.GridSearch.KM:
            result_dict, grid_params, grid_score = self._run_km_all(testing_feature_array, testing_label_array)
        elif ml_type == Constant.GridSearch.RF:
            result_dict, grid_params, grid_score = self._run_rf_all(training_feature_array, training_label_array,
                                                                    testing_feature_array, testing_label_array)
        elif ml_type == Constant.DNN.DNN:  # dnn
            result_dict = self._run_dnn(training_feature_array, training_label_array,
                                        testing_feature_array, testing_label_array)
            grid_params = None
            grid_score = None
        else:
            result_dict = None
            grid_params = None
            grid_score = None

        return result_dict, grid_params, grid_score

    def __get_separate_dataset_array(self, dataset_dict) -> tuple:
        training_feature_array = np.array(dataset_dict[Constant.TRAINING_FEATURE])
        training_label_array = np.array(dataset_dict[Constant.TRAINING_LABEL])
        testing_feature_array = np.array(dataset_dict[Constant.TESTING_FEATURE])
        testing_label_array = np.array(dataset_dict[Constant.TESTING_LABEL])

        norm_training_feature_array = self.__scaler.fit_transform(training_feature_array)
        norm_testing_feature_array = self.__scaler.fit_transform(testing_feature_array)

        return (norm_training_feature_array, training_label_array.ravel(),
                norm_testing_feature_array, testing_label_array.ravel())
