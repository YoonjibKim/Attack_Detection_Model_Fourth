import json
import numpy as np
import Constant
from Classifier import Classifier
from sklearnex import patch_sklearn  # 인텔 최적화 활용

patch_sklearn()  # scikit-learn 코드를 인텔 최적화로 패치


class GridSearch:
    def __init__(self):
        self.__top_dict, self.__std_dict = self.__load_dataset()

    def rum_ml(self):
        classifier = Classifier()

        top_all_dict = {}
        for scenario, dataset in self.__top_dict.items():
            print('TOP ALL Scenario: ' + scenario)

            target_ml = Constant.GridSearch.top_target_ml_dict[scenario]
            result_dict, grid_params, grid_score = classifier.run_all(dataset, target_ml)
            top_all_dict[scenario] = {Constant.CLASSIFICATION_RESULT: result_dict,
                                      Constant.GRID_PARAMETER: grid_params, Constant.GRID_SCORE: grid_score}
            print('TOP ALL Score: ' + str(grid_score))

        with open(Constant.GRID_TOP_ALL_PATH, 'w') as f:
            json.dump(top_all_dict, f, default=self.__default_converter)

        stat_all_dict = {}
        for scenario, dataset in self.__std_dict.items():
            print('STAT ALL Scenario: ' + scenario)

            target_ml = Constant.GridSearch.std_target_ml_dict[scenario]
            result_dict, grid_params, grid_score = classifier.run_all(dataset, target_ml)
            stat_all_dict[scenario] = {Constant.CLASSIFICATION_RESULT: result_dict,
                                       Constant.GRID_PARAMETER: grid_params, Constant.GRID_SCORE: grid_score}
            print('STAT ALL Score: ' + str(grid_score))

        with open(Constant.GRID_STAT_ALL_PATH, 'w') as f:
            json.dump(stat_all_dict, f, default=self.__default_converter)

        top_bayesian_dict = {}
        for scenario, dataset in self.__top_dict.items():
            print('TOP Bayesian Scenario: ' + scenario)

            target_ml = Constant.GridSearch.top_target_ml_dict[scenario]
            result_dict, grid_params, grid_score = classifier.run_bayesian(dataset, target_ml)
            top_bayesian_dict[scenario] = {Constant.CLASSIFICATION_RESULT: result_dict,
                                           Constant.GRID_PARAMETER: grid_params, Constant.GRID_SCORE: grid_score}
            print('TOP Bayesian Score: ' + str(grid_score))

        with open(Constant.GRID_TOP_BAYESIAN_PATH, 'w') as f:
            json.dump(top_bayesian_dict, f, default=self.__default_converter)

        stat_bayesian_dict = {}
        for scenario, dataset in self.__std_dict.items():
            print('STAT Bayesian Scenario: ' + scenario)

            target_ml = Constant.GridSearch.std_target_ml_dict[scenario]
            result_dict, grid_params, grid_score = classifier.run_bayesian(dataset, target_ml)
            stat_bayesian_dict[scenario] = {Constant.CLASSIFICATION_RESULT: result_dict,
                                            Constant.GRID_PARAMETER: grid_params, Constant.GRID_SCORE: grid_score}
            print('STAT Bayesian Score: ' + str(grid_score))

        with open(Constant.GRID_STAT_BAYESIAN_PATH, 'w') as f:
            json.dump(stat_bayesian_dict, f, default=self.__default_converter)

    @classmethod
    def __default_converter(cls, o):
        if isinstance(o, np.int64):
            return int(o)  # or str(o) if you prefer to convert it to string
        raise TypeError

    @classmethod
    def __load_dataset(cls) -> tuple:
        target_top_path_dict = Constant.GridSearch.top_target_path_dict

        top_dict = {}
        for name, path in target_top_path_dict.items():
            with open(path, 'r') as f:
                temp_dict = json.load(f)
                tokens = name.split('_')
                category = Constant.GridSearch.category_dict[tokens[4]]
                additional_tokens = tokens[5:] if len(tokens) > 4 else []
                full_name_token_list = []
                for token in additional_tokens:
                    full_name_token_list.append(Constant.GridSearch.symbol_dict[token])
                type_dict = temp_dict[category]
                key = str(full_name_token_list)
                target_size = Constant.GridSearch.top_target_size_dict[name]

                for type_name, category_dict in type_dict.items():
                    if key in category_dict:
                        training_list = category_dict[key][Constant.TESTING_FEATURE]
                        training_size = len(training_list)
                        if target_size == training_size:
                            valid_type = type_name
                            break

                valid_category_dict = type_dict[valid_type][key]
                valid_category_dict.pop(Constant.CSR, None)
                top_dict[name] = valid_category_dict

        target_std_path_dict = Constant.GridSearch.std_target_path_dict

        std_dict = {}
        for name, path in target_std_path_dict.items():
            with open(path, 'r') as f:
                temp_dict = json.load(f)
                token = name.split('_')[4]
                category = Constant.GridSearch.category_dict[token]
                target_list = temp_dict[category]
                std_dict[name] = target_list

        return top_dict, std_dict
