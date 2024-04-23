import copy
import json
import numpy as np
import pandas as pd
import Constant
from sklearn.model_selection import train_test_split
from ML_Algorithm.MyDNN import MyDNN


class DNN(MyDNN):
    def __init__(self):
        MyDNN.__init__(self)
        # self.__generate_dataset()
        self.__dataset_dict, self.__stat_cycle_dict = self.__load_dataset()

    def run(self):
        cs_result_dict = {}
        gs_result_dict = {}
        cs_stat_cycle_dict = {}
        gs_stat_cycle_dict = {}

        for scenario, dataset_dict in self.__dataset_dict.items():
            print(scenario)

            cs_tr_data_array = dataset_dict[Constant.DNN.CS_TR_DATA]
            cs_tr_label_array = dataset_dict[Constant.DNN.CS_TR_LABEL]
            cs_te_data_array = dataset_dict[Constant.DNN.CS_TE_DATA]
            cs_te_label_array = dataset_dict[Constant.DNN.CS_TE_LABEL]

            cs_stat_cycle_data_df = self.__stat_cycle_dict[scenario][Constant.CS_STAT_CYCLE_DATA]
            cs_stat_cycle_label_df = self.__stat_cycle_dict[scenario][Constant.CS_STAT_CYCLE_LABEL]

            cs_fig_save_path = Constant.DNN.CS_FIG_PATH + '_' + scenario + '.png'
            param_cs_result_dict, cs_stat_cycle_result = self._run_dnn(cs_tr_data_array, cs_tr_label_array,
                                                                       cs_te_data_array, cs_te_label_array,
                                                                       cs_fig_save_path, cs_stat_cycle_data_df,
                                                                       cs_stat_cycle_label_df)
            cs_result_dict[scenario] = param_cs_result_dict
            cs_stat_cycle_dict[scenario] = cs_stat_cycle_result

            gs_tr_data_array = dataset_dict[Constant.DNN.GS_TR_DATA]
            gs_tr_label_array = dataset_dict[Constant.DNN.GS_TR_LABEL]
            gs_te_data_array = dataset_dict[Constant.DNN.GS_TE_DATA]
            gs_te_label_array = dataset_dict[Constant.DNN.GS_TE_LABEL]

            gs_stat_cycle_data_df = self.__stat_cycle_dict[scenario][Constant.GS_STAT_CYCLE_DATA]
            gs_stat_cycle_label_df = self.__stat_cycle_dict[scenario][Constant.GS_STAT_CYCLE_LABEL]

            gs_fig_save_path = Constant.DNN.GS_FIG_PATH + '_' + scenario + '.png'
            param_gs_result_dict, gs_stat_cycle_result = self._run_dnn(gs_tr_data_array, gs_tr_label_array,
                                                                       gs_te_data_array, gs_te_label_array,
                                                                       gs_fig_save_path, gs_stat_cycle_data_df,
                                                                       gs_stat_cycle_label_df)
            gs_result_dict[scenario] = param_gs_result_dict
            gs_stat_cycle_dict[scenario] = gs_stat_cycle_result

        with open(Constant.DNN.CS_RESULT_PATH, 'w') as f:
            json.dump(cs_result_dict, f)
        with open(Constant.DNN.GS_RESULT_PATH, 'w') as f:
            json.dump(gs_result_dict, f)
        with open(Constant.DNN.CS_STAT_CYCLE_RESULT_PATH, 'w') as f:
            json.dump(cs_stat_cycle_dict, f)
        with open(Constant.DNN.GS_STAT_CYCLE_RESULT_PATH, 'w') as f:
            json.dump(gs_stat_cycle_dict, f)

    @classmethod
    def __load_dataset(cls) -> tuple:
        dataset_dict = {}
        stat_cycle_dict = {}

        for scenario in Constant.RAW_DATASET_PATH_DICT.keys():
            cs_tr_data_path = Constant.DNN.CS_TR_DATA_PATH + '_' + scenario + '.csv'
            cs_tr_label_path = Constant.DNN.CS_TR_LABEL_PATH + '_' + scenario + '.csv'
            cs_te_data_path = Constant.DNN.CS_TE_DATA_PATH + '_' + scenario + '.csv'
            cs_te_label_path = Constant.DNN.CS_TE_LABEL_PATH + '_' + scenario + '.csv'
            gs_tr_data_path = Constant.DNN.GS_TR_DATA_PATH + '_' + scenario + '.csv'
            gs_tr_label_path = Constant.DNN.GS_TR_LABEL_PATH + '_' + scenario + '.csv'
            gs_te_data_path = Constant.DNN.GS_TE_DATA_PATH + '_' + scenario + '.csv'
            gs_te_label_path = Constant.DNN.GS_TE_LABEL_PATH + '_' + scenario + '.csv'

            cs_tr_data_df = pd.read_csv(cs_tr_data_path)
            cs_tr_label_df = pd.read_csv(cs_tr_label_path)
            cs_te_data_df = pd.read_csv(cs_te_data_path)
            cs_te_label_df = pd.read_csv(cs_te_label_path)
            gs_tr_data_df = pd.read_csv(gs_tr_data_path)
            gs_tr_label_df = pd.read_csv(gs_tr_label_path)
            gs_te_data_df = pd.read_csv(gs_te_data_path)
            gs_te_label_df = pd.read_csv(gs_te_label_path)

            dataset_dict[scenario] = {Constant.DNN.CS_TR_DATA: cs_tr_data_df,
                                      Constant.DNN.CS_TR_LABEL: cs_tr_label_df,
                                      Constant.DNN.CS_TE_DATA: cs_te_data_df,
                                      Constant.DNN.CS_TE_LABEL: cs_te_label_df,
                                      Constant.DNN.GS_TR_DATA: gs_tr_data_df,
                                      Constant.DNN.GS_TR_LABEL: gs_tr_label_df,
                                      Constant.DNN.GS_TE_DATA: gs_te_data_df,
                                      Constant.DNN.GS_TE_LABEL: gs_te_label_df}

            cs_stat_cycle_data_path = Constant.CS_STAT_CYCLE_DATA_PATH + '_' + scenario + '.csv'
            cs_stat_cycle_label_path = Constant.CS_STAT_CYCLE_LABEL_PATH + '_' + scenario + '.csv'
            gs_stat_cycle_data_path = Constant.GS_STAT_CYCLE_DATA_PATH + '_' + scenario + '.csv'
            gs_stat_cycle_label_path = Constant.GS_STAT_CYCLE_LABEL_PATH + '_' + scenario + '.csv'

            cs_stat_cycle_data_df = pd.read_csv(cs_stat_cycle_data_path)
            cs_stat_cycle_label_df = pd.read_csv(cs_stat_cycle_label_path)
            gs_stat_cycle_data_df = pd.read_csv(gs_stat_cycle_data_path)
            gs_stat_cycle_label_df = pd.read_csv(gs_stat_cycle_label_path)

            stat_cycle_dict[scenario] = {Constant.CS_STAT_CYCLE_DATA: cs_stat_cycle_data_df,
                                         Constant.CS_STAT_CYCLE_LABEL: cs_stat_cycle_label_df,
                                         Constant.GS_STAT_CYCLE_DATA: gs_stat_cycle_data_df,
                                         Constant.GS_STAT_CYCLE_LABEL: gs_stat_cycle_label_df}

        return dataset_dict, stat_cycle_dict

    def __generate_dataset(self):
        for scenario, path in Constant.RAW_DATASET_PATH_DICT.items():
            print(scenario)

            cs_stat_path = path + '/' + Constant.STAT + '/' + Constant.CS + '/' + Constant.STAT + '.json'
            gs_stat_path = path + '/' + Constant.STAT + '/' + Constant.GS + '/' + Constant.STAT + '.json'

            cs_top_path = path + '/' + Constant.TOP + '/' + Constant.CS + '/' + Constant.TOP + '.json'
            gs_top_path = path + '/' + Constant.TOP + '/' + Constant.GS + '/' + Constant.TOP + '.json'

            time_delta_path = path + '/' + Constant.STAT + '/' + Constant.CS + '/' + Constant.TIME_DELTA + '.json'

            with open(cs_stat_path, 'r') as f:
                cs_stat_scenario_dict = json.load(f)
            with open(gs_stat_path, 'r') as f:
                gs_stat_scenario_dict = json.load(f)
            with open(cs_top_path, 'r') as f:
                cs_top_scenario_dict = json.load(f)
            with open(gs_top_path, 'r') as f:
                gs_top_scenario_dict = json.load(f)
            with open(time_delta_path, 'r') as f:
                cs_time_delta_scenario_dict = json.load(f)

            def __find_lowest_lists_1(lst, collector):
                if all(isinstance(x, list) for x in lst):
                    for sublst in lst:
                        __find_lowest_lists_1(sublst, collector)
                else:
                    collector.append(lst)

            def __calculate_average_size(lists):
                valid_lists = [lst for lst in lists if any(x != 0 for x in lst)]
                if valid_lists:
                    average_size = int(np.mean([len(lst) for lst in valid_lists]))
                else:
                    average_size = 0
                return average_size

            def __resize_list(lst, target_length):
                if target_length == 0 or len(lst) == 0:
                    return [0] * target_length
                elif len(lst) == target_length:
                    return lst
                else:
                    return list(np.interp(np.linspace(0, len(lst) - 1, num=target_length), np.arange(len(lst)), lst))

            def __find_and_resize_lowest_lists(lst, average_size):
                if all(isinstance(x, list) for x in lst):
                    return [__find_and_resize_lowest_lists(sublst, average_size) for sublst in lst]
                else:
                    if all(x == 0 for x in lst):
                        return [0] * average_size
                    else:
                        return __resize_list(lst, average_size)

            def __standardize_list_sizes(data):
                lowest_lists = []
                __find_lowest_lists_1(data, lowest_lists)

                average_size = __calculate_average_size(lowest_lists)

                return __find_and_resize_lowest_lists(data, average_size)

            def __extract_lowest_lists(lst):
                collector = []

                def __find_lowest_lists_2(sublst):
                    if isinstance(sublst, list) and any(isinstance(x, list) for x in sublst):
                        for item in sublst:
                            __find_lowest_lists_2(item)
                    else:
                        collector.append(sublst)

                __find_lowest_lists_2(lst)
                return collector

            (adjusted_cs_normal_top_branch_list, adjusted_cs_attack_top_branch_list, adjusted_cs_normal_top_cycle_list,
             adjusted_cs_attack_top_cycle_list, adjusted_cs_normal_top_instruction_list,
             adjusted_cs_attack_top_instruction_list, adjusted_gs_normal_top_branch_list,
             adjusted_gs_attack_top_branch_list, adjusted_gs_normal_top_cycle_list,
             adjusted_gs_attack_top_cycle_list, adjusted_gs_normal_top_instruction_list,
             adjusted_gs_attack_top_instruction_list) = self.__get_top_balanced_list(cs_top_scenario_dict,
                                                                                     gs_top_scenario_dict)

            adjusted_normal_td_list, adjusted_attack_td_list \
                = self.__get_time_delta_balanced_list(cs_time_delta_scenario_dict)

            (adjusted_cs_normal_stat_branch_list, adjusted_cs_attack_stat_branch_list,
             adjusted_cs_normal_stat_cycle_list, adjusted_cs_attack_stat_cycle_list,
             adjusted_cs_normal_stat_instruction_list, adjusted_cs_attack_stat_instruction_list,
             adjusted_gs_normal_stat_branch_list, adjusted_gs_attack_stat_branch_list,
             adjusted_gs_normal_stat_cycle_list, adjusted_gs_attack_stat_cycle_list,
             adjusted_gs_normal_stat_instruction_list, adjusted_gs_attack_stat_instruction_list, stat_cycle_dict) \
                = self.__get_stat_balanced_list(cs_stat_scenario_dict, gs_stat_scenario_dict)

            temp_all_cs_normal_combination_list \
                = __standardize_list_sizes([adjusted_cs_normal_stat_branch_list, adjusted_cs_normal_stat_cycle_list,
                                            adjusted_cs_normal_stat_instruction_list,
                                            adjusted_cs_normal_top_branch_list, adjusted_cs_normal_top_cycle_list,
                                            adjusted_cs_normal_top_instruction_list, adjusted_normal_td_list])
            temp_all_cs_attack_combination_list \
                = __standardize_list_sizes([adjusted_cs_attack_stat_branch_list, adjusted_cs_attack_stat_cycle_list,
                                            adjusted_cs_attack_stat_instruction_list,
                                            adjusted_cs_attack_top_branch_list, adjusted_cs_attack_top_cycle_list,
                                            adjusted_cs_attack_top_instruction_list, adjusted_attack_td_list])
            temp_all_gs_normal_combination_list \
                = __standardize_list_sizes([adjusted_gs_normal_stat_branch_list, adjusted_gs_normal_stat_cycle_list,
                                            adjusted_gs_normal_stat_instruction_list,
                                            adjusted_gs_normal_top_branch_list, adjusted_gs_normal_top_cycle_list,
                                            adjusted_gs_normal_top_instruction_list, adjusted_normal_td_list])
            temp_all_gs_attack_combination_list \
                = __standardize_list_sizes([adjusted_gs_attack_stat_branch_list, adjusted_gs_attack_stat_cycle_list,
                                            adjusted_gs_attack_stat_instruction_list,
                                            adjusted_gs_attack_top_branch_list, adjusted_gs_attack_top_cycle_list,
                                            adjusted_gs_attack_top_instruction_list, adjusted_attack_td_list])

            all_cs_normal_combination_data_list = __extract_lowest_lists(temp_all_cs_normal_combination_list)
            all_cs_attack_combination_data_list = __extract_lowest_lists(temp_all_cs_attack_combination_list)
            all_gs_normal_combination_data_list = __extract_lowest_lists(temp_all_gs_normal_combination_list)
            all_gs_attack_combination_data_list = __extract_lowest_lists(temp_all_gs_attack_combination_list)

            all_cs_normal_combination_label_list = [Constant.NORMAL_LABEL] * len(all_cs_normal_combination_data_list[0])
            all_cs_attack_combination_label_list = [Constant.ATTACK_LABEL] * len(all_cs_attack_combination_data_list[0])
            all_gs_normal_combination_label_list = [Constant.NORMAL_LABEL] * len(all_gs_normal_combination_data_list[0])
            all_gs_attack_combination_label_list = [Constant.ATTACK_LABEL] * len(all_gs_attack_combination_data_list[0])

            all_cs_normal_combination_data_array = np.array(all_cs_normal_combination_data_list)
            all_cs_normal_combination_data_array = all_cs_normal_combination_data_array.T
            all_cs_attack_combination_data_array = np.array(all_cs_attack_combination_data_list)
            all_cs_attack_combination_data_array = all_cs_attack_combination_data_array.T
            all_gs_normal_combination_data_array = np.array(all_gs_normal_combination_data_list)
            all_gs_normal_combination_data_array = all_gs_normal_combination_data_array.T
            all_gs_attack_combination_data_array = np.array(all_gs_attack_combination_data_list)
            all_gs_attack_combination_data_array = all_gs_attack_combination_data_array.T

            all_cs_normal_combination_label_list.extend(all_cs_attack_combination_label_list)
            all_gs_normal_combination_label_list.extend(all_gs_attack_combination_label_list)

            cs_data_array = np.vstack((all_cs_normal_combination_data_array, all_cs_attack_combination_data_array))
            gs_data_array = np.vstack((all_gs_normal_combination_data_array, all_gs_attack_combination_data_array))
            cs_label_array = np.array(all_cs_normal_combination_label_list)
            gs_label_array = np.array(all_gs_normal_combination_label_list)

            cs_X_train, cs_X_test, cs_y_train, cs_y_test \
                = train_test_split(cs_data_array, cs_label_array, test_size=0.2, random_state=42)
            gs_X_train, gs_X_test, gs_y_train, gs_y_test \
                = train_test_split(gs_data_array, gs_label_array, test_size=0.2, random_state=42)

            target_cs_cycle_list = cs_X_train[:, 3].tolist()
            target_gs_cycle_list = gs_X_train[:, 3].tolist()

            ref_cs_attack_cycle_list = stat_cycle_dict[Constant.CS_CYCLE_ATTACK]
            ref_cs_normal_cycle_list = stat_cycle_dict[Constant.CS_CYCLE_NORMAL]
            ref_gs_attack_cycle_list = stat_cycle_dict[Constant.GS_CYCLE_ATTACK]
            ref_gs_normal_cycle_list = stat_cycle_dict[Constant.GS_CYCLE_NORMAL]

            ref_cs_attack_cycle_label_list = [Constant.ATTACK_LABEL] * len(ref_cs_attack_cycle_list)
            ref_cs_normal_cycle_label_list = [Constant.NORMAL_LABEL] * len(ref_cs_normal_cycle_list)
            ref_gs_attack_cycle_label_list = [Constant.ATTACK_LABEL] * len(ref_gs_attack_cycle_list)
            ref_gs_normal_cycle_label_list = [Constant.NORMAL_LABEL] * len(ref_gs_normal_cycle_list)

            ref_cs_attack_cycle_array = np.array([ref_cs_attack_cycle_list, ref_cs_attack_cycle_label_list])
            ref_cs_normal_cycle_array = np.array([ref_cs_normal_cycle_list, ref_cs_normal_cycle_label_list])
            ref_gs_attack_cycle_array = np.array([ref_gs_attack_cycle_list, ref_gs_attack_cycle_label_list])
            ref_gs_normal_cycle_array = np.array([ref_gs_normal_cycle_list, ref_gs_normal_cycle_label_list])

            ref_cs_cycle_array = np.concatenate((ref_cs_normal_cycle_array, ref_cs_attack_cycle_array), axis=1)
            ref_gs_cycle_array = np.concatenate((ref_gs_normal_cycle_array, ref_gs_attack_cycle_array), axis=1)

            cs_removal_indices \
                = np.any([np.isclose(ref_cs_cycle_array[0, :], value) for value in target_cs_cycle_list], axis=0)
            unique_cs_cycle_array = ref_cs_cycle_array[:, ~cs_removal_indices]

            gs_removal_indices \
                = np.any([np.isclose(ref_gs_cycle_array[0, :], value) for value in target_gs_cycle_list], axis=0)
            unique_gs_cycle_array = ref_gs_cycle_array[:, ~gs_removal_indices]

            unique_cs_cycle_data_array = unique_cs_cycle_array[0]
            unique_cs_cycle_label_array = unique_cs_cycle_array[1]
            unique_gs_cycle_data_array = unique_gs_cycle_array[0]
            unique_gs_cycle_label_array = unique_gs_cycle_array[1]

            padded_unique_cs_cycle_data_array = np.zeros((unique_cs_cycle_data_array.size, cs_X_test.shape[1]))
            padded_unique_cs_cycle_data_array[:, 3] = unique_cs_cycle_data_array
            padded_unique_gs_cycle_data_array = np.zeros((unique_gs_cycle_data_array.size, gs_X_test.shape[1]))
            padded_unique_gs_cycle_data_array[:, 3] = unique_gs_cycle_data_array

            unique_cs_cycle_data_df = pd.DataFrame(padded_unique_cs_cycle_data_array)
            unique_cs_cycle_data_df.to_csv(Constant.CS_STAT_CYCLE_DATA_PATH + '_' + scenario + '.csv', index=False)
            cs_cycle_label_df = pd.DataFrame(unique_cs_cycle_label_array)
            cs_cycle_label_df.to_csv(Constant.CS_STAT_CYCLE_LABEL_PATH + '_' + scenario + '.csv', index=False)

            unique_gs_cycle_data_df = pd.DataFrame(padded_unique_gs_cycle_data_array)
            unique_gs_cycle_data_df.to_csv(Constant.GS_STAT_CYCLE_DATA_PATH + '_' + scenario + '.csv', index=False)
            gs_cycle_label_df = pd.DataFrame(unique_gs_cycle_label_array)
            gs_cycle_label_df.to_csv(Constant.GS_STAT_CYCLE_LABEL_PATH + '_' + scenario + '.csv', index=False)

            df_cs_X_train = pd.DataFrame(cs_X_train)
            df_cs_X_train.to_csv(Constant.DNN.CS_TR_DATA_PATH + '_' + scenario + '.csv', index=False)
            df_cs_y_train = pd.DataFrame(cs_y_train)
            df_cs_y_train.to_csv(Constant.DNN.CS_TR_LABEL_PATH + '_' + scenario + '.csv', index=False)

            df_cs_X_test = pd.DataFrame(cs_X_test)
            df_cs_X_test.to_csv(Constant.DNN.CS_TE_DATA_PATH + '_' + scenario + '.csv', index=False)
            df_cs_y_test = pd.DataFrame(cs_y_test)
            df_cs_y_test.to_csv(Constant.DNN.CS_TE_LABEL_PATH + '_' + scenario + '.csv', index=False)

            df_gs_X_train = pd.DataFrame(gs_X_train)
            df_gs_X_train.to_csv(Constant.DNN.GS_TR_DATA_PATH + '_' + scenario + '.csv', index=False)
            df_gs_y_train = pd.DataFrame(gs_y_train)
            df_gs_y_train.to_csv(Constant.DNN.GS_TR_LABEL_PATH + '_' + scenario + '.csv', index=False)

            df_gs_X_test = pd.DataFrame(gs_X_test)
            df_gs_X_test.to_csv(Constant.DNN.GS_TE_DATA_PATH + '_' + scenario + '.csv', index=False)
            df_gs_y_test = pd.DataFrame(gs_y_test)
            df_gs_y_test.to_csv(Constant.DNN.GS_TE_LABEL_PATH + '_' + scenario + '.csv', index=False)

    @classmethod
    def __split_dictionary(cls, data_dict):
        return [data_dict[key] for key in data_dict]

    def __get_top_balanced_list(self, cs_top_dict, gs_top_dict):
        temp_list = self.__split_dictionary(cs_top_dict)
        gs_top_dict = gs_top_dict[Constant.GS_ID]

        def __sync_dictionaries(param_attack_dict, param_normal_dict) -> tuple:
            temp_attack_dict = copy.deepcopy(param_attack_dict)
            temp_normal_dict = copy.deepcopy(param_normal_dict)

            for key in temp_attack_dict:
                if key not in temp_normal_dict:
                    temp_normal_dict[key] = [0.0] * len(temp_attack_dict[key][Constant.DATA_POINT])
                else:
                    temp_normal_dict[key] = temp_attack_dict[key][Constant.DATA_POINT]

            for key in temp_normal_dict:
                if key not in temp_attack_dict:
                    temp_attack_dict[key] = [0.0] * len(temp_normal_dict[key][Constant.DATA_POINT])
                else:
                    temp_attack_dict[key] = temp_attack_dict[key][Constant.DATA_POINT]

            return temp_attack_dict, temp_normal_dict

        def __reorder_dictionaries(dict1, dict2) -> tuple:
            common_keys = set(dict1.keys()) & set(dict2.keys())

            sorted_keys = sorted(common_keys)

            new_dict1 = {key: dict1[key] for key in sorted_keys}
            new_dict2 = {key: dict2[key] for key in sorted_keys}

            return new_dict1, new_dict2

        def __get_adjusted_top_list(param_category_dict):
            attack_dict = param_category_dict[Constant.ATTACK]
            normal_dict = param_category_dict[Constant.NORMAL]

            padded_attack_dict, padded_normal_dict = __sync_dictionaries(attack_dict, normal_dict)

            reordered_attack_dict, reordered_normal_dict \
                = __reorder_dictionaries(padded_attack_dict, padded_normal_dict)

            attack_list = list(reordered_attack_dict.values())
            normal_list = list(reordered_normal_dict.values())

            adjusted_attack_list = self.__adjust_list_sizes(attack_list)
            adjusted_normal_list = self.__adjust_list_sizes(normal_list)

            return adjusted_normal_list, adjusted_attack_list

        def __resize_multiple_lists(*data_groups, default_length=5):
            all_lengths = [len(sublist) for data in data_groups for group in data for sublist in group if
                           not all(v == 0 for v in sublist)]

            average_length = np.mean(all_lengths) if all_lengths else default_length

            new_data_groups = []

            for data in data_groups:
                new_data = []
                for group in data:
                    new_group = []
                    for sublist in group:
                        current_length = len(sublist)
                        if current_length == 0 or all(v == 0 for v in sublist):
                            new_sublist = [0] * int(average_length)
                        else:
                            new_indices = np.linspace(0, current_length - 1, int(average_length))
                            new_sublist = np.interp(new_indices, np.arange(current_length), sublist).tolist()
                        new_group.append(new_sublist)
                    new_data.append(new_group)
                new_data_groups.append(new_data)

            return new_data_groups

        param_branch_attack_list = []
        param_branch_normal_list = []
        param_cycle_attack_list = []
        param_cycle_normal_list = []
        param_instruction_attack_list = []
        param_instruction_normal_list = []

        for cs_station_dict in temp_list:
            branch_dict = cs_station_dict[Constant.BRANCH]
            cycle_dict = cs_station_dict[Constant.CYCLES]
            instruction_dict = cs_station_dict[Constant.INSTRUCTIONS]

            adjusted_branch_attack_list, adjusted_branch_normal_list = __get_adjusted_top_list(branch_dict)
            adjusted_cycle_attack_list, adjusted_cycle_normal_list = __get_adjusted_top_list(cycle_dict)
            adjusted_instruction_attack_list, adjusted_instruction_normal_list \
                = __get_adjusted_top_list(instruction_dict)

            param_branch_attack_list.append(adjusted_branch_attack_list)
            param_branch_normal_list.append(adjusted_branch_normal_list)
            param_cycle_attack_list.append(adjusted_cycle_attack_list)
            param_cycle_normal_list.append(adjusted_cycle_normal_list)
            param_instruction_attack_list.append(adjusted_instruction_attack_list)
            param_instruction_normal_list.append(adjusted_instruction_normal_list)

        temp_cs_attack_list \
            = __resize_multiple_lists(param_branch_attack_list, param_cycle_attack_list, param_instruction_attack_list)
        temp_cs_normal_list \
            = __resize_multiple_lists(param_branch_normal_list, param_cycle_normal_list, param_instruction_normal_list)

        adjusted_cs_branch_attack_list = temp_cs_attack_list[0]
        adjusted_cs_cycle_attack_list = temp_cs_attack_list[1]
        adjusted_cs_instruction_attack_list = temp_cs_attack_list[2]
        adjusted_cs_branch_normal_list = temp_cs_normal_list[0]
        adjusted_cs_cycle_normal_list = temp_cs_normal_list[1]
        adjusted_cs_instruction_normal_list = temp_cs_normal_list[2]

        param_gs_attack_branch_list, param_gs_normal_branch_list \
            = __get_adjusted_top_list(gs_top_dict[Constant.BRANCH])
        param_gs_attack_cycle_list, param_gs_normal_cycle_list \
            = __get_adjusted_top_list(gs_top_dict[Constant.CYCLES])
        param_gs_attack_instruction_list, param_gs_normal_instruction_list \
            = __get_adjusted_top_list(gs_top_dict[Constant.INSTRUCTIONS])

        temp_gs_attack_list = __resize_multiple_lists([param_gs_attack_branch_list], [param_gs_attack_cycle_list],
                                                      [param_gs_attack_instruction_list])
        temp_gs_normal_list = __resize_multiple_lists([param_gs_normal_branch_list], [param_gs_normal_cycle_list],
                                                      [param_gs_normal_instruction_list])

        adjusted_gs_attack_branch_list = temp_gs_attack_list[0][0]
        adjusted_gs_attack_cycle_list = temp_gs_attack_list[1][0]
        adjusted_gs_attack_instruction_list = temp_gs_attack_list[2][0]

        adjusted_gs_normal_branch_list = temp_gs_normal_list[0][0]
        adjusted_gs_normal_cycle_list = temp_gs_normal_list[1][0]
        adjusted_gs_normal_instruction_list = temp_gs_normal_list[2][0]

        return (adjusted_cs_branch_normal_list, adjusted_cs_branch_attack_list, adjusted_cs_cycle_normal_list,
                adjusted_cs_cycle_attack_list, adjusted_cs_instruction_normal_list, adjusted_cs_instruction_attack_list,
                adjusted_gs_normal_branch_list, adjusted_gs_attack_branch_list, adjusted_gs_normal_cycle_list,
                adjusted_gs_attack_cycle_list, adjusted_gs_normal_instruction_list, adjusted_gs_attack_instruction_list)

    @classmethod
    def __get_time_delta_balanced_list(cls, time_delta_dict) -> tuple:
        temp_list = cls.__split_dictionary(time_delta_dict)
        normal_cs_list = []
        attack_cs_list = []

        for cs_data_dict in temp_list:
            temp_normal_cs_list = cs_data_dict[Constant.TIME_DELTA][Constant.NORMAL][Constant.DATA_POINT]
            temp_attack_cs_list = cs_data_dict[Constant.TIME_DELTA][Constant.ATTACK][Constant.DATA_POINT]

            normal_cs_list.append(temp_normal_cs_list)
            attack_cs_list.append(temp_attack_cs_list)

        adjusted_normal_cs_list = cls.__adjust_list_sizes(normal_cs_list)
        adjusted_attack_cs_list = cls.__adjust_list_sizes(attack_cs_list)

        return adjusted_normal_cs_list, adjusted_attack_cs_list

    @classmethod
    def __get_stat_balanced_list(cls, cs_stat_dict, gs_stat_dict) -> tuple:
        temp_list = cls.__split_dictionary(cs_stat_dict)
        gs_dict = gs_stat_dict[Constant.GS_ID]

        def __get_category_data_list(stat_dict) -> tuple:
            branch_dict = stat_dict[Constant.BRANCH]
            cycle_dict = stat_dict[Constant.CYCLES]
            instruction_dict = stat_dict[Constant.INSTRUCTIONS]

            __normal_branch_list = branch_dict.get(Constant.NORMAL, {}).get(Constant.DATA_POINT, None)
            __attack_branch_list = branch_dict.get(Constant.ATTACK, {}).get(Constant.DATA_POINT, None)
            __normal_cycle_list = cycle_dict.get(Constant.NORMAL, {}).get(Constant.DATA_POINT, None)
            __attack_cycle_list = cycle_dict.get(Constant.ATTACK, {}).get(Constant.DATA_POINT, None)
            __normal_instruction_list = instruction_dict.get(Constant.NORMAL, {}).get(Constant.DATA_POINT, None)
            __attack_instruction_list = instruction_dict.get(Constant.ATTACK, {}).get(Constant.DATA_POINT, None)

            return (__normal_branch_list, __attack_branch_list, __normal_cycle_list,
                    __attack_cycle_list, __normal_instruction_list, __attack_instruction_list)

        cs_normal_branch_list = []
        cs_attack_branch_list = []
        cs_normal_cycle_list = []
        cs_attack_cycle_list = []
        cs_normal_instruction_list = []
        cs_attack_instruction_list = []

        for cs_data_dict in temp_list:
            (temp_normal_branch_list, temp_attack_branch_list, temp_normal_cycle_list,
             temp_attack_cycle_list, temp_normal_instruction_list, temp_attack_instruction_list) \
                = __get_category_data_list(cs_data_dict)

            cs_normal_branch_list.append(temp_normal_branch_list)
            cs_attack_branch_list.append(temp_attack_branch_list)
            cs_normal_cycle_list.append(temp_normal_cycle_list)
            cs_attack_cycle_list.append(temp_attack_cycle_list)
            cs_normal_instruction_list.append(temp_normal_instruction_list)
            cs_attack_instruction_list.append(temp_attack_instruction_list)

        adjusted_cs_normal_branch_list = cls.__adjust_list_sizes(cs_normal_branch_list)
        adjusted_cs_attack_branch_list = cls.__adjust_list_sizes(cs_attack_branch_list)
        adjusted_cs_normal_cycle_list = cls.__adjust_list_sizes(cs_normal_cycle_list)
        adjusted_cs_attack_cycle_list = cls.__adjust_list_sizes(cs_attack_cycle_list)
        adjusted_cs_normal_instruction_list = cls.__adjust_list_sizes(cs_normal_instruction_list)
        adjusted_cs_attack_instruction_list = cls.__adjust_list_sizes(cs_attack_instruction_list)

        (gs_normal_branch_list, gs_attack_branch_list, gs_normal_cycle_list, gs_attack_cycle_list,
         gs_normal_instruction_list, gs_attack_instruction_list) = __get_category_data_list(gs_dict)

        param_stat_dict \
            = {Constant.CS_CYCLE_ATTACK: cs_attack_cycle_list[0], Constant.CS_CYCLE_NORMAL: cs_normal_cycle_list[0],
               Constant.GS_CYCLE_ATTACK: gs_attack_cycle_list, Constant.GS_CYCLE_NORMAL: gs_normal_cycle_list}

        return (adjusted_cs_normal_branch_list, adjusted_cs_attack_branch_list, adjusted_cs_normal_cycle_list,
                adjusted_cs_attack_cycle_list, adjusted_cs_normal_instruction_list, adjusted_cs_attack_instruction_list,
                gs_normal_branch_list, gs_attack_branch_list, gs_normal_cycle_list, gs_attack_cycle_list,
                gs_normal_instruction_list, gs_attack_instruction_list, param_stat_dict)

    @classmethod
    def __adjust_list_sizes(cls, lists):
        non_zero_lists = [lst for lst in lists if any(x != 0 for x in lst)]
        sizes = [len(lst) for lst in non_zero_lists]
        average_size = int(np.mean(sizes)) if sizes else 0

        adjusted_lists = []

        for lst in lists:
            if not lst or not all(isinstance(x, (int, float)) for x in lst):
                adjusted_list = [0] * average_size
                adjusted_lists.append(adjusted_list)
                continue

            current_size = len(lst)
            if all(x == 0 for x in lst):
                adjusted_list = [0] * average_size
            else:
                if current_size == 1 or len(set(lst)) == 1:
                    adjusted_list = [lst[0]] * average_size
                else:
                    xp = np.arange(current_size)
                    x = np.linspace(0, current_size - 1, average_size)
                    try:
                        adjusted_list = np.interp(x, xp, lst).tolist()
                    except ValueError:
                        adjusted_list = [np.mean(lst)] * average_size

            adjusted_lists.append(adjusted_list)

        return adjusted_lists
