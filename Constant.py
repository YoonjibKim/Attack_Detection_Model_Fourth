SCENARIO_1 = 'Correct_ID_Random_CS_Off_Gaussian_Off'
SCENARIO_2 = 'Correct_ID_Random_CS_Off_Gaussian_On'
SCENARIO_3 = 'Correct_ID_Random_CS_On_Gaussian_Off'
SCENARIO_4 = 'Correct_ID_Random_CS_On_Gaussian_On'
SCENARIO_5 = 'Wrong_CS_TS_Random_CS_Off_Gaussian_Off'
SCENARIO_6 = 'Wrong_CS_TS_Random_CS_Off_Gaussian_On'
SCENARIO_7 = 'Wrong_CS_TS_Random_CS_On_Gaussian_Off'
SCENARIO_8 = 'Wrong_CS_TS_Random_CS_On_Gaussian_On'
SCENARIO_9 = 'Wrong_EV_TS_Random_CS_Off_Gaussian_Off'
SCENARIO_10 = 'Wrong_EV_TS_Random_CS_Off_Gaussian_On'
SCENARIO_11 = 'Wrong_EV_TS_Random_CS_On_Gaussian_Off'
SCENARIO_12 = 'Wrong_EV_TS_Random_CS_On_Gaussian_On'
SCENARIO_13 = 'Wrong_ID_Random_CS_Off_Gaussian_Off'
SCENARIO_14 = 'Wrong_ID_Random_CS_Off_Gaussian_On'
SCENARIO_15 = 'Wrong_ID_Random_CS_On_Gaussian_Off'
SCENARIO_16 = 'Wrong_ID_Random_CS_On_Gaussian_On'

FULL_SCENARIO_NAME_LIST = [SCENARIO_1, SCENARIO_2, SCENARIO_3, SCENARIO_4, SCENARIO_5, SCENARIO_6, SCENARIO_7,
                           SCENARIO_8, SCENARIO_9, SCENARIO_10, SCENARIO_11, SCENARIO_12, SCENARIO_13, SCENARIO_14,
                           SCENARIO_15, SCENARIO_16]

STD_CS_DIR_PATH = 'Dataset/Optimal_Hyperparam/Raw/STD/CS'
STD_GS_DIR_PATH = 'Dataset/Optimal_Hyperparam/Raw/STD/GS'
TOP_CS_DIR_PATH = 'Dataset/Optimal_Hyperparam/Raw/TOP/CS'
TOP_GS_DIR_PATH = 'Dataset/Optimal_Hyperparam/Raw/TOP/GS'

TRAINING_FEATURE = 'training_feature'
TRAINING_LABEL = 'training_label'
TESTING_FEATURE = 'testing_feature'
TESTING_LABEL = 'testing_label'

CSR = 'combined_sampling_resolution'

ATTACK = 'attack'
NORMAL = 'normal'

CID_RCOFF_GOFF = 'cid_rcoff_goff'
CID_RCOFF_GON = 'cid_rcoff_gon'
CID_RCON_GOFF = 'cid_rcon_goff'
CID_RCON_GON = 'cid_rcon_gon'
WCT_RCOFF_GOFF = 'wct_rcoff_goff'
WCT_RCOFF_GON = 'wct_rcoff_gon'
WCT_RCON_GOFF = 'wct_rcon_goff'
WCT_RCON_GON = 'wct_rcon_gon'
WET_RCOFF_GOFF = 'wet_rcoff_goff'
WET_RCOFF_GON = 'wet_rcoff_gon'
WET_RCON_GOFF = 'wet_rcon_goff'
WET_RCON_GON = 'wet_rcon_gon'
WID_RCOFF_GOFF = 'wid_rcoff_goff'
WID_RCOFF_GON = 'wid_rcoff_gon'
WID_RCON_GOFF = 'wid_rcon_goff'
WID_RCON_GON = 'wid_rcon_go'

DNN_RAW_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_PATH = 'Dataset/DNN/Raw/Correct_ID/Random_CS_Off/Gaussian_Off'
DNN_RAW_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_ON_PATH = 'Dataset/DNN/Raw/Correct_ID/Random_CS_Off/Gaussian_On'
DNN_RAW_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_OFF_PATH = 'Dataset/DNN/Raw/Correct_ID/Random_CS_On/Gaussian_Off'
DNN_RAW_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_ON_PATH = 'Dataset/DNN/Raw/Correct_ID/Random_CS_On/Gaussian_On'
DNN_RAW_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_OFF_PATH = 'Dataset/DNN/Raw/Wrong_CS_TS/Random_CS_Off/Gaussian_Off'
DNN_RAW_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_ON_PATH = 'Dataset/DNN/Raw/Wrong_CS_TS/Random_CS_Off/Gaussian_On'
DNN_RAW_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_OFF_PATH = 'Dataset/DNN/Raw/Wrong_CS_TS/Random_CS_On/Gaussian_Off'
DNN_RAW_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_ON_PATH = 'Dataset/DNN/Raw/Wrong_CS_TS/Random_CS_On/Gaussian_On'
DNN_RAW_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_OFF_PATH = 'Dataset/DNN/Raw/Wrong_EV_TS/Random_CS_Off/Gaussian_Off'
DNN_RAW_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_ON_PATH = 'Dataset/DNN/Raw/Wrong_EV_TS/Random_CS_Off/Gaussian_On'
DNN_RAW_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_OFF_PATH = 'Dataset/DNN/Raw/Wrong_EV_TS/Random_CS_On/Gaussian_Off'
DNN_RAW_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_ON_PATH = 'Dataset/DNN/Raw/Wrong_EV_TS/Random_CS_On/Gaussian_On'
DNN_RAW_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_PATH = 'Dataset/DNN/Raw/Wrong_ID/Random_CS_Off/Gaussian_Off'
DNN_RAW_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_ON_PATH = 'Dataset/DNN/Raw/Wrong_ID/Random_CS_Off/Gaussian_On'
DNN_RAW_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_OFF_PATH = 'Dataset/DNN/Raw/Wrong_ID/Random_CS_On/Gaussian_Off'
DNN_RAW_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_ON_PATH = 'Dataset/DNN/Raw/Wrong_ID/Random_CS_On/Gaussian_On'

RAW_DATASET_PATH_DICT = {CID_RCOFF_GOFF: DNN_RAW_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_PATH,
                         CID_RCOFF_GON: DNN_RAW_CORRECT_ID_RANDOM_CS_OFF_GAUSSIAN_ON_PATH,
                         CID_RCON_GOFF: DNN_RAW_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_OFF_PATH,
                         CID_RCON_GON: DNN_RAW_CORRECT_ID_RANDOM_CS_ON_GAUSSIAN_ON_PATH,
                         WCT_RCOFF_GOFF: DNN_RAW_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_OFF_PATH,
                         WCT_RCOFF_GON: DNN_RAW_WRONG_CS_TS_RANDOM_CS_OFF_GAUSSIAN_ON_PATH,
                         WCT_RCON_GOFF: DNN_RAW_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_OFF_PATH,
                         WCT_RCON_GON: DNN_RAW_WRONG_CS_TS_RANDOM_CS_ON_GAUSSIAN_ON_PATH,
                         WET_RCOFF_GOFF: DNN_RAW_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_OFF_PATH,
                         WET_RCOFF_GON: DNN_RAW_WRONG_EV_TS_RANDOM_CS_OFF_GAUSSIAN_ON_PATH,
                         WET_RCON_GOFF: DNN_RAW_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_OFF_PATH,
                         WET_RCON_GON: DNN_RAW_WRONG_EV_TS_RANDOM_CS_ON_GAUSSIAN_ON_PATH,
                         WID_RCOFF_GOFF: DNN_RAW_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_OFF_PATH,
                         WID_RCOFF_GON: DNN_RAW_WRONG_ID_RANDOM_CS_OFF_GAUSSIAN_ON_PATH,
                         WID_RCON_GOFF: DNN_RAW_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_OFF_PATH,
                         WID_RCON_GON: DNN_RAW_WRONG_ID_RANDOM_CS_ON_GAUSSIAN_ON_PATH}

STAT = 'STAT'
TOP = 'TOP'
TIME_DELTA = 'time_diff'
CS = 'CS'
GS = 'GS'
SAMPLING_COUNT = 'sampling_count'
DATA_POINT = 'data_point'

CS_1_ID = '2-39-89-25'
CS_2_ID = '2-39-139-28'
CS_3_ID = '2-39-131-30'
GS_ID = 'GS_1'

BRANCH = 'branch'
CYCLES = 'cycles'
INSTRUCTIONS = 'instructions'

ATTACK_LABEL = 1
NORMAL_LABEL = 0


class GridSearch:
    AB = 'ada_boost'
    AC = 'agglomerative_clustering'
    DT = 'decision_tree'
    GNB = 'gaussian_naive_bayes'
    GB = 'gradient_boost'
    KM = 'kmeans'
    RF = 'random_forest'

    top_scn_1_gs_b = 'top_scn_1_gs_b_pgc'
    top_scn_2_gs_b = 'top_scn_2_gs_b_pgc'
    top_scn_3_cs_c = 'top_scn_3_cs_c_me'
    top_scn_3_gs_i = 'top_scn_3_gs_i_pgc'
    top_scn_5_gs_i = 'top_scn_5_gs_i_pgc'
    top_scn_7_cs_i = 'top_scn_7_cs_i_pgc'
    top_scn_7_gs_i = 'top_scn_7_gs_i_pgc'
    top_scn_9_gs_b = 'top_scn_9_gs_b_pgc'
    top_scn_11_cs_i = 'top_scn_11_cs_i_pgc'
    top_scn_11_gs_i = 'top_scn_11_gs_i_pgc'
    top_scn_12_gs_b = 'top_scn_12_gs_b_petsi_pgc'
    top_scn_13_gs_b = 'top_scn_13_gs_b_pgc'
    top_scn_15_cs_i = 'top_scn_15_cs_i_pgc'
    top_scn_15_cs_b = 'top_scn_15_gs_b_pgc'

    symbol_dict = {'pgc': 'psi_group_change', 'me': 'memset_erms', 'petsi': '__perf_event_task_sched_in'}

    instructions = 'instructions'
    branch = 'branch'
    cycles = 'cycles'

    category_dict = {'i': instructions, 'b': branch, 'c': cycles}

    top_target_size_dict = {top_scn_1_gs_b: 217, top_scn_2_gs_b: 120, top_scn_3_cs_c: 78,
                            top_scn_3_gs_i: 118, top_scn_5_gs_i: 255, top_scn_7_cs_i: 80,
                            top_scn_7_gs_i: 112, top_scn_9_gs_b: 199, top_scn_11_cs_i: 83,
                            top_scn_11_gs_i: 104, top_scn_12_gs_b: 8, top_scn_13_gs_b: 214,
                            top_scn_15_cs_i: 96, top_scn_15_cs_b: 122}

    top_target_ml_dict = {top_scn_1_gs_b: DT, top_scn_2_gs_b: AB, top_scn_3_cs_c: DT, top_scn_3_gs_i: DT,
                          top_scn_5_gs_i: AB, top_scn_7_cs_i: AB, top_scn_7_gs_i: GNB, top_scn_9_gs_b: DT,
                          top_scn_11_cs_i: RF, top_scn_11_gs_i: GB, top_scn_12_gs_b: AC, top_scn_13_gs_b: GB,
                          top_scn_15_cs_i: AB, top_scn_15_cs_b: AB}

    top_target_path_dict = {top_scn_1_gs_b: TOP_GS_DIR_PATH + '/' + SCENARIO_1 + '.json',
                            top_scn_2_gs_b: TOP_GS_DIR_PATH + '/' + SCENARIO_2 + '.json',
                            top_scn_3_cs_c: TOP_CS_DIR_PATH + '/' + SCENARIO_3 + '.json',
                            top_scn_3_gs_i: TOP_GS_DIR_PATH + '/' + SCENARIO_3 + '.json',
                            top_scn_5_gs_i: TOP_GS_DIR_PATH + '/' + SCENARIO_5 + '.json',
                            top_scn_7_cs_i: TOP_CS_DIR_PATH + '/' + SCENARIO_7 + '.json',
                            top_scn_7_gs_i: TOP_GS_DIR_PATH + '/' + SCENARIO_7 + '.json',
                            top_scn_9_gs_b: TOP_GS_DIR_PATH + '/' + SCENARIO_9 + '.json',
                            top_scn_11_cs_i: TOP_CS_DIR_PATH + '/' + SCENARIO_11 + '.json',
                            top_scn_11_gs_i: TOP_GS_DIR_PATH + '/' + SCENARIO_11 + '.json',
                            top_scn_12_gs_b: TOP_GS_DIR_PATH + '/' + SCENARIO_12 + '.json',
                            top_scn_13_gs_b: TOP_GS_DIR_PATH + '/' + SCENARIO_13 + '.json',
                            top_scn_15_cs_i: TOP_CS_DIR_PATH + '/' + SCENARIO_15 + '.json',
                            top_scn_15_cs_b: TOP_GS_DIR_PATH + '/' + SCENARIO_15 + '.json'}

    std_scn_2_cs_b_i = 'std_scn_2_cs_b_i'
    std_scn_4_cs_b_c_i = 'std_scn_4_cs_b_c_i'
    std_scn_4_gs_b_c_i = 'std_scn_4_gs_b_c_i'
    std_scn_6_cs_b_c_i = 'std_scn_6_cs_b_c_i'
    std_scn_6_gs_b_c = 'std_scn_6_gs_b_c'
    std_scn_8_cs_b_c_i = 'std_scn_8_cs_b_c_i'
    std_scn_8_gs_b_c = 'std_scn_8_gs_b_c'
    std_scn_10_cs_b_i = 'std_scn_10_cs_b_i'
    std_scn_10_gs_b_c_i = 'std_scn_10_gs_b_c_i'
    std_scn_11_cs_b = 'std_scn_11_cs_b'
    std_scn_11_gs_c_i = 'std_scn_11_gs_c_i'
    std_scn_12_cs_b_c_i = 'std_scn_12_cs_b_c_i'
    std_scn_12_cs_i = 'std_scn_12_cs_i'
    std_scn_14_cs_b_i = 'std_scn_14_cs_b_c_i'
    std_scn_14_gs_b_c_i = 'std_scn_14_gs_b_c_i'
    std_scn_15_cs_b_c = 'std_scn_15_cs_b_c'
    std_scn_16_cs_b_i = 'std_scn_16_cs_b_c_i'
    std_scn_16_gs_b_c_i = 'std_scn_16_gs_b_c_i'

    std_target_ml_dict = {std_scn_2_cs_b_i: AB, std_scn_4_cs_b_c_i: AB, std_scn_4_gs_b_c_i: AB,
                          std_scn_6_cs_b_c_i: AB, std_scn_6_gs_b_c: RF, std_scn_8_cs_b_c_i: AB,
                          std_scn_8_gs_b_c: RF, std_scn_10_cs_b_i: RF, std_scn_10_gs_b_c_i: AB,
                          std_scn_11_cs_b: RF, std_scn_11_gs_c_i: RF, std_scn_12_cs_b_c_i: AB,
                          std_scn_12_cs_i: KM, std_scn_14_cs_b_i: AB, std_scn_14_gs_b_c_i: RF,
                          std_scn_15_cs_b_c: AB, std_scn_16_cs_b_i: AB, std_scn_16_gs_b_c_i: DT}

    std_target_path_dict = {std_scn_2_cs_b_i: STD_CS_DIR_PATH + '/' + SCENARIO_2 + '.json',
                            std_scn_4_cs_b_c_i: STD_CS_DIR_PATH + '/' + SCENARIO_4 + '.json',
                            std_scn_4_gs_b_c_i: STD_GS_DIR_PATH + '/' + SCENARIO_4 + '.json',
                            std_scn_6_cs_b_c_i: STD_CS_DIR_PATH + '/' + SCENARIO_6 + '.json',
                            std_scn_6_gs_b_c: STD_GS_DIR_PATH + '/' + SCENARIO_6 + '.json',
                            std_scn_8_cs_b_c_i: STD_CS_DIR_PATH + '/' + SCENARIO_8 + '.json',
                            std_scn_8_gs_b_c: STD_GS_DIR_PATH + '/' + SCENARIO_8 + '.json',
                            std_scn_10_cs_b_i: STD_CS_DIR_PATH + '/' + SCENARIO_10 + '.json',
                            std_scn_10_gs_b_c_i: STD_GS_DIR_PATH + '/' + SCENARIO_10 + '.json',
                            std_scn_11_cs_b: STD_CS_DIR_PATH + '/' + SCENARIO_11 + '.json',
                            std_scn_11_gs_c_i: STD_GS_DIR_PATH + '/' + SCENARIO_11 + '.json',
                            std_scn_12_cs_b_c_i: STD_CS_DIR_PATH + '/' + SCENARIO_12 + '.json',
                            std_scn_12_cs_i: STD_CS_DIR_PATH + '/' + SCENARIO_12 + '.json',
                            std_scn_14_cs_b_i: STD_CS_DIR_PATH + '/' + SCENARIO_14 + '.json',
                            std_scn_14_gs_b_c_i: STD_GS_DIR_PATH + '/' + SCENARIO_14 + '.json',
                            std_scn_15_cs_b_c: STD_CS_DIR_PATH + '/' + SCENARIO_15 + '.json',
                            std_scn_16_cs_b_i: STD_CS_DIR_PATH + '/' + SCENARIO_16 + '.json',
                            std_scn_16_gs_b_c_i: STD_GS_DIR_PATH + '/' + SCENARIO_16 + '.json'}


class DNN:
    DNN = 'dnn'
    CS_TR_DATA_PATH = 'Dataset/DNN/ML_Ready/cs_tr_data'
    CS_TR_LABEL_PATH = 'Dataset/DNN/ML_Ready/cs_tr_label'
    CS_TE_DATA_PATH = 'Dataset/DNN/ML_Ready/cs_te_data'
    CS_TE_LABEL_PATH = 'Dataset/DNN/ML_Ready/cs_te_label'
    GS_TR_DATA_PATH = 'Dataset/DNN/ML_Ready/gs_tr_data'
    GS_TR_LABEL_PATH = 'Dataset/DNN/ML_Ready/gs_tr_label'
    GS_TE_DATA_PATH = 'Dataset/DNN/ML_Ready/gs_te_data'
    GS_TE_LABEL_PATH = 'Dataset/DNN/ML_Ready/gs_te_label'

    CS_TR_DATA = 'cs_tr_data'
    CS_TR_LABEL = 'cs_tr_label'
    CS_TE_DATA = 'cs_te_data'
    CS_TE_LABEL = 'cs_te_label'
    GS_TR_DATA = 'gs_tr_data'
    GS_TR_LABEL = 'gs_tr_label'
    GS_TE_DATA = 'gs_te_data'
    GS_TE_LABEL = 'gs_te_label'

    CS_FIG_PATH = 'Result/DNN/CS/fig'
    GS_FIG_PATH = 'Result/DNN/GS/fig'

    CS_RESULT_PATH = 'Result/DNN/CS/result.json'
    GS_RESULT_PATH = 'Result/DNN/GS/result.json'


GRID_PARAMETER = 'grid_param'
GRID_SCORE = 'grid_score'
CLASSIFICATION_RESULT = 'classification_result'
GRID_TOP_BAYESIAN_PATH = 'Result/Optimal_Hyperparameter/top_bayesian.json'
GRID_STAT_BAYESIAN_PATH = 'Result/Optimal_Hyperparameter/stat_bayesian.json'
GRID_TOP_ALL_PATH = 'Result/Optimal_Hyperparameter/top_all.json'
GRID_STAT_ALL_PATH = 'Result/Optimal_Hyperparameter/stat_all.json'
