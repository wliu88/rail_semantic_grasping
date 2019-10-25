from main.DataReader import DataReader

from main.algorithms.BaseFeaturesModel import BaseFeaturesModel
from main.algorithms.FrequencyTableModel import FrequencyTableModel
from main.algorithms.RandomModel import RandomModel
from main.algorithms.WideAndDeepModel.CAGEAlgorithm import CAGEAlgorithm

from main.Metrics import score_1, score_2, score_3, score_4


if __name__ == "__main__":
    data_reader = DataReader("/home/weiyu/catkin_ws/src/rail_semantic_grasping/data")
    base_features_model = BaseFeaturesModel(classifier="KNN")
    frequence_table_model = FrequencyTableModel(use_affordance=True, use_material=False, use_context=True)
    random_model = RandomModel()
    cage_algorithm = CAGEAlgorithm()

    # To run a test, decide which algorithm and which experiment you want to use.
    #
    # e.g., if you want to run experiment 4 with CAGE algorithm, just uncomment the following code:
    # experiments = data_reader.prepare_data_4()
    # results = cage_algorithm.run_experiments(data_reader.data, experiments)
    # score_4(results)

    # To save model for robot experiments, just provide the file you want to save the model to.
    #
    # e.g.,
    # experiments = data_reader.prepare_data_4(repeat_num=1, test_percentage=0.1)
    # results = cage_algorithm.run_experiments(data_reader.data, experiments, "/home/weiyu/catkin_ws/src/rail_semantic_grasping/models/robo_exp_wd_large.pkl")
    # score_4(results)
    #
    # e.g.,
    # experiments = data_reader.prepare_data_4(repeat_num=1, test_percentage=0.1)
    # results = base_features_model.run_experiments(data_reader.data, experiments, "/home/weiyu/catkin_ws/src/rail_semantic_grasping/models/robo_exp_vf.pkl")
    # score_4(results)


