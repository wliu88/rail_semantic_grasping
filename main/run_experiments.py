from main.DataReader import DataReader

from main.algorithms.BaseFeaturesModel import BaseFeaturesModel
from main.algorithms.FrequencyTableModel import FrequencyTableModel
from main.algorithms.RandomModel import RandomModel
from main.algorithms.WideAndDeepModel.CAGEAlgorithm import CAGEAlgorithm

from main.Metrics import score_1, score_2, score_3, score_4


def run():
    for test_percentage in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print("\nTest percentage:", test_percentage)
        print("KNN")
        experiments = data_reader.prepare_data_4(test_percentage=test_percentage)
        results = base_features_model.run_experiments(data_reader.data, experiments)
        score_4(results)
        print("CAGE")
        experiments = data_reader.prepare_data_4(test_percentage=test_percentage)
        results = base_features_model.run_experiments(data_reader.data, experiments)
        score_4(results)


if __name__ == "__main__":
    data_reader = DataReader("/home/weiyu/catkin_ws/src/rail_semantic_grasping/data")
    base_features_model = BaseFeaturesModel(classifier="KNN")
    frequence_table_model = FrequencyTableModel(use_affordance=True, use_material=False, use_context=True)
    random_model = RandomModel()
    cage_algorithm = CAGEAlgorithm()

    experiments = data_reader.prepare_data_2()
    #
    # results = base_features_model.run_experiments(data_reader.data, experiments)
    #
    results = frequence_table_model.run_experiments(data_reader.data, experiments)
    #
    # results = random_model.run_experiments(data_reader.data, experiments)

    # results = cage_algorithm.run_experiments(data_reader.data, experiments)

    score_2(results)


