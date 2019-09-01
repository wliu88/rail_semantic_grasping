from main.DataReader import DataReader

from main.algorithms.BaseFeaturesModel import BaseFeaturesModel
from main.algorithms.FrequencyTableModel import FrequencyTableModel
from main.algorithms.RandomModel import RandomModel
from main.algorithms.EmbeddingModel1.CAGEAlgorithm import CAGEAlgorithm

from main.Metrics import score_1, score_2, score_3, score_4, score_embedding_3

if __name__ == "__main__":
    data_reader = DataReader("/home/weiyu/catkin_ws/src/rail_semantic_grasping/data")
    base_features_model = BaseFeaturesModel(classifier="KNN")
    frequence_table_model = FrequencyTableModel(use_affordance=True, use_material=False)
    random_model = RandomModel()
    cage_algorithm = CAGEAlgorithm()

    # experiments = data_reader.prepare_data_1()
    #
    # # results = base_features_model.run_experiments(data_reader.data, experiments)
    #
    # # results = frequence_table_model.run_experiments(data_reader.data, experiments)
    #
    # results = random_model.run_experiments(data_reader.data, experiments)

    experiments = data_reader.prepare_data_3()

    results = cage_algorithm.run_experiments(data_reader.data, experiments)

    score_embedding_3(results)
