import torch
import torch.nn as nn

import main.DataSpecification as DataSpecification

torch.manual_seed(1)


class PartEncoder(nn.Module):
    """
    This class is used to encode information of a part
    """

    def __init__(self, affordance_vocab_size, material_vocab_size, affordance_embedding_dim, material_embedding_dim, part_encoder_dim):
        super(PartEncoder, self).__init__()

        # embeddings
        self.affordance_embeddings = nn.Embedding(affordance_vocab_size, affordance_embedding_dim, padding_idx=0)#.cuda()
        self.material_embeddings = nn.Embedding(material_vocab_size, material_embedding_dim, padding_idx=0)#.cuda()

        # propagation model
        self.part_propagation = nn.Linear(affordance_embedding_dim + material_embedding_dim,
                                          part_encoder_dim)#.cuda()

        # Important: padding of parts works because we use relu. So all values in embeds are larger than 0. The padding
        #            will have all 0s.
        self.relu = nn.ReLU()#.cuda()

    def forward(self, parts):
        # parts: [batch_size, num_parts, 2]
        affordance_embeds = self.affordance_embeddings(parts[:, :, 0])
        material_embeds = self.affordance_embeddings(parts[:, :, 1])

        part_embeds = torch.cat((affordance_embeds, material_embeds), dim=-1)
        part_encodes = self.relu(self.part_propagation(part_embeds))

        return part_encodes


class ObjectEncoder(nn.Module):
    """
    This class is used to encode information of an object
    """

    def __init__(self, part_encoder,
                 task_vocab_size, object_vocab_size, state_vocab_size,
                 task_embedding_dim, object_embedding_dim, state_embedding_dim,
                 part_encoder_dim, object_encoder_dim,
                 pooling_method):

        super(ObjectEncoder, self).__init__()

        # part encoder
        self.part_encoder_dim = part_encoder_dim
        self.part_encoder = part_encoder

        # part pooling
        self.pooling_method = pooling_method

        # embeddings
        self.task_embeddings = nn.Embedding(task_vocab_size, task_embedding_dim)#.cuda()
        self.object_embeddings = nn.Embedding(object_vocab_size, object_embedding_dim)#.cuda()
        self.state_embeddings = nn.Embedding(state_vocab_size, state_embedding_dim)#.cuda()

        # propagation model
        self.object_propagation = nn.Linear(part_encoder_dim + object_embedding_dim + state_embedding_dim,
                                            object_encoder_dim)#.cuda()

        self.relu = nn.ReLU()#.cuda()

    def forward(self, tasks, object_classes, states, parts):
        # tasks, object_classes, states: [batch_size, 1]
        # parts: [batch_size, 2 * num_parts]

        batch_size = parts.shape[0]
        parts = parts.view(batch_size, -1, 2)
        # parts after reshape: [batch_size, num_parts, 2]
        parts_encodes_all = self.part_encoder(parts)
        # parts_encodes_all: [batch_size, num_parts, part_encoder_dim]

        # pooling
        if self.pooling_method == "max":
            parts_encodes, _ = torch.max(parts_encodes_all, dim=1)

        # 2. task, object, state
        task_embeds = self.task_embeddings(tasks)
        object_embeds = self.object_embeddings(object_classes)
        state_embeds = self.state_embeddings(states)

        # 3. create object encodes
        object_embeds = torch.cat((object_embeds, state_embeds, parts_encodes), dim=1)
        object_encodes = self.relu(self.object_propagation(object_embeds))

        # 4. combine task with object encodes
        # Important: We can also use a nn here to transform task embeds
        all_encodes = torch.cat((task_embeds, object_encodes), dim=1)

        return all_encodes


class GraspEncoder(nn.Module):
    """
    This class is used to encode information of a part
    """

    def __init__(self, part_encoder, part_encoder_dim, grasp_encoder_dim):
        super(GraspEncoder, self).__init__()

        # part encoder
        self.part_encoder = part_encoder

        # propagation model
        self.grasp_propagation = nn.Linear(part_encoder_dim, grasp_encoder_dim)#.cuda()

        self.relu = nn.ReLU()#.cuda()

    def forward(self, grasp):
        # grasp is (affordance, material)
        # grasp: [batch_size, 2]
        batch_size = grasp.shape[0]
        part_encodes = self.part_encoder(grasp.view(batch_size, 1, 2)).view(batch_size, -1)
        grasp_encodes = self.relu(self.grasp_propagation(part_encodes))

        return grasp_encodes


class CAGEModel(nn.Module):

    def __init__(self, affordance_vocab_size, material_vocab_size, task_vocab_size, object_vocab_size, state_vocab_size,
                 affordance_embedding_dim, material_embedding_dim, task_embedding_dim, object_embedding_dim, state_embedding_dim,
                 base_features_dim,
                 part_encoder_dim, object_encoder_dim, grasp_encoder_dim,
                 part_pooling_method,
                 label_dim):
        super(CAGEModel, self).__init__()

        # params
        self.affordance_vocab_size = affordance_vocab_size
        self.material_vocab_size = material_vocab_size
        self.state_vocab_size = state_vocab_size
        self.object_vocab_size = object_vocab_size
        self.task_vocab_size = task_vocab_size
        self.base_features_dim = base_features_dim

        # part encoder
        self.part_encoder = PartEncoder(affordance_vocab_size, material_vocab_size,
                                        affordance_embedding_dim, material_embedding_dim,
                                        part_encoder_dim)

        # object encoder
        self.object_encoder = ObjectEncoder(self.part_encoder,
                                            task_vocab_size, object_vocab_size, state_vocab_size,
                                            task_embedding_dim, object_embedding_dim, state_embedding_dim,
                                            part_encoder_dim, object_encoder_dim,
                                            pooling_method=part_pooling_method)

        # grasp encoder
        self.grasp_encoder = GraspEncoder(self.part_encoder, part_encoder_dim, grasp_encoder_dim)

        # wide part
        self.wide_fc = nn.Linear(task_vocab_size + object_vocab_size + state_vocab_size + affordance_vocab_size + material_vocab_size, label_dim)  #.cuda()

        # deep part
        # + object_encoder_dim + task_embedding_dim
        self.fc1 = nn.Linear(self.base_features_dim, 512)#.cuda()
        self.fc2 = nn.Linear(512, 256)#.cuda()
        self.fc3 = nn.Linear(256, label_dim)#.cuda()
        self.relu = nn.ReLU()#.cuda()

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, semantic_features, base_features):
        # x: [batch_size, features_dimension]

        tasks = semantic_features[:, 0]
        object_classes = semantic_features[:, 1]
        states = semantic_features[:, 2]
        grasps = semantic_features[:, 3:5]
        parts = semantic_features[:, 5:]

        # Wide part
        # create one hot encodings
        task_encodes = create_one_hot(tasks, self.task_vocab_size)
        object_encodes = create_one_hot(object_classes, self.object_vocab_size)
        states_encodes = create_one_hot(states, self.state_vocab_size)
        grasp_affordances = grasps[:, 0]
        grasp_affordances_encodes = create_one_hot(grasp_affordances, self.affordance_vocab_size)
        grasp_materials = grasps[:, 1]
        grasp_materials_encodes = create_one_hot(grasp_materials, self.material_vocab_size)

        wide_inputs = torch.cat((task_encodes, object_encodes, states_encodes, grasp_affordances_encodes, grasp_materials_encodes), dim=1)
        wide_preds = self.wide_fc(wide_inputs)

        # Deep part
        object_encodes = self.object_encoder(tasks, object_classes, states, parts)
        # grasp_encodes = self.grasp_encoder(grasps)
        # final_encodes = torch.cat((object_encodes, grasp_encodes), dim=1)
        # preds = self.fc(final_encodes)

        # deep_inputs = torch.cat((object_encodes, base_features), dim=1)
        deep_inputs = base_features

        deep_encodes_1 = self.relu(self.fc1(deep_inputs))
        deep_encodes_2 = self.relu(self.fc2(deep_encodes_1))
        deep_preds = self.relu(self.fc3(deep_encodes_2))

        # predication
        log_probs = self.logsoftmax(wide_preds + deep_preds)
        # log_probs: [batch_size, label_dim]

        return log_probs


def create_one_hot(idxs, one_hot_length):
    """
    Create one hot encoding
    :param idxs: [batch_size]
    :param one_hot_length:
    :return: [batch_size, one_hot_length]
    """
    one_hot_encodes = torch.FloatTensor(idxs.shape[0], one_hot_length).zero_()
    one_hot_encodes.scatter_(1, idxs.view(-1, 1), 1)
    return one_hot_encodes




