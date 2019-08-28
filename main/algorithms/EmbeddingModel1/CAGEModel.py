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
                 part_encoder_dim, object_encoder_dim, grasp_encoder_dim,
                 part_pooling_method,
                 label_dim):
        super(CAGEModel, self).__init__()

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

        # prediction layers
        self.fc = nn.Linear(object_encoder_dim + grasp_encoder_dim + task_embedding_dim, label_dim)#.cuda()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x: [batch_size, features_dimension]

        tasks = x[:, 0]
        object_classes = x[:, 1]
        states = x[:, 2]
        grasps = x[:, 3:5]
        parts = x[:, 5:]

        object_encodes = self.object_encoder(tasks, object_classes, states, parts)
        grasp_encodes = self.grasp_encoder(grasps)

        final_encodes = torch.cat((object_encodes, grasp_encodes), dim=1)
        preds = self.fc(final_encodes)
        log_probs = self.logsoftmax(preds)
        # log_probs: [batch_size, label_dim]

        return log_probs




