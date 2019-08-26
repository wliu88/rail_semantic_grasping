import torch
import torch.nn as nn

import main.DataSpecification as DataSpecification

torch.manual_seed(1)


class PartEncoder(nn.Module):
    """
    This class is used to encode information of a part
    """

    def __init__(self, affordance_embedding_dim, material_embedding_dim, part_encoder_dim):
        super(PartEncoder, self).__init__()

        # vocabs
        self.affordance_to_idx = {}
        self.material_to_idx = {}
        self.build_vocabs()

        # embeddings
        self.affordance_embeddings = nn.Embedding(len(self.affordance_to_idx), affordance_embedding_dim)#.cuda()
        self.material_embeddings = nn.Embedding(len(self.material_to_idx), material_embedding_dim)#.cuda()

        # propagation model
        self.part_propagation = nn.Linear(affordance_embedding_dim + material_embedding_dim,
                                          part_encoder_dim)#.cuda()

        self.relu = nn.ReLU()#.cuda()

    def build_vocabs(self):
        for aff in DataSpecification.AFFORDANCES:
            self.affordance_to_idx[aff] = len(self.affordance_to_idx)
        for mat in DataSpecification.MATERIALS:
            self.material_to_idx[mat] = len(self.material_to_idx)
        # misspell in data
        self.material_to_idx["platic"] = self.material_to_idx["plastic"]
        print(self.affordance_to_idx)
        print(self.material_to_idx)

    def forward(self, part):

        # part is (affordance, material)

        aff = torch.tensor(self.affordance_to_idx[part[0]], dtype=torch.long)#.cuda()
        mat = self.material_to_idx[part[1]]
        mat = torch.tensor(mat, dtype=torch.long)#.cuda()
        aff_embeds = self.affordance_embeddings(aff)
        mat_embeds = self.material_embeddings(mat)
        part_embeds = torch.cat((aff_embeds, mat_embeds))
        part_encodes = self.relu(self.part_propagation(part_embeds))

        return part_encodes


class ObjectEncoder(nn.Module):
    """
    This class is used to encode information of an object
    """

    def __init__(self, part_encoder, task_embedding_dim, object_embedding_dim, state_embedding_dim, part_encoder_dim,
                 object_encoder_dim, pooling_method):

        super(ObjectEncoder, self).__init__()

        # vocabs
        self.task_to_idx = {}
        self.object_to_idx = {}
        self.state_to_idx = {}
        self.build_vocabs()

        # part encoder
        self.part_encoder_dim = part_encoder_dim
        self.part_encoder = part_encoder

        # part pooling
        self.pooling_method = pooling_method

        # embeddings
        self.task_embeddings = nn.Embedding(len(self.task_to_idx), task_embedding_dim)#.cuda()
        self.object_embeddings = nn.Embedding(len(self.object_to_idx), object_embedding_dim)#.cuda()
        self.state_embeddings = nn.Embedding(len(self.state_to_idx), state_embedding_dim)#.cuda()

        # propagation model
        self.object_propagation = nn.Linear(part_encoder_dim + task_embedding_dim + object_embedding_dim + state_embedding_dim,
                                            object_encoder_dim)#.cuda()

        self.relu = nn.ReLU()#.cuda()

    def build_vocabs(self):
        for task in DataSpecification.TASKS:
            self.task_to_idx[task] = len(self.task_to_idx)
        for obj in DataSpecification.OBJECTS:
            self.object_to_idx[obj] = len(self.object_to_idx)

        states = set()
        for obj in DataSpecification.STATES:
            states.update(DataSpecification.STATES[obj])
        for state in states:
            self.state_to_idx[state] = len(self.state_to_idx)

    def forward(self, task, object_class, state, parts):
        # 1. parts
        # parts is a list of part
        parts_encodes_all = torch.FloatTensor(len(parts), self.part_encoder_dim)#.cuda()
        for i, part in enumerate(parts):
            # encode each part
            parts_encodes_all[i, :] = self.part_encoder(part)
        # pooling
        if self.pooling_method == "max":
            parts_encodes, _ = torch.max(parts_encodes_all, dim=0)

        # 2. task, object, state
        tsk = torch.tensor(self.task_to_idx[task], dtype=torch.long)#.cuda()
        obj = torch.tensor(self.object_to_idx[object_class], dtype=torch.long)#.cuda()
        st = torch.tensor(self.state_to_idx[state], dtype=torch.long)#.cuda()
        task_embeds = self.task_embeddings(tsk)
        object_embeds = self.object_embeddings(obj)
        state_embeds = self.state_embeddings(st)

        # 3. combine
        object_embeds = torch.cat((task_embeds, object_embeds, state_embeds, parts_encodes))

        object_encodes = self.relu(self.object_propagation(object_embeds))

        return object_encodes


# Important: part encoder in grasp encoder and object encoder should be shared. So part encoder should not be initialized
#            in object encoder

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
        part_encodes = self.part_encoder(grasp)
        grasp_encodes = self.relu(self.grasp_propagation(part_encodes))

        return grasp_encodes


class CAGEModel(nn.Module):

    def __init__(self, affordance_embedding_dim, material_embedding_dim,
                 task_embedding_dim, object_embedding_dim, state_embedding_dim,
                 part_encoder_dim, object_encoder_dim, grasp_encoder_dim,
                 part_pooling_method,
                 label_dim):
        super(CAGEModel, self).__init__()

        assert grasp_encoder_dim == object_encoder_dim

        # part encoder
        self.part_encoder = PartEncoder(affordance_embedding_dim, material_embedding_dim, part_encoder_dim)

        # object encoder
        self.object_encoder = ObjectEncoder(self.part_encoder, task_embedding_dim, object_embedding_dim,
                                            state_embedding_dim, part_encoder_dim, object_encoder_dim, part_pooling_method)

        # grasp encoder
        self.grasp_encoder = GraspEncoder(self.part_encoder, part_encoder_dim, grasp_encoder_dim)

        # prediction layers
        self.fc = nn.Linear(object_encoder_dim + grasp_encoder_dim, label_dim)#.cuda()
        self.logsoftmax = nn.LogSoftmax(dim=0)

    def forward(self, task, object_class, state, parts, grasp):
        object_encodes = self.object_encoder(task, object_class, state, parts)
        grasp_encodes = self.grasp_encoder(grasp)

        final_encodes = torch.cat((object_encodes, grasp_encodes))
        preds = self.fc(final_encodes)
        log_probs = self.logsoftmax(preds)

        return log_probs




