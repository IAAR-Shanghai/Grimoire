import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, context_size, grimoire_size, output_size) -> None:
        super().__init__()
        self.context_layer1 = nn.Linear(context_size, 1024)
        self.grimoire_layer1 = nn.Linear(grimoire_size, 1024)
        self.context_layer2 = nn.Linear(1024, 1024)
        self.grimoire_layer2 = nn.Linear(1024, 1024)
        self.joint_layer1 = nn.Linear(4096, 2048)
        self.joint_layer2 = nn.Linear(2048, 1024)
        self.joint_layer3 = nn.Linear(1024, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, context, grimoire):
        context1 = self.relu(self.context_layer1(context))
        grimoire1 = self.relu(self.grimoire_layer1(grimoire))
        context2 = self.relu(self.context_layer2(context1)) + context1
        grimoire2 = self.relu(self.grimoire_layer2(grimoire1)) + grimoire1
        joint_row = torch.stack([context2, grimoire2], dim=1)

        # self-attention
        weight = torch.matmul(joint_row, joint_row.transpose(1, 2))
        weight = nn.functional.softmax(weight, dim=1)
        joint = torch.matmul(weight, joint_row).flatten(start_dim=1)
        joint = torch.cat([joint, context2, grimoire2], dim=1)

        joint = self.relu(self.joint_layer1(joint))
        joint = self.dropout(joint)
        joint = self.relu(self.joint_layer2(joint))
        joint = self.dropout(joint)
        joint = self.relu(self.joint_layer3(joint))
        joint = self.norm(joint)
        joint = self.head(joint)
        joint = self.sigmoid(joint)
        return joint
