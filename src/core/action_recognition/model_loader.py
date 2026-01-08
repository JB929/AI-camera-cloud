import torch
import torch.nn as nn
import numpy as np

class SimpleActionModel(nn.Module):
    def __init__(self, num_joints=18, num_classes=6):
        super().__init__()
        self.fc1 = nn.Linear(num_joints * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch, num_joints * 2)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# Class labels (for reference)
ACTION_LABELS = ["Standing", "Sitting", "Running", "Falling", "Waving", "Other"]

def load_action_model(model_path=None):
    model = SimpleActionModel()
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print("✅ Loaded pretrained action recognition model.")
    else:
        print("⚙️ Using randomly initialized action model (training optional).")
    model.eval()
    return model

