import torch
import numpy as np
import glob
from src.core.train_ultra_action_model import ActionModel

model = ActionModel()
model.load_state_dict(torch.load("action_model_movenet_ultra.pt", map_location="cpu"))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

samples = glob.glob("autotrain_buffer/*.npy")
print(f"[AUTOLEARN] Found {len(samples)} new samples")

if len(samples) == 0:
    exit(0)

for path in samples:
    seq = np.load(path)  # shape (16,17,3)
    seq = seq.reshape(16, 51)

    # PLACEHOLDER: label must be assigned manually
    # You can set "Unknown" or predicted label
    target = torch.tensor([5])  # 5 = 'Other'

    x = torch.from_numpy(seq).unsqueeze(0).float()
    out = model(x)
    loss = loss_fn(out, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "action_model_movenet_ultra.pt")

print("[AUTOLEARN] Model updated successfully!")

