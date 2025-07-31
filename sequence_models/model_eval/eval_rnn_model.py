import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sentence_transformers import SentenceTransformer
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sequence_models.model_training.rnn_model import EventRNN

class Decoder(nn.Module):
    def __init__(self, emb_size, out_size, hidden_size=128):
        super().__init__()
        self.l0 = nn.Linear(emb_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, out_size)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.l0(x))
        return self.l1(x)
    
task_map = {
    0: "Capture the flag and return to base",
    1: "Capture the flag and defend it"
}

subtask_map = {
    0: "Explore the environment",
    1: "Navigate back to base",
    2: "Defending the flag",
    3: "Staying idle at base",
}

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model from .pth checkpoint
def load_event_rnn_model(decoder: Decoder, checkpoint_path: str = "event_rnn_best.pth") -> EventRNN:
    model = EventRNN(
        event_dim=3, y_dim=1024, latent_dim=1024, state_dim=6, input_dim=64, num_layers=1, decoder=decoder
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def load_decoder_model(emb_size: int = 1024, out_size: int = 100) -> Decoder:
    model = Decoder(emb_size, out_size)
    model.load_state_dict(torch.load("decoders/llm0_decoder_model_grid_scale.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Predict full sequence of state labels
def predict_state_rollout(task: str, subtask: str, events: list[list[float]], model: EventRNN, decoder_model: Decoder) -> dict:
    # Embed sentence
    embedder = SentenceTransformer("thenlper/gte-large")
    y = embedder.encode(task, convert_to_tensor=True).unsqueeze(0)  # (1, y_dim)
    if subtask:
        h = embedder.encode(subtask, convert_to_tensor=True).unsqueeze(0)  # (1, y_dim)
    else:
        h = torch.zeros_like(y)

    T = len(events)
    e = torch.tensor(events, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, T, e_dim)
    y = y.unsqueeze(1).expand(-1, T, -1).to(DEVICE)                         # (1, T, y_dim)
    grids = []
    tasks = []
    subtasks = []

    with torch.no_grad():
        state_logits, hiddens, _ = model._initalized_rollout(e, y, h)  # (1, T, state_dim)
        probs = torch.sigmoid(state_logits).squeeze(0)   # (T, state_dim)
        preds = (probs > 0.5).int().tolist()             # list of length T, each is list[state_dim]
        
        # Task
        task_index = probs[:,:2].argmax(dim=-1)  # (T,)
        tasks = [task_map[i.item()] for i in task_index]

        # Subtask
        decoded_output = torch.sigmoid(decoder_model(hiddens)).squeeze(0)  # (out_size)
        subtask_indices = decoded_output[:,-4:].argmax(dim=-1)  # (T,)
        decoded_output = decoded_output[:,:-4] + subtask_indices.unsqueeze(-1)  # here problem
        grids = decoded_output.reshape(-1,10,10).detach().cpu().numpy()
        subtasks = [subtask_map[i.item()] for i in subtask_indices]

    return {
        "pred_probs": probs.cpu().tolist(),
        "pred_labels": preds,
        "decoded_grid": grids,
        "tasks": tasks,
        "subtasks": subtasks
    }

# Run
decoder_model = load_decoder_model(out_size=10*10 + 4)  # 10x10 grid + 4 additional classes
model = load_event_rnn_model(decoder_model, "sequence_models/event_rnn_best_gru-in64-bs128.pth")

# events = [
#     [1., 1., 1.],
#     [0., 1., 0.],
#     [0., 0., 1.],
#     [1., 1., 0.],
#     [0., 0., 1.],
#     [0., 0., 1.],
#     [0., 0., 0.],
#     [0., 0., 0.]
# ]
events = [
    [0., 0., 0.],
]

task = "Ok agents, find the flag in the north-east corner and defend it as best you can.\n"
subtask = "Look for the target" # Initialization subtask - We don't need to start at the beginning of the automaton.

result = predict_state_rollout(
    task=task,
    subtask=subtask,
    events=events,
    model=model,
    decoder_model=decoder_model
)

def pretty_print(task, subtask, events, result):
    # ----- Inputs -----
    print("---Input---")
    print(f"  Task    : {task.strip()}")
    if subtask:
        print(f"  Subtask : {subtask.strip()}")
    else:
        print("  Subtask : None")
    print("  Events  (Found Flag, Spotted Enemy, Reached Base):")
    for idx, event in enumerate(events, 1):
        print(f"    {idx:>2}. {event}")

    # ----- Outputs -----
    print("\n---Outputs---")
    for t in range(len(result["pred_labels"])):
        label = result["pred_labels"][t]
        prob = result["pred_probs"][t]
        tk = result["tasks"][t]
        st = result["subtasks"][t]
        rounded_prob = np.round(prob, 3)
        print(f"  {t:>2}. Prob    : {rounded_prob}")
        print(f"      Label   : {label}")
        print(f"      Task    : {tk}")
        print(f"      Subtask : {st}")

pretty_print(task, subtask, events, result)


grids = result["decoded_grid"]
last_grid = grids[-1]
# Build the UI
root = tk.Tk()
root.title("Decoder UI")

# Create a placeholder for the matplotlib plot
fig, ax = plt.subplots(figsize=(4, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=10)

# Clear entire figure
fig.clf()

# Create fresh axes
ax = fig.add_subplot(111)
im = ax.imshow(last_grid, cmap='viridis')
ax.set_title(f"Decoder Output ({10}x{10} Grid)")

# Create a fresh colorbar
colorbar = fig.colorbar(im, ax=ax)
canvas.draw()
root.mainloop()

