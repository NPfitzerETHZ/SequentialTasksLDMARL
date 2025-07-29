import torch
import torch.nn as nn
import tkinter as tk
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from sequence_models.model_training.mlp_decoder import Decoder

# Set device
device = "mps" if torch.backends.mps.is_available() else "cpu"
output_grid_dim = 10

# Load sentence transformer
llm = SentenceTransformer('thenlper/gte-large', device=device)

# Load the trained decoder model
model_path = "decoders/llm0_decoder_model_grid_scale.pth"  # Update this path if needed
embedding_size = llm.encode(["dummy"], device=device).shape[1]
model = Decoder(embedding_size, output_grid_dim*output_grid_dim + 4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Build the UI
root = tk.Tk()
root.title("Decoder UI")

# Widgets
tk.Label(root, text="Enter a sentence:").pack(pady=5)

# Larger multiline text input
entry = tk.Text(root, width=60, height=6, wrap="word")
entry.pack(pady=5)

# Create a placeholder for the matplotlib plot
fig, ax = plt.subplots(figsize=(4, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=10)

result_var = tk.StringVar()
tk.Label(root, textvariable=result_var, wraplength=500, justify="left").pack(pady=10)

colorbar = None

def predict():
    global colorbar
    sentence = entry.get("1.0", tk.END).strip()
    if not sentence:
        result_var.set("Please enter a sentence.")
        return

    with torch.no_grad():
        embedding = torch.tensor(llm.encode([sentence]), device=device).squeeze(0)
        prediction = torch.sigmoid(model(embedding)).cpu()  # shape: (105,)
        index = prediction[-4:].argmax().item()  # Get the index of the maximum value in the last 4 elements
        prediction = prediction[:-4]

        # Step 1: Min-max normalize to [0, 1]
        min_val = prediction.min()
        max_val = prediction.max()

        # Step 2: Apply fixed threshold (e.g., 0.8)
        threshold = 0.8 * (max_val - min_val) + min_val
        above_thresh = prediction >= threshold

        # Step 3: Subtract threshold *only* from values above it
        rescaled = torch.zeros_like(prediction)
        rescaled[above_thresh] = prediction[above_thresh]
        print(index)

        grid = rescaled.reshape(output_grid_dim, output_grid_dim).numpy() + index

    # Clear entire figure
    fig.clf()

    # Create fresh axes
    ax = fig.add_subplot(111)
    im = ax.imshow(grid, cmap='viridis')
    ax.set_title(f"Decoder Output ({output_grid_dim}x{output_grid_dim} Grid)")

    # Create a fresh colorbar
    colorbar = fig.colorbar(im, ax=ax)

    canvas.draw()
    result_var.set("Prediction completed and visualized.")


    # Clear entire figure
    fig.clf()

    # Create fresh axes
    ax = fig.add_subplot(111)
    im = ax.imshow(grid, cmap='viridis')
    ax.set_title(f"Decoder Output ({output_grid_dim}x{output_grid_dim} Grid)")

    # Create a fresh colorbar
    colorbar = fig.colorbar(im, ax=ax)

    canvas.draw()
    result_var.set("Prediction completed and visualized.")


tk.Button(root, text="Get Decoder Output", command=predict).pack(pady=10)

# Start the UI
root.mainloop()
