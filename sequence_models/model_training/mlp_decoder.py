import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
import pandas as pd
from sentence_transformers import SentenceTransformer
from sequence_models.model_training.process_gemini_data import  collect_merged_train_test_data_from_embeddings


class Decoder(nn.Module):
    def __init__(self, emb_size, out_size, hidden_size=128):
        super().__init__()
        self.l0 = nn.Linear(emb_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, out_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.act(self.l0(x))
        return self.l1(x)


# # Define loss function
# def loss_fn(model, emb, goal):
#     pred = model(emb)
#     return torch.mean(torch.norm(pred - goal, dim=-1))
bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
def loss_fn(model, emb, goal):
    pred = model(emb)
    return bce_loss(pred, goal)

if __name__ == "__main__":
    
    # Define available LLMs
    llms = {
    #    SentenceTransformer('BAAI/bge-large-en-v1.5'): "BAAI/bge-large-en-v1.5",
    #    SentenceTransformer('hkunlp/instructor-large'): "hkunlp/instructor-large",
        SentenceTransformer('thenlper/gte-large'): "thenlper/gte-large"
    }

    json_data_file = "sequence_models/data/merged.json"
    # Initialize WandB
    wandb.login()
    wandb.init(project='grid_decoder_llm', name='mlp_decoder_grid_scale')
    results = {}
    m = 0
    patience_counter = 0
    for llm, llm_name in llms.items():
        wandb.init(project='grid_decoder_llm', name=llm_name)
        batch_size = 64
        epochs = 1500

        train, test = collect_merged_train_test_data_from_embeddings(json_path=json_data_file,train_ratio=0.8,test_ratio=0.2, device="mps")

        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        model = Decoder(train["task_embedding"].shape[1],train["goal"].shape[1]).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=5e-5)
        
        pbar = tqdm.tqdm(total=epochs * train["task_embedding"].shape[0] // batch_size)
        best_val_loss = float('inf')

        for epoch in range(epochs):
            for i in range(train["task_embedding"].shape[0] // batch_size):
                
                emb = torch.tensor(train["task_embedding"][i * batch_size : (i + 1) * batch_size], dtype=torch.float32).to(device)
                goal = torch.tensor(train["goal"][i * batch_size : (i + 1) * batch_size], dtype=torch.float32).to(device)

                optimizer.zero_grad()
                loss = loss_fn(model, emb, goal)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    v_emb = torch.tensor(test["task_embedding"], dtype=torch.float32).to(device)
                    v_goal = torch.tensor(test["goal"], dtype=torch.float32).to(device)
                    val_loss = loss_fn(model, v_emb, v_goal)
                    best_val_loss = min(val_loss.item(), best_val_loss)

                pbar.update()
                pbar.set_description(f"LLM: {llm_name}, epoch: {epoch}, loss: {loss.item():0.4f}, val_loss: {val_loss.item():0.4f}, best val_loss: {best_val_loss:0.4f}")
                wandb.log({
                    "epoch": epoch,
                    "loss": loss.item(),
                    "eval_loss": val_loss.item(),
                    "best_eval_loss": best_val_loss,
                })
        
        results[llm_name] = best_val_loss
        wandb.finish()
        torch.save(model.state_dict(), f"llm{m}_decoder_model_grid_scale.pth")
        m += 1

    print(results)