import random
import torch
import numpy as np
import json

from sequence_models.model_training.rnn_model import EventRNN

import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

DECODER_OUTPUT_SIZE = 100
MAX_SEQ_LEN = 8

# Tasks
EXPLORE = 0
NAVIGATE = 1
IDLE = 2
DEFEND_WIDE = 3
DEFEND_TIGHT = 4

train_dict = None
total_dict_size = None
data_grid_size = None
decoder_model = None

class Decoder(nn.Module):
    def __init__(self, emb_size, out_size, hidden_size=128):
        super().__init__()
        self.l0 = nn.Linear(emb_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, out_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.act(self.l0(x))
        return torch.sigmoid(self.l1(x))

def load_decoder(model_path, embedding_size, device):
    
    global decoder_model
    decoder_model = Decoder(emb_size= embedding_size, out_size=DECODER_OUTPUT_SIZE+4)
    decoder_model.load_state_dict(torch.load(model_path, map_location=device))
    decoder_model.eval()
    
def load_sequence_model(model_path, embedding_size, event_size, state_size, device):
    
    global sequence_model
    sequence_model = EventRNN(event_dim=event_size, y_dim=embedding_size, latent_dim=embedding_size, input_dim=64, state_dim=state_size, decoder=decoder_model).to(device)
    sequence_model.load_state_dict(torch.load(model_path, map_location=device))
    sequence_model.eval()
    
def load_task_data(
    json_path,
    device='cpu'):
    global train_dict
    global total_dict_size

    # Resolve path to ensure it's absolute and correct regardless of cwd
    project_root = Path(__file__).resolve().parents[3]  # Adjust depending on depth of current file
    full_path = project_root / json_path

    with full_path.open('r') as f:
        data = json.load(f)

    np.random.shuffle(data)

    def process_dataset(dataset):
        output = {}

        if all("states" in entry for entry in dataset):
            states = [entry["states"] for entry in dataset]
            output["states"] = states
        
        if all("y" in entry for entry in dataset):
            task = [entry["y"] for entry in dataset]
            output["task"] = torch.tensor(task, dtype=torch.float32, device=device)
        
        if all("h" in entry for entry in dataset):
            embeddings = [torch.tensor(entry["h"],dtype=torch.float32, device=device) for entry in dataset]
            output["subtasks"] = embeddings
        
        if all("events" in entry for entry in dataset):
            events = []
            for entry in dataset:
                e_all = torch.zeros((MAX_SEQ_LEN, 3), dtype=torch.float32, device=device)
                e = torch.tensor(entry["events"], dtype=torch.float32, device=device)
                e_all[:e.shape[0], :] = e
                events.append(e_all)
            output["event"] = torch.stack(events)
        
        if all("summary" in entry for entry in dataset):
            sentences = [entry["summary"] for entry in dataset]
            output["summary"] = sentences
        
        if all("responses" in entry for entry in dataset):
            responses = [entry["responses"] for entry in dataset]
            output["responses"] = responses

    train_dict = process_dataset(data)
    total_dict_size = len(next(iter(train_dict.values())))
    

class LanguageUnit:

    def __init__(self, batch_size, embedding_size, use_embedding_ratio, device='cpu'):

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.use_embedding_ratio = use_embedding_ratio
        self.device = device

        # Task
        event_dim = 3
        self.embedding_size = embedding_size
        self.task_embeddings = torch.zeros((self.batch_size,embedding_size),device=self.device)
        self.subtask_embeddings = torch.zeros((self.batch_size,embedding_size),device=self.device)
        self.event_sequence = torch.zeros((self.batch_size,MAX_SEQ_LEN,event_dim),device=self.device)
        self.sequence_length = torch.zeros((self.batch_size,), dtype=torch.int, device=self.device)
        self.states = torch.zeros((self.batch_size,), device=self.device)
        self.summary = [ "" for _ in range(self.batch_size)]
        self.response = [ "" for _ in range(self.batch_size)]
        
    def sample_dataset(self, env_index: torch.Tensor, forced_state=None):
        
        packet_size = env_index.shape[0]
        
        # --- pick indices ------------------------------------------------------------     # or any key with same length
        if packet_size <= total_dict_size:
            # Normal case: sample *without* replacement
            sample_indices = torch.randperm(total_dict_size, device=self.device)[:packet_size]
        else:
            # Need repeats → build “base” + “extra” indices
            repeats, remainder = divmod(packet_size, total_dict_size)

            # 1) repeat every index the same number of times
            base = torch.arange(total_dict_size, device=self.device).repeat(repeats)

            # 2) top-up with a random subset for the leftover slots
            extra = torch.randperm(total_dict_size, device=self.device)[:remainder] \
                    if remainder > 0 else torch.empty(0, dtype=torch.long, device=self.device)

            sample_indices = torch.cat([base, extra])
        
        # Sample tensors
        task_dict = {key: value[sample_indices] for key, value in train_dict.items() if key in train_dict and key not in ["states", "subtasks", "responses", "summary"]}
        # Sample sentences
        indices_list = sample_indices.tolist()
        task_dict["summary"] = [train_dict["summary"][i] for i in indices_list]
        task_dict["subtasks"] = [train_dict["subtasks"][i] for i in indices_list]
        task_dict["states"] = [train_dict["states"][i] for i in indices_list]
        task_dict["responses"] = [train_dict["responses"][i] for i in indices_list]
        
        subtask_indices = torch.zeros(packet_size, dtype=torch.int, device=self.device)

        if "task" in task_dict:
            self.task_embeddings[env_index] = task_dict["task"].unsqueeze(1)
        
        if "summary" in task_dict:
            for i , idx in enumerate(env_index):
                self.summary[idx] = task_dict["summary"][i]
        
        if "event" in task_dict:
            event = task_dict["event"]
            self.event_sequence[env_index] = event.unsqueeze(1)
        
        if "states" in task_dict: 
            for i , idx in enumerate(env_index):
                state_found = False
                states = task_dict["states"][i][1:]
                if forced_state is not None:
                    state_found = forced_state in states
                if state_found:
                    matching_indices = [i+1 for i, state in enumerate(states) if state == forced_state]
                    idx = random.choice(matching_indices)
                    subtask_indices[i] = idx
                else:
                    num_subtasks = task_dict["subtasks"][i].shape[0]
                    subtask_idx =  random.randint(0, num_subtasks - 1) if num_subtasks > 0 else 0
                    subtask_indices[i] = subtask_idx
                    
                state = task_dict["states"][i][subtask_indices[i]]
                if state == 'E':
                    self.states[idx] = EXPLORE
                elif state == 'N':
                    self.states[idx] = NAVIGATE
                elif state == 'F':
                    self.states[idx] = IDLE
                elif state == 'P1':
                    self.states[idx] = DEFEND_WIDE
                elif state == 'P2':
                    self.states[idx] = DEFEND_TIGHT
                else:
                        raise ValueError(f"Unknown state {state} in task data")
        
        if "subtasks" in task_dict and "responses" in task_dict:
            for i , idx in enumerate(env_index):
                rnd = random.random()
                if rnd < self.use_embedding_ratio:
                    self.subtask_embeddings[idx] = task_dict["subtasks"][i][subtask_indices[i]]
                self.sequence_length[idx] = max(1 , subtask_indices[i])
                self.response[idx] = task_dict["responses"][i][subtask_indices[i]]

    
    def get_subtask_embedding_from_rnn(self, env_index: torch.Tensor) -> torch.Tensor:
        """ Get the subtask embedding from the RNN model for the given environments. """
        
        # Get the subtask embeddings for the given environments
        return self.subtask_embeddings[env_index].unsqueeze(1)
    
    def observe_event_vector(self):
    
    def observe_task_embeddings(self):

        return self.task_embeddings.flatten(start_dim=1,end_dim=-1)
    
    def observe_subtask_embeddings(self):

        return self.subtask_embeddings.flatten(start_dim=1,end_dim=-1)

    def compute_subtask_embedding_from_rnn(self, env_index: torch.Tensor):
        """ Get the subtask embedding from the RNN model for the given environments. """
        e = self.event_sequence[env_index] # (B, MAX_SEQ_LEN, event_dim)
        y = self.task_embeddings[env_index].unsqueeze(1).expand(-1, MAX_SEQ_LEN, -1)  # (B, MAX_SEQ_LEN, emb_size)
        lengths = self.sequence_length[env_index]  # (B,)
        
        mask = (
            torch.arange(e.size(1), device=lengths.device)
            .unsqueeze(0).expand(lengths.size(0), -1)
            < lengths.unsqueeze(1)
        )
        
        state_one_hot_logits, sequence, _ = sequence_model._rollout(e, y, lengths)
        state_one_hot = F.sigmoid(state_one_hot_logits) * mask.unsqueeze(-1)  # (B, MAX_SEQ_LEN, state_dim + autonmaton_dim)
        sequence = sequence * mask.unsqueeze(-1)  # (B, MAX_SEQ_LEN, emb_size) 
        # Decode the state one_hot into a state index
        # First two values are Automaton index. Next 4 values are state one-hot encoding
        autonmatons = torch.argmax(state_one_hot[:,:,:2],dim=-1)
        autonmaton_index = autonmatons[torch.arange(env_index.size(0)), lengths - 1]
        states = torch.argmax(state_one_hot[:,:,2:],dim=-1)
        
        state_index = states[torch.arange(env_index.size(0)), lengths - 1]
        subtask = sequence[torch.arange(env_index.size(0)), lengths - 1, :]

        # Map rnn state representation to the environment Flags: EXPLORE, NAVIGATE, IDLE, DEFEND_WIDE, DEFEND_TIGHT
        state_index = state_index + (state_index != EXPLORE).float() * autonmaton_index * 2

        self.states[env_index] = state_index
        self.subtask_embeddings[env_index] = subtask
        
    def reset_all(self):
        
        self.task_embeddings.zero_()
        self.subtask_embeddings.zero_()
        self.states.zero_()
        self.event_sequence.zero_()
        self.sequence_length.zero_()
        self.summary = [ ""  for _ in range(self.batch_size)]
        self.response = [ ""  for _ in range(self.batch_size)]
    
    def reset_env(self, env_index):

        self.task_embeddings[env_index].zero_()
        self.subtask_embeddings[env_index].zero_()
        self.states[env_index].zero_()
        self.event_sequence[env_index].zero_()
        self.sequence_length[env_index] = 0
        self.summary[env_index] = ""
        self.response[env_index] = ""
        
            
