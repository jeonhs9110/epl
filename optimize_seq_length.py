import pandas as pd
import numpy as np
from collections import defaultdict
import contextlib
import torch
import torch.nn as nn
from prediction_model import get_master_data, SoccerDataset
import os
import copy
import prediction_model
import json

@contextlib.contextmanager
def override_seq_length(new_len):
    """Temporarily override the global SEQ_LENGTH constant during search."""
    original = prediction_model.SEQ_LENGTH
    prediction_model.SEQ_LENGTH = new_len
    try:
        yield
    finally:
        prediction_model.SEQ_LENGTH = original

print("[1] Loading master data...")
master_df, le_team, le_league = get_master_data()
if master_df is None:
    print("No data found!")
    exit(1)

# We will analyze all leagues present in the master dataset
league_names = master_df['league_name'].dropna().unique().tolist()
league_ids = {}

for l_name in league_names:
    if l_name in le_league.classes_:
        league_ids[l_name] = le_league.transform([l_name])[0]


seq_lengths_to_test = [3, 5, 7, 10]
best_lengths = {}

print(f"[2] Testing leagues: {list(league_ids.keys())}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleNet(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=20, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(128, 2)
        
    def forward(self, h_seq, a_seq):
        _, (h_hn, _) = self.lstm(h_seq)
        _, (a_hn, _) = self.lstm(a_seq)
        combined = torch.cat([h_hn[-1], a_hn[-1]], dim=-1)
        out = self.fc(combined)
        return torch.exp(out) # Predict lambda_h, lambda_a

for l_name, l_id in league_ids.items():
    print(f"\n======================================")
    print(f"Optimizing Sequence Length for: {l_name}")
    print(f"======================================")
    
    league_df = master_df[master_df['league_id'] == l_id].copy()
    if len(league_df) < 100:
        print("Not enough data. Skipping.")
        best_lengths[l_name] = 5
        continue
        
    league_df = league_df.sort_values('date_obj')
    split_idx = int(len(league_df) * 0.8)
    train_df = league_df.iloc[:split_idx]
    val_df = league_df.iloc[split_idx:]
    
    league_results = {}
    
    for seq_len in seq_lengths_to_test:
        print(f"  Testing SEQ_LENGTH = {seq_len}...")
        
        # Use context manager so the global is restored after each test
        with override_seq_length(seq_len):
            train_dataset = SoccerDataset(train_df)
            val_dataset = SoccerDataset(val_df)
            
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
            
            model = SimpleNet(seq_len).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.PoissonNLLLoss(log_input=False)
            
            for epoch in range(3):
                model.train()
                for batch in train_loader:
                    opt = optimizer
                    opt.zero_grad()
                    
                    h_seq = batch['h_seq'].to(DEVICE)
                    a_seq = batch['a_seq'].to(DEVICE)
                    hg = batch['hg'].to(DEVICE)
                    ag = batch['ag'].to(DEVICE)
                    
                    pred = model(h_seq, a_seq)
                    loss = criterion(pred[:, 0], hg) + criterion(pred[:, 1], ag)
                    loss.backward()
                    opt.step()
                    
            model.eval()
            total_loss = 0
            batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    h_seq = batch['h_seq'].to(DEVICE)
                    a_seq = batch['a_seq'].to(DEVICE)
                    hg = batch['hg'].to(DEVICE)
                    ag = batch['ag'].to(DEVICE)
                    
                    pred = model(h_seq, a_seq)
                    loss = criterion(pred[:, 0], hg) + criterion(pred[:, 1], ag)
                    total_loss += loss.item()
                    batches += 1
                    
            avg_val_loss = total_loss / max(1, batches)
            league_results[seq_len] = avg_val_loss
            print(f"    Val Loss: {avg_val_loss:.4f}")
        
    best_seq = min(league_results, key=league_results.get)
    print(f"--> Best Sequence Length for {l_name}: {best_seq}")
    best_lengths[l_name] = best_seq

print("\n\nFINAL OPTIMAL SEQUENCE LENGTHS DICTIONARY:")
print(json.dumps(best_lengths, indent=4))

output_file = 'optimal_seq_lengths.json'
with open(output_file, 'w') as f:
    json.dump(best_lengths, f, indent=4)
print(f"Saved optimal lengths to {output_file}")

