import os
import torch
import json
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Import Prediction Model (Shared Logic)
import prediction_model as pm
from prediction_model import LeagueAwareModel, train_one_epoch, get_dataloader, PPOAgent, train_ppo_agent

# Config matches app.py
TRAINING_HISTORY_FILE = 'training_history.json'
CURRENT_MODEL_PATH = 'models/FOBO_LEAGUE_AWARE_current.pth'
PREVIOUS_MODEL_PATH = 'models/FOBO_LEAGUE_AWARE_previous.pth'
FINAL_MODEL_PATH = 'models/FOBO_LEAGUE_AWARE_final.pth'
ACC_MODEL_PATH = 'models/FOBO_LEAGUE_AWARE_best_acc.pth'

DEVICE = pm.DEVICE

def load_history():
    if os.path.exists(TRAINING_HISTORY_FILE):
        try:
             with open(TRAINING_HISTORY_FILE, 'r') as f: return json.load(f)
        except: return []
    return []

def save_history(hist):
    with open(TRAINING_HISTORY_FILE, 'w') as f: json.dump(hist, f)

def train_deep_model():
    print("\n" + "="*60)
    print("INDEPENDENT DEEP LEARNING TRAINING")
    print("="*60)
    print(f"Using Device: {DEVICE}")

    # Check Test Mode
    env_test = os.environ.get('FOBO_TEST_MODE', 'false').lower()
    if env_test == 'true':
        EPOCHS = 1
        print("\n  TEST MODE ACTIVE: Training limited to 1 Epoch.  \n")
    else:
        EPOCHS = 1000
    PATIENCE = 100

    # 1. Load Data
    print("Loading Dataset...")
    master_df, le_team, le_league = pm.get_master_data()
    if master_df is None:
        print("Error: No data found.")
        return False
        
    num_teams = len(le_team.classes_)
    num_leagues = len(le_league.classes_)

    # 2. Instantiate Model
    print("Initializing Model...")
    model = LeagueAwareModel(num_teams, num_leagues).to(DEVICE)
    
    # 3. Prepare Training
    train_loader, _ = get_dataloader(batch_size=pm.BATCH_SIZE)
    if train_loader is None: return False
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Warmup for 10 epochs (linear: 0 → 1.0 of base LR),
    # then cosine anneal from base LR down to 1e-6.
    WARMUP_EPOCHS = 10
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                total_iters=WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(EPOCHS - WARMUP_EPOCHS, 1),
                                         eta_min=1e-6)
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[WARMUP_EPOCHS])

    training_history = load_history()
    best_loss = float('inf')
    best_acc = 0.0
    patience_counter = 0

    print(f"Starting Training for {EPOCHS} Epochs (Warmup: {WARMUP_EPOCHS} epochs)...")
    
    # CHECK FOR SKIP FLAG
    if os.environ.get('FOBO_SKIP_DL_TRAIN', 'false').lower() == 'true':
        print("\n[INFO] Skipping Independent DL Training (Requested by User).")
        if os.path.exists(CURRENT_MODEL_PATH):
             print(f"Loading existing best model from: {CURRENT_MODEL_PATH}")
             try:
                 model.load_state_dict(torch.load(CURRENT_MODEL_PATH, map_location=DEVICE))
             except RuntimeError as e:
                 print(f"[CRITICAL ERROR HANDLED] Shape Mismatch! {e}")
                 print(">> Falling back to untrained model (Cannot skip training on architecture change!).")
        else:
             print("[WARNING] No existing model found to load! Proceeding with untrained model (not recommended).")
    else:
        # 4. Training Loop
        for epoch in range(EPOCHS):
            loss, acc = train_one_epoch(model, train_loader, optimizer, DEVICE)
            scheduler.step()
    
            training_history.append({'epoch': epoch + 1, 'loss': float(loss), 'accuracy': float(acc)})
            save_history(training_history)
    
            # Save Best Loss (Current Model)
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
                torch.save(model.state_dict(), CURRENT_MODEL_PATH)
                print(f"Epoch {epoch + 1} | Loss: {loss:.4f} | Acc: {acc:.1f}% | Saved New Best Loss")
            else:
                patience_counter += 1
                if (epoch+1) % 10 == 0:
                     print(f"Epoch {epoch + 1} | Loss: {loss:.4f} | Acc: {acc:.1f}%")
            
            # Save Best Acc
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), ACC_MODEL_PATH)
                print(f"  -> New Best Acc: {acc:.1f}%")
    
            if patience_counter >= PATIENCE:
                print("Early Stopping.")
                break
                
        # Save Final (Only if we trained)
        torch.save(model.state_dict(), FINAL_MODEL_PATH)
        print("Saved Final Model State.")
    
    # ==========================================
    # 5. REINFORCEMENT LEARNING FINE-TUNING
    # ==========================================
    print("\n" + "="*60)
    print("STARTING REINFORCEMENT LEARNING (PPO) FINE-TUNING")
    print("="*60)
    
    # Initialize PPO Agent
    # State Dim = Output of extract_features (Concat of embeddings + GNN + temporal features)
    # Level 8 Upgrade: 6 * 256 + 32 + (2 * 256 for Cross-Attention) = 2080
    STATE_DIM = 2080 
    ACTION_DIM = 4 # 0:Home, 1:Draw, 2:Away, 3:Pass
    
    ppo_agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, lr=0.0003, entropy_coef=0.15).to(DEVICE)
    
    # Reload best-loss weights before PPO training.
    # During the DL training loop the model variable ends on the final epoch,
    # which may differ from the best-loss checkpoint saved to CURRENT_MODEL_PATH.
    # At inference, model_current always loads CURRENT_MODEL_PATH (best-loss), so the
    # PPO agent must be trained on exactly those weights to keep embedding spaces aligned.
    if not os.environ.get('FOBO_SKIP_DL_TRAIN', 'false').lower() == 'true':
        if os.path.exists(CURRENT_MODEL_PATH):
            model.load_state_dict(torch.load(CURRENT_MODEL_PATH, map_location=DEVICE))
            print("Reloaded best-loss weights for PPO training (aligns with inference model_current).")

    rl_epochs = 20
    if os.environ.get('FOBO_TEST_MODE', 'false').lower() == 'true':
        print("\n  TEST MODE ACTIVE: Reducing RL Epochs to 1.  \n")
        rl_epochs = 1

    trained_agent = train_ppo_agent(model, ppo_agent, epochs=rl_epochs)
    
    # Save Agent
    torch.save(trained_agent.state_dict(), "models/ppo_agent.pth")
    print(">> PPO Agent Optimized and Saved to models/ppo_agent.pth")

    
    return True

if __name__ == "__main__":
    train_deep_model()
