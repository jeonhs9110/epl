import os
import prediction_model as pm
import pandas as pd
import numpy as np
import torch
from scipy.stats import poisson

# Configuration
DEVICE = pm.DEVICE
_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_DIR, 'models')

def calculate_advanced_stats(home_lam, away_lam):
    # Safety Check
    if not isinstance(home_lam, (int, float)) or not isinstance(away_lam, (int, float)) or \
       np.isnan(home_lam) or np.isnan(away_lam):
        return {
            "win": 0.0, "draw": 0.0, "loss": 0.0,
            "over_2_5": 0.0,
            "predicted_score": "0-0"
        }

    max_goals = 10
    prob_matrix = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            prob_matrix[h, a] = poisson.pmf(h, home_lam) * poisson.pmf(a, away_lam)

    prob_home_win = np.sum(np.tril(prob_matrix, -1))
    prob_draw = np.sum(np.diag(prob_matrix))
    prob_away_win = np.sum(np.triu(prob_matrix, 1))

    return {
        "win": prob_home_win * 100,
        "draw": prob_draw * 100,
        "loss": prob_away_win * 100
    }

def find_optimal_thresholds():
    print("Loading Data & Models (Clean Mode)...")
    master_df, le_team, le_league = pm.get_master_data()
    
    num_teams = len(le_team.classes_)
    num_leagues = len(le_league.classes_)
    
    model = pm.LeagueAwareModel(num_teams, num_leagues).to(DEVICE)
    try:
        model.load_state_dict(torch.load(os.path.join(_MODELS_DIR, 'FOBO_LEAGUE_AWARE_current.pth'), map_location=DEVICE))
        model.eval()
        print("Model Loaded.")
    except Exception as e:
        print(f"Model file not found or error: {e}")
        return

    RL_STATE_DIM = (pm.EMBED_DIM * 6) + pm.LEAGUE_EMBED_DIM
    agent = pm.PPOAgent(RL_STATE_DIM).to(DEVICE)
    try:
        agent.load_state_dict(torch.load(os.path.join(_MODELS_DIR, 'ppo_agent.pth'), map_location=DEVICE))
        print("RL Agent Loaded.")
    except:
        print("RL Agent not found! Proceeding with only Model stats if possible.")
        agent = None

    # Calculate ELOs
    print("Calculating ELOs...")
    elo_ratings, h_elos, a_elos = pm.calculate_dynamic_elo(master_df)
    master_df['h_elo'] = h_elos
    master_df['a_elo'] = a_elos
    
    # Normalize
    all_elos = np.concatenate([h_elos, a_elos])
    min_e, max_e = np.min(all_elos), np.max(all_elos)
    if max_e == min_e:
        master_df['h_elo_norm'] = 0.0
        master_df['a_elo_norm'] = 0.0
    else:
        master_df['h_elo_norm'] = (master_df['h_elo'] - min_e) / (max_e - min_e)
        master_df['a_elo_norm'] = (master_df['a_elo'] - min_e) / (max_e - min_e)

    # Use 'Test' set logic: Last 20% of each league
    test_rows = []
    leagues = master_df['league_name'].unique()
    for lg in leagues:
        lg_df = master_df[master_df['league_name'] == lg].sort_values('date_obj')
        if len(lg_df) < 50: continue
        cutoff = int(len(lg_df) * 0.8)
        test_chunk = lg_df.iloc[cutoff:]
        test_rows.append(test_chunk)
    
    if not test_rows:
        print("Not enough data.")
        return
        
    df = pd.concat(test_rows)
    print(f"Analyzing {len(df)} matches...")
    
    model_results = [] # (conf, is_hit)
    rl_results = []    # (conf, is_hit)
    
    with torch.no_grad():
        for i, row in df.iterrows():
            try:
                h_id, a_id, l_id = row['home_id'], row['away_id'], row['league_id']
                date = row['date_obj']
                
                # Features
                h_seq = torch.from_numpy(pm.get_team_history(h_id, date, master_df)).float().unsqueeze(0).to(DEVICE)
                a_seq = torch.from_numpy(pm.get_team_history(a_id, date, master_df)).float().unsqueeze(0).to(DEVICE)
                
                o1 = float(row.get('odds_1', 0.0))
                ox = float(row.get('odds_x', 0.0))
                o2 = float(row.get('odds_2', 0.0))
                odds_t = torch.tensor([o1, ox, o2], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                # ELO Tensors
                he = row['h_elo_norm']
                ae = row['a_elo_norm']
                h_elo_t = torch.tensor([he], dtype=torch.float32).to(DEVICE)
                a_elo_t = torch.tensor([ae], dtype=torch.float32).to(DEVICE)
                
                t_h, t_a, t_l = torch.tensor([h_id], device=DEVICE), torch.tensor([a_id], device=DEVICE), torch.tensor([l_id], device=DEVICE)
                
                # --- MODEL ---
                pred = model(h_seq, a_seq, t_h, t_a, t_l, odds_t, h_elo_t, a_elo_t)
                lam = pred[0]
                h_lam, a_lam = lam[0,0].item(), lam[0,1].item()
                stats = calculate_advanced_stats(h_lam, a_lam)
                
                hg, ag = row['home team total goal'], row['away team total goal']
                res = 'D'
                if hg > ag: res = 'H'
                elif ag > hg: res = 'A'
                
                # Model Logic - Determine Confidence based on Prediction
                if stats['win'] > stats['draw'] and stats['win'] > stats['loss']:
                    model_results.append((stats['win'], 1 if res == 'H' else 0))
                elif stats['loss'] > stats['win'] and stats['loss'] > stats['draw']:
                    model_results.append((stats['loss'], 1 if res == 'A' else 0))
                
                # --- RL ---
                if agent:
                    state = model.extract_features(h_seq, a_seq, t_h, t_a, t_l, h_elo_t, a_elo_t)
                    probs = agent.actor(state)
                    act = torch.argmax(probs).item()
                    conf = probs[0, act].item() * 100
                    
                    is_hit = False
                    if act == 0 and res == 'H': is_hit = True
                    elif act == 1 and res == 'D': is_hit = True
                    elif act == 2 and res == 'A': is_hit = True
                    
                    if act != 3: # Ignore pass
                         rl_results.append((conf, 1 if is_hit else 0))
                    
            except Exception as e: 
                # print(e)
                continue

    # --- RESULTS ---
    print("\n--- OPTIMIZATION RESULTS ---")
    
    def find_best(results, name, start=50):
        best_thresh = start
        best_acc = 0
        
        for t in range(start, 95):
            subset = [x for x in results if x[0] >= t]
            if len(subset) < 20: continue
            hits = sum(x[1] for x in subset)
            acc = (hits / len(subset)) * 100
            print(f"{name} Thresh {t}%: Acc {acc:.1f}% ({len(subset)} bets)")
            
            # Simple heuristic: Maximize Acc, but ensure sample size isn't tiny
            if acc >= best_acc: # Greedy
                best_acc = acc
                best_thresh = t
        return best_thresh, best_acc

    mt, ma = find_best(model_results, "Model", 55)
    rt, ra = find_best(rl_results, "RL Agent", 60)
    
    print(f"\nFINAL OPTIMAL THRESHOLDS:")
    print(f"Model Confidence: {mt}% (Exp Acc: {ma:.1f}%)")
    print(f"RL Confidence: {rt}% (Exp Acc: {ra:.1f}%)")

if __name__ == "__main__":
    find_optimal_thresholds()
