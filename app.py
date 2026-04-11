import os
import torch
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, jsonify, Response, stream_with_context, request
from scipy.stats import poisson
import matplotlib
import traceback
import shutil
from torch.optim.lr_scheduler import CosineAnnealingLR
import json

_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_HISTORY_FILE = os.path.join(_DIR, 'training_history.json')

matplotlib.use('Agg')  # Required for non-GUI backend
import matplotlib.pyplot as plt
import io
import base64

# IMPORT YOUR MODELS
import prediction_model as pm
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("XGBoost not installed. Hybrid model disabled.")

from prediction_model import LeagueAwareModel, train_one_epoch, get_dataloader

app = Flask(__name__)

# --- CONFIGURATION ---
# --- CONFIGURATION ---
DEVICE = pm.DEVICE # Use the centralized device configuration
print(f"\nUSING DEVICE (Inherited): {DEVICE}\n")

# Calculate RL State Dim dynamically
# Level 8 Upgrade: 6 * EMBED_DIM (base) + LEAGUE_DIM + (2 * EMBED_DIM for cross attention)
RL_STATE_DIM = (pm.EMBED_DIM * 6) + pm.LEAGUE_EMBED_DIM + (pm.EMBED_DIM * 2)
_MODELS_DIR = os.path.join(_DIR, 'models')
os.makedirs(_MODELS_DIR, exist_ok=True)
CURRENT_MODEL_PATH = os.path.join(_MODELS_DIR, 'FOBO_LEAGUE_AWARE_current.pth') # Best Loss
PREVIOUS_MODEL_PATH = os.path.join(_MODELS_DIR, 'FOBO_LEAGUE_AWARE_previous.pth')
FINAL_MODEL_PATH = os.path.join(_MODELS_DIR, 'FOBO_LEAGUE_AWARE_final.pth')
ACC_MODEL_PATH = os.path.join(_MODELS_DIR, 'FOBO_LEAGUE_AWARE_best_acc.pth')

# CHECK ENV VAR FOR TRAINING OVERRIDE
env_skip = os.environ.get('FOBO_SKIP_TRAINING', 'false').lower()
SKIP_TRAINING = (env_skip == 'true')

# CHECK TEST MODE
env_test = os.environ.get('FOBO_TEST_MODE', 'false').lower()
if env_test == 'true':
    EPOCHS = 1
    print("\n  TEST MODE ACTIVE: Training limited to 1 Epoch.  \n")
else:
    EPOCHS = 1000
PATIENCE = 100

# --- 3. HELPER FUNCTIONS FOR PERSISTENCE ---
def load_history():
    if os.path.exists(TRAINING_HISTORY_FILE):
        try:
             with open(TRAINING_HISTORY_FILE, 'r') as f: return json.load(f)
        except: return []
    return []

def save_history(hist):
    with open(TRAINING_HISTORY_FILE, 'w') as f: json.dump(hist, f)

# --- GLOBAL VARIABLES ---
model_prev = None  # Model 1 (Yesterday/Previous)
model_current = None  # Model 2 (Today/Fresh)
model_final = None  # Final epoch weights
model_acc = None    # Best accuracy weights
policy_agent = None ## RL AGENT
hybrid_model = None ## XGBOOST HYBRID
lgbm_model = None   ## LIGHTGBM HYBRID
calibrator = None   ## PROBABILITY CALIBRATOR (PLATT SCALING)
le_team = None
training_history = load_history()
le_league = None
master_df = None
elo_ratings = {}
min_elo = 1500
max_elo = 1500

# --- GLOBAL OPTIMAL THRESHOLDS (Cache) ---
optimal_pred_threshold = 75
optimal_rl_threshold = 85

# --- LEAGUE MAPPING (Updated for Robustness) ---
LEAGUE_MAPPING = {
    'Champions': 'Champions League',
    'Premier': 'Premier League',
    'Ligue 1': 'Ligue 1',
    'Ligue 2': 'Ligue 2',
    'Bundesliga': 'Bundesliga',
    '2. Bundesliga': '2 Bundesliga',
    'Eredivisie': 'Eredivisie',
    'Serie A': 'Serie A',
    'Serie B': 'Serie B',
    'La Liga': 'La Liga',
    'La Liga 2': 'Laliga2',
    'Championship': 'Championship',
    'Europa': 'Europa League'
}


# --- HELPER FUNCTIONS ---
def get_model_timestamp(filepath):
    """Returns the formatted modification time of a file."""
    if os.path.exists(filepath):
        ts = os.path.getmtime(filepath)
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    return "N/A"


# Helper to calculate Implied Odds from ELO
def get_implied_odds(h_elo, a_elo):
    """
    Generates synthetic 'implied odds' when real odds are missing (0.0).
    Using standard Elo win expectancy formula and an assumed 25% draw baseline.
    """
    if h_elo <= 0 or a_elo <= 0: return 2.5, 3.2, 2.7
    
    # ELO win expectancy
    e_home = 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo) / 400.0))
    e_away = 1.0 - e_home
    
    # Assume 25% draw probability
    draw_prob = 0.25
    home_prob = e_home * (1.0 - draw_prob)
    away_prob = e_away * (1.0 - draw_prob)
    
    # Convert probability to Decimal Odds (1 / prob), applying a small 5% bookmaker margin
    margin = 0.95
    o1 = 1.0 / max(home_prob * margin, 0.01)
    ox = 1.0 / max(draw_prob * margin, 0.01)
    o2 = 1.0 / max(away_prob * margin, 0.01)
    
    return round(o1, 2), round(ox, 2), round(o2, 2)

# Helper to calculate Power Rankings for internal use
def calculate_power_rankings(target_league):
    # Filter master_df for this league
    league_df = master_df[master_df['league_name'] == target_league].copy()
    
    if league_df.empty:
        return {}, []
        
    teams = set(league_df['home team'].unique()) | set(league_df['away team'].unique())
    ranking_data = []
    
    for team in teams:
        # Get all matches for this team
        team_matches = league_df[(league_df['home team'] == team) | (league_df['away team'] == team)].sort_values('date_obj')
        
        played = len(team_matches)
        pts = 0
        gf = 0
        ga = 0
        form_seq = []
        
        for _, m in team_matches.iterrows():
            is_home = (m['home team'] == team)
            h_score = m['home team total goal']
            a_score = m['away team total goal']
            
            my_score = h_score if is_home else a_score
            opp_score = a_score if is_home else h_score
            
            gf += my_score
            ga += opp_score
            
            res_char = 'D'
            if my_score > opp_score:
                pts += 3
                res_char = 'W'
            elif my_score < opp_score:
                res_char = 'L'
            else:
                pts += 1
                
            form_seq.append(res_char)
            
        gd = gf - ga
        
        # Calculate Form Score (Last 9 - Optimal)
        # W=3, D=1, L=0
        recent_form = form_seq[-9:]
        form_pts = 0
        for r in recent_form:
            if r == 'W': form_pts += 3
            elif r == 'D': form_pts += 1
        
        # POWER INDEX FORMULA
        power_index = (pts * 1.0) + (form_pts * 1.4) + (gd * 0.25)
        
        ranking_data.append({
            'team': team,
            'played': played,
            'pts': pts,
            'gd': int(gd),
            'form': recent_form,
            'power_index': round(power_index, 1)
        })
        
    # Sort by Power Index Descending
    ranking_data.sort(key=lambda x: x['power_index'], reverse=True)
    
    # Create Dict mapping Team -> Rank
    rank_map = {}
    for i, row in enumerate(ranking_data):
        row['rank'] = i + 1
        rank_map[row['team']] = i + 1
        
    return rank_map, ranking_data


def get_recent_matches_info(team_id, current_date, df, team_encoder, n=5):
    """Fetches the n most recent matches for a specific team."""
    mask = ((df['home_id'] == team_id) | (df['away_id'] == team_id)) & (df['date_obj'] < current_date)
    history = df[mask].sort_values('date_obj', ascending=False).head(n)

    matches = []
    for _, row in history.iterrows():
        is_home = (row['home_id'] == team_id)
        opp_id = row['away_id'] if is_home else row['home_id']
        opp_name = team_encoder.inverse_transform([opp_id])[0]

        score_str = f"{int(row['home team total goal'])} - {int(row['away team total goal'])}"
        hg, ag = row['home team total goal'], row['away team total goal']

        res_char = 'D'
        if is_home:
            if hg > ag:
                res_char = 'W'
            elif hg < ag:
                res_char = 'L'
        else:
            if ag > hg:
                res_char = 'W'
            elif ag < hg:
                res_char = 'L'

        matches.append({
            'date': row['date_obj'].strftime('%Y-%m-%d'),
            'league': row['league_name'],
            'matchup': f"vs {opp_name}" if is_home else f"@ {opp_name}",
            'score': score_str,
            'result': res_char
        })
    return matches


def generate_calibration_plot(calibration_data):
    """Generates a graph and returns it as a base64 string."""
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')

    colors = {'Home Win': 'blue', 'Away Win': 'red', 'Draw': 'green',
              'Over 2.5': 'orange', 'BTTS': 'brown'}

    for pred_type, data in calibration_data.items():
        if not data: continue
        confs = [x[0] for x in data]
        hits = [x[1] for x in data]
        color = colors.get(pred_type, 'black')
        plt.plot(confs, hits, marker='o', linewidth=2, label=pred_type, color=color)

    plt.title('Reliability Diagram: Model Calibration (Test Set)', fontsize=16)
    plt.xlabel('Predicted Confidence', fontsize=12)
    plt.ylabel('Actual Hit Rate', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url



def calculate_advanced_stats(home_lam, away_lam):
    # Safety Check
    if not isinstance(home_lam, (int, float)) or not isinstance(away_lam, (int, float)) or \
       np.isnan(home_lam) or np.isnan(away_lam):
        return {
            "win": 0.0, "draw": 0.0, "loss": 0.0,
            "over_2_5": 0.0, "under_2_5": 0.0,
            "btts_yes": 0.0, "btts_no": 0.0,
            "predicted_score": "0-0",
            "projected_home_xg": 0.0, "projected_away_xg": 0.0
        }

    max_goals = 10
    prob_matrix = np.zeros((max_goals, max_goals))
    for h in range(max_goals):
        for a in range(max_goals):
            prob_matrix[h, a] = poisson.pmf(h, home_lam) * poisson.pmf(a, away_lam)

    prob_over_2_5 = np.sum(prob_matrix[np.add.outer(np.arange(max_goals), np.arange(max_goals)) > 2.5])
    prob_btts_yes = np.sum(prob_matrix[1:, 1:])
    prob_home_win = np.sum(np.tril(prob_matrix, -1))
    prob_draw = np.sum(np.diag(prob_matrix))
    prob_away_win = np.sum(np.triu(prob_matrix, 1))

    # Predict Exact Score
    predicted_home_goals, predicted_away_goals = np.unravel_index(prob_matrix.argmax(), prob_matrix.shape)
    predicted_score = f"{predicted_home_goals}-{predicted_away_goals}"
    
    # helper
    def clean(v): return 0.0 if np.isnan(v) else float(v)

    return {
        "win": round(clean(prob_home_win * 100), 1),
        "draw": round(clean(prob_draw * 100), 1),
        "loss": round(clean(prob_away_win * 100), 1),
        "over_2_5": round(clean(prob_over_2_5 * 100), 1),
        "under_2_5": round(clean((1.0 - prob_over_2_5) * 100), 1),
        "btts_yes": round(clean(prob_btts_yes * 100), 1),
        "btts_no": round(clean((1.0 - prob_btts_yes) * 100), 1),
        "predicted_score": predicted_score,
        "projected_home_xg": round(clean(home_lam), 2),
        "projected_away_xg": round(clean(away_lam), 2)
    }


def initialize_system():
    global model_prev, model_current, model_final, model_acc, le_team, le_league, master_df, training_history
    print("\n=== FOBO AI STARTUP (Level 4: Context-Aware Engine) ===")

    # 1. Load Data
    master_df, le_team, le_league = pm.get_master_data()
    
    # --- LEVEL 7: CALCULATE ELO ---
    global elo_ratings, min_elo, max_elo
    elo_ratings, _, _ = pm.calculate_dynamic_elo(master_df)
    if elo_ratings:
        vals = elo_ratings.values()
        min_elo = min(vals)
        max_elo = max(vals)
    print(f"DEBUG: ELO System Initialized (Range: {min_elo:.1f} - {max_elo:.1f})")

    num_teams = len(le_team.classes_)
    num_leagues = len(le_league.classes_)
    
    print(f"DEBUG: Model Init - Teams: {num_teams}, Leagues: {num_leagues}")

    # 2. ROTATION LOGIC
    if not SKIP_TRAINING and os.path.exists(CURRENT_MODEL_PATH):
        print(f"Rotating Models: Moving '{CURRENT_MODEL_PATH}' to '{PREVIOUS_MODEL_PATH}'")
        shutil.copy2(CURRENT_MODEL_PATH, PREVIOUS_MODEL_PATH)

    # 3. Instantiate Models
    model_prev = LeagueAwareModel(num_teams, num_leagues).to(DEVICE)
    model_current = LeagueAwareModel(num_teams, num_leagues).to(DEVICE)
    model_final = LeagueAwareModel(num_teams, num_leagues).to(DEVICE)
    model_acc = LeagueAwareModel(num_teams, num_leagues).to(DEVICE)

    # 4. Train Fresh Model
    if not SKIP_TRAINING:
        print(f"--- [TRAINING] Fresh Level 4 Model ---")
        train_loader, _ = get_dataloader(batch_size=pm.BATCH_SIZE)
        optimizer = torch.optim.Adam(model_current.parameters(), lr=0.0001)
        # Cosine Annealing for better convergence
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        best_loss = float('inf')
        best_acc = 0.0

        patience_counter = 0

        for epoch in range(EPOCHS):
            loss, acc = train_one_epoch(model_current, train_loader, optimizer, DEVICE)
            scheduler.step()

            training_history.append({'epoch': epoch + 1, 'loss': float(loss), 'accuracy': float(acc)})
            save_history(training_history)

            # Save Best Loss (Current Model)
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
                torch.save(model_current.state_dict(), CURRENT_MODEL_PATH)
                print(f"Epoch {epoch + 1} | Loss: {loss:.4f} | Acc: {acc:.1f}% | Saved New Best Loss")
            else:
                patience_counter += 1
            
            # Save Best Acc
            if acc > best_acc:
                best_acc = acc
                torch.save(model_current.state_dict(), ACC_MODEL_PATH)
                print(f"  -> New Best Acc: {acc:.1f}%")

            if patience_counter >= PATIENCE:
                print("Early Stopping.")
                break
        
        # Save Final State
        torch.save(model_current.state_dict(), FINAL_MODEL_PATH)
        print("Saved Final Model State.")
        
        # Load the models into memory
        if os.path.exists(FINAL_MODEL_PATH):
            try: model_final.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=DEVICE))
            except: pass
        if os.path.exists(ACC_MODEL_PATH):
            try: model_acc.load_state_dict(torch.load(ACC_MODEL_PATH, map_location=DEVICE))
            except: pass
            
        # Ensure model_current has the best loss weights (it might have drifted if we didn't reload or if it ended on a non-best epoch)
        if os.path.exists(CURRENT_MODEL_PATH):
            try:
                model_current.load_state_dict(torch.load(CURRENT_MODEL_PATH, map_location=DEVICE))
            except Exception as e:
                print(f"[CRITICAL ERROR HANDLED] Failed to load Best Loss Model: {e}")
                print(">> Falling back to current in-memory weights (Last Epoch).")

    else:
        # Load all 3 if skipping training
        if os.path.exists(CURRENT_MODEL_PATH):
            try: model_current.load_state_dict(torch.load(CURRENT_MODEL_PATH, map_location=DEVICE))
            except: print("Mismatch in Current (Best Loss) Model")
        
        if os.path.exists(FINAL_MODEL_PATH):
            try: model_final.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=DEVICE))
            except: print("Mismatch in Final Model")
            
        if os.path.exists(ACC_MODEL_PATH):
            try: model_acc.load_state_dict(torch.load(ACC_MODEL_PATH, map_location=DEVICE))
            except: print("Mismatch in Best Acc Model")

    # 5. Load Previous Model
    if os.path.exists(PREVIOUS_MODEL_PATH):
        try:
            model_prev.load_state_dict(torch.load(PREVIOUS_MODEL_PATH, map_location=DEVICE))
            model_prev.eval()
        except:
            print("  Previous model incompatible. Skipping.")
            model_prev = None
    else:
        model_prev = None

    model_current.eval()
    
    # 6. Load or Train RL Agent
    global policy_agent
    policy_path = os.path.join(_MODELS_DIR, 'ppo_agent.pth')
    policy_agent = pm.PPOAgent(state_dim=RL_STATE_DIM).to(DEVICE)
    
    rl_needs_training = False
    if not os.path.exists(policy_path):
        print("RL Agent not found. Training needed.")
        rl_needs_training = True
    elif not SKIP_TRAINING:
        print("Main model retrained. Retraining RL Agent.")
        rl_needs_training = True
        
    if rl_needs_training:
        print("--- [TRAINING] RL Policy Agent (PPO) ---")
        # Use model_current (Best Loss) as the reference for RL training
        ppo_epochs = 5 if EPOCHS == 1 else 130
        pm.train_ppo_agent(model_current, policy_agent, epochs=ppo_epochs) 
        torch.save(policy_agent.state_dict(), policy_path)
        print("RL Agent trained and saved.")
    else:
        try:
             policy_agent.load_state_dict(torch.load(policy_path, map_location=DEVICE))
             policy_agent.eval()
             print("Loaded RL Policy Agent.")
        except:
             print("RL Agent load failed. Training fresh...")
             ppo_epochs = 5 if EPOCHS == 1 else 100
             pm.train_ppo_agent(model_current, policy_agent, epochs=ppo_epochs) # Fallback training
             torch.save(policy_agent.state_dict(), policy_path)
    
    # 7. Load Hybrid Model (XGBoost)
    hybrid_path = os.path.join(_MODELS_DIR, 'xgb_classifier.json')
    if xgb and os.path.exists(hybrid_path):
        try:
            hybrid_model = xgb.XGBClassifier()
            hybrid_model.load_model(hybrid_path)
            print("Loaded Hybrid Ensemble (XGBoost).")
        except Exception as e:
            print(f"Failed to load Hybrid Model: {e}")
            hybrid_model = None
    else:
        print("Hybrid Model not found or XGBoost missing.")
        hybrid_model = None

    # 8. Load LightGBM Model
    global lgbm_model
    lgbm_path = os.path.join(_MODELS_DIR, 'lgbm_classifier.joblib')
    if os.path.exists(lgbm_path):
        try:
            import joblib
            lgbm_model = joblib.load(lgbm_path)
            print("Loaded LightGBM Ensemble.")
        except Exception as e:
            print(f"Failed to load LightGBM Model: {e}")
            lgbm_model = None
    else:
        lgbm_model = None

    # 9. Load Probability Calibrator (Platt scaling for XGBoost)
    global calibrator
    calibrator_path = os.path.join(_MODELS_DIR, 'calibrator.joblib')
    if os.path.exists(calibrator_path):
        try:
            import joblib
            calibrator = joblib.load(calibrator_path)
            print("Loaded Probability Calibrator.")
        except Exception as e:
            print(f"Failed to load Calibrator: {e}")
            calibrator = None
    else:
        calibrator = None

    print("System Initialization Complete.")


# predict_with_xgboost mock function removed for full extraction feature routing


# STARTUP
initialize_system()


# --- ROUTE 1: HOME PAGE ---
@app.route('/')
def index():
    # Filter out old seasons
    old_seasons = ['2022 2023', '2023 2024', '2024 2025']
    leagues = [str(l) for l in le_league.classes_ if not any(old in str(l) for old in old_seasons)]
    league_teams = {}
    for league in leagues:
        teams_raw = master_df[master_df['league_name'] == league]['home team'].unique()
        league_teams[league] = sorted([str(t) for t in teams_raw])
    return render_template('index.html', leagues=leagues, league_teams=league_teams)


# --- ROUTE 2: PREDICTION API ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        data['home'] = data['home'].strip()
        data['away'] = data['away'].strip()
        data['league'] = pm.normalize_league_name(data['league'])
        
        home_idx = le_team.transform([data['home']])[0]
        away_idx = le_team.transform([data['away']])[0]
        league_idx = le_league.transform([data['league']])[0]

        # Date Logic: Use inputted date or today
        if 'date' in data and data['date']:
            pred_date = pd.Timestamp(data['date'])
        else:
            pred_date = pd.Timestamp.now()

        # Extract Odds if provided, else default to 0.0
        o1 = float(data.get('odds_home', 0.0) or 0.0)
        ox = float(data.get('odds_draw', 0.0) or 0.0)
        o2 = float(data.get('odds_away', 0.0) or 0.0)
        
        if o1 == 0.0 or ox == 0.0 or o2 == 0.0:
            h_elo_val = elo_ratings.get(home_idx, 1500.0)
            a_elo_val = elo_ratings.get(away_idx, 1500.0)
            o1, ox, o2 = get_implied_odds(h_elo_val, a_elo_val)
            predict_odds_source = 'implied_elo'
        else:
            predict_odds_source = 'real'

        # Predict with all 3 models using the shared internal function
        # This ensures consistency with ELOs and Tactical features
        res_final = predict_match_internal(model_final, home_idx, away_idx, league_idx, pred_date, master_df, (o1, ox, o2))
        res_loss = predict_match_internal(model_current, home_idx, away_idx, league_idx, pred_date, master_df, (o1, ox, o2))
        res_acc = predict_match_internal(model_acc, home_idx, away_idx, league_idx, pred_date, master_df, (o1, ox, o2))
            
        # Use Best Loss as the "Base" for legacy logic variables if needed, but we'll try to be generic
        # For the existing boost logic below, we'll iterate through valid results.
        all_results = [r for r in [res_final, res_loss, res_acc] if r is not None]

        # --- NEW: LOSING STREAK PENALTY (Applied to Current Model) ---
        def get_losing_streak(matches_info):
            streak = 0
            for m in matches_info:
                if m['result'] == 'L':
                    streak += 1
                else:
                    break
            return streak
        
        # We need recent matches info for streaks
        home_recent = get_recent_matches_info(home_idx, pred_date, master_df, le_team)
        away_recent = get_recent_matches_info(away_idx, pred_date, master_df, le_team)

        h_streak = get_losing_streak(home_recent)
        a_streak = get_losing_streak(away_recent)
            
        if h_streak >= 3 or a_streak >= 3:
            # Penalty Multipliers
            def get_penalty(s):
                if s >= 5: return 0.7   # -30%
                if s == 4: return 0.8   # -20%
                if s == 3: return 0.9   # -10%
                return 1.0
            
            h_factor = get_penalty(h_streak)
            a_factor = get_penalty(a_streak)
            
            # Apply to Win Probabilities
            # For Home Team: Reduce 'win'
            # For Away Team: Reduce 'loss' (which is Away Win in our key map)
            
            # We need to be careful about normalization. 
            # If we reduce Win, we should probably distribute the diff to Draw/Loss or just re-normalize.
            # Here, we will just reduce the probability and then re-normalize to 100%.
            
            streak_h_mult = h_factor
            streak_a_mult = a_factor
        else:
            streak_h_mult = 1.0
            streak_a_mult = 1.0

        # --- NEW: WINNING STREAK BOOST ---
        def get_winning_streak(matches_info):
            streak = 0
            for m in matches_info:
                if m['result'] == 'W':
                    streak += 1
                else:
                    break
            return streak

        h_win_streak = get_winning_streak(home_recent)
        a_win_streak = get_winning_streak(away_recent)
        
        if h_win_streak >= 2: streak_h_mult *= (1.0 + (h_win_streak * 0.03)) # +3% per win in streak
        if a_win_streak >= 2: streak_a_mult *= (1.0 + (a_win_streak * 0.03))

        # --- NEW: FIGHTING SPIRIT / UNDERDOG BOOST ---
        # Condition 1: Low Rank vs High Rank Win
        # Condition 2: Close High Scoring Loss (e.g. 2-3, 3-4)

        def get_league_rankings(league_name, check_date):
            # Simple points calc: W=3, D=1
            # Filter matches up to date
            subset = master_df[(master_df['league_name'] == league_name) & (master_df['date_obj'] < check_date)]
            points = {}
            # This is a bit expensive to do every time, but precise.
            # Optimization: Could cache or approximate with recent form, but user asked for "Rank"
            for _, m in subset.iterrows():
                h, a = m['home team'], m['away team']
                hg, ag = m['home team total goal'], m['away team total goal']
                if h not in points: points[h] = 0
                if a not in points: points[a] = 0
                if hg > ag: points[h] += 3
                elif ag > hg: points[a] += 3
                else:
                    points[h] += 1
                    points[a] += 1
            
            # Sort descending
            sorted_teams = sorted(points.keys(), key=lambda k: points[k], reverse=True)
            # Return dict {team: rank_index} (1-based)
            return {t: i+1 for i, t in enumerate(sorted_teams)}

        def check_boost_condition(team_id, team_name, recent_matches, league_name):
            if not recent_matches: return 0.0
            
            # Re-query last match raw
            last_mask = ((master_df['home_id'] == team_id) | (master_df['away_id'] == team_id)) & (master_df['date_obj'] < pred_date)
            last_match_df = master_df[last_mask].sort_values('date_obj').tail(1)
            
            if last_match_df.empty: return 0.0
            
            last = last_match_df.iloc[0]
            match_date = last['date_obj']
            
            # Determine Identity and Result
            if last['home_id'] == team_id:
                goals_for, goals_against = last['home team total goal'], last['away team total goal']
                opp_name = last['away team']
                won = goals_for > goals_against
                was_home = True
            else:
                goals_for, goals_against = last['away team total goal'], last['home team total goal']
                opp_name = last['home team']
                won = goals_for > goals_against
                was_home = False

            # Logic 1: Fighting Spirit (Scores 2+ goals in a high scoring loss)
            # "High scoring game" (Total >= 5, e.g. 3-2, 4-2) AND Losing
            total_goals = goals_for + goals_against
            if not won and goals_for >= 2 and total_goals >= 5:
                # Away Fighting Spirit is worth more (15%) vs Home (10%)
                return 1.15 if not was_home else 1.10

            # Logic 2: Giant Killing (Low Ranked beats/draws High Ranked)
            if won or (last['home team total goal'] == last['away team total goal']): # Win or Draw
                ranks = get_league_rankings(league_name, match_date)
                my_rank = ranks.get(team_name, 99)
                opp_rank = ranks.get(opp_name, 99)
                
                rank_diff = my_rank - opp_rank # Positive means I am lower ranked (e.g. Me=15, Opp=2 -> Diff=13)
                
                # Giant Killing (Win)
                if won and rank_diff >= 8: 
                    return 1.20 if not was_home else 1.15
                
                # Impressive Draw (Draw vs Top Tier)
                # If I am bottom half (>10) and Opponent is Top 6 (<=6)
                if not won and my_rank > 10 and opp_rank <= 6:
                    return 1.10 # +10% Confidence for holding a giant
                    
            return 1.0

        # Get Team Names for Rank Lookup
        h_name = data['home']
        a_name = data['away']
        curr_league = data['league']
        
        # --- 1. EXISTING BOOSTS (Underdog/Fighting Spirit) ---
        h_boost = check_boost_condition(home_idx, h_name, home_recent, curr_league)
        a_boost = check_boost_condition(away_idx, a_name, away_recent, curr_league)
        
        # --- 2. FATIGUE / REST DAYS LOGIC ---
        def get_days_since_last(team_id):
            last_mask = ((master_df['home_id'] == team_id) | (master_df['away_id'] == team_id)) & (master_df['date_obj'] < pred_date)
            last_matches = master_df[last_mask].sort_values('date_obj').tail(1)
            if last_matches.empty: return 99 # Fresh
            last_date = last_matches.iloc[0]['date_obj']
            return (pred_date - last_date).days

        h_rest = get_days_since_last(home_idx)
        a_rest = get_days_since_last(away_idx)
        
        # Penalty if Rest < 3 AND Opponent > 6
        if h_rest < 3 and a_rest > 6:
            h_boost *= 0.9 # Fatigue Penalty
            a_boost *= 1.1 # Freshness Boost
        elif a_rest < 3 and h_rest > 6:
            a_boost *= 0.9
            h_boost *= 1.1

        # --- 3. HEAD-TO-HEAD (BOGEY TEAM) LOGIC ---
        # Check last 5 H2H matches
        mask_h2h = (((master_df['home_id'] == home_idx) & (master_df['away_id'] == away_idx)) |
                    ((master_df['home_id'] == away_idx) & (master_df['away_id'] == home_idx))) & (master_df['date_obj'] < pred_date)
        h2h_matches = master_df[mask_h2h].sort_values('date_obj').tail(5)
        
        h_h2h_wins = 0
        a_h2h_wins = 0
        for _, m in h2h_matches.iterrows():
            if m['home_id'] == home_idx: # Home is Home
                if m['home team total goal'] > m['away team total goal']: h_h2h_wins += 1
                elif m['away team total goal'] > m['home team total goal']: a_h2h_wins += 1
            else: # Home is Away
                if m['away team total goal'] > m['home team total goal']: h_h2h_wins += 1
                elif m['home team total goal'] > m['away team total goal']: a_h2h_wins += 1
        
        if h_h2h_wins >= 4: h_boost *= 1.15 # Massive psychological edge
        if a_h2h_wins >= 4: a_boost *= 1.15

        # --- 4. DEFENSIVE FORTRESS (CLEAN SHEETS) ---
        # If 3 consecutive clean sheets -> Boost Draw and Under 2.5
        def check_clean_sheets(matches_info):
            cs = 0
            for m in matches_info:
                # Parse score "1 - 0"
                parts = m['score'].split('-')
                hg, ag = int(parts[0].strip()), int(parts[1].strip())
                # Check if 'we' conceded 0
                # Helper returns "vs Opp" or "@ Opp" to denote Venue
                is_home_match = "vs" in m['matchup']
                conceded = ag if is_home_match else hg
                if conceded == 0: cs += 1
                else: break
            return cs

        h_cs = check_clean_sheets(home_recent)
        a_cs = check_clean_sheets(away_recent)
        
        draw_mult = 1.0
        under_mult = 1.0
        
        if h_cs >= 3: 
            draw_mult *= 1.1
            under_mult *= 1.15
        if a_cs >= 3:
            draw_mult *= 1.1
            under_mult *= 1.15

        # --- 5. AVERAGE RAW MODEL OUTPUTS (Before contextual adjustments) ---
        # Individual model results (res_final, res_loss, res_acc) are kept as pure model
        # outputs so the UI comparison table shows unmodified per-checkpoint probabilities.
        # Contextual boosts are applied once to the consensus average below.
        res_avg = {}
        if all_results:
            keys = ['win', 'draw', 'loss', 'over_2_5', 'under_2_5', 'btts_yes', 'btts_no']
            for k in keys:
                val = sum([r[k] for r in all_results]) / len(all_results)
                res_avg[k] = round(val, 1)

            # Score is a string ("1-0") — use best-loss model's score; fall back to first.
            rep = res_loss if res_loss else all_results[0]
            res_avg['predicted_score'] = rep['predicted_score']
        else:
            res_avg = None

        # --- 6. APPLY CONTEXTUAL BOOSTS TO CONSENSUS AVERAGE (Single Pass) ---
        # Boosts are external contextual signals (form, fatigue, H2H, fortress).
        # Applying them once to the averaged output is cleaner than boosting each model
        # independently (which causes three separate renormalisations before averaging).
        if res_avg:
            m_h_boost = h_boost * streak_h_mult
            m_a_boost = a_boost * streak_a_mult
            m_draw_mult = draw_mult
            m_under_mult = under_mult

            # Tight Match Theory: High BTTS (>55) but Low Over 2.5 (<45) → draw boost
            if res_avg['btts_yes'] >= 55.0 and res_avg['over_2_5'] < 45.0:
                m_draw_mult *= 1.15

            # Apply Win/Draw/Loss Multipliers
            if m_h_boost != 1.0 or m_a_boost != 1.0 or m_draw_mult != 1.0:
                w = res_avg['win'] * m_h_boost
                d = res_avg['draw'] * m_draw_mult
                l = res_avg['loss'] * m_a_boost
                tot = w + d + l
                if tot > 0:
                    res_avg['win'] = round((w / tot) * 100, 1)
                    res_avg['draw'] = round((d / tot) * 100, 1)
                    res_avg['loss'] = round((l / tot) * 100, 1)

            # Apply Over/Under Multipliers
            if m_under_mult != 1.0:
                u = res_avg['under_2_5'] * m_under_mult
                o = res_avg['over_2_5']
                tot = u + o
                if tot > 0:
                    res_avg['under_2_5'] = round((u / tot) * 100, 1)
                    res_avg['over_2_5'] = round((o / tot) * 100, 1)
        
        # --- TECHNICAL FACTOR REPORT ---
        tech_report = {
            'elo_ratings': {
                'home_raw': round(elo_ratings.get(home_idx, 1500.0), 1),
                'away_raw': round(elo_ratings.get(away_idx, 1500.0), 1),
                'home_norm': round((elo_ratings.get(home_idx, 1500.0) - min_elo) / (max_elo - min_elo), 4),
                'away_norm': round((elo_ratings.get(away_idx, 1500.0) - min_elo) / (max_elo - min_elo), 4)
            },
            'streaks': {
                'home_loss_streak': h_streak,
                'away_loss_streak': a_streak,
                'home_win_streak': h_win_streak,
                'away_win_streak': a_win_streak,
                'streak_multipliers': {'home': round(streak_h_mult, 2), 'away': round(streak_a_mult, 2)}
            },
            'rest_days': {
                'home': h_rest if h_rest != 99 else "N/A",
                'away': a_rest if a_rest != 99 else "N/A",
                'fatigue_mod': "Active" if (h_rest < 3 and a_rest > 6) or (a_rest < 3 and h_rest > 6) else "None"
            },
            'boosts': {
                'home_situational': round(h_boost, 2), # Fighting Spirit / Underdog
                'away_situational': round(a_boost, 2),
                'h2h_dominance': "Home" if h_h2h_wins >= 4 else ("Away" if a_h2h_wins >= 4 else "None"),
                'defensive_fortress': {
                    'home_clean_sheets': h_cs,
                    'away_clean_sheets': a_cs,
                    'active': h_cs >= 3 or a_cs >= 3
                }
            },
            'tight_match_logic': res_avg['btts_yes'] >= 55.0 and res_avg['over_2_5'] < 45.0 if res_avg else False
        }

        # --- RL AGENT RECOMMENDATION (PPO with Context) ---
        rl_data = {'recommendation': 'N/A', 'confidence': 0.0}
        
        # We MUST use model_current to match the PPO training environment
        if policy_agent and model_current:
            try:
                # Prepare Tensors for RL Agent
                # Re-generate sequences (since they are local) or reuse if we refactor.
                # For now, we generate them here.
                h_seq_np = pm.get_team_history(home_idx, pred_date, master_df)
                a_seq_np = pm.get_team_history(away_idx, pred_date, master_df)
                
                h_seq_tensor = torch.from_numpy(h_seq_np).float().unsqueeze(0).to(DEVICE)
                a_seq_tensor = torch.from_numpy(a_seq_np).float().unsqueeze(0).to(DEVICE)
                
                h_id_tensor = torch.tensor([home_idx], device=DEVICE)
                a_id_tensor = torch.tensor([away_idx], device=DEVICE)
                l_id_tensor = torch.tensor([league_idx], device=DEVICE)
                
                # ELO Tensors
                h_elo_val = elo_ratings.get(home_idx, 1500.0)
                a_elo_val = elo_ratings.get(away_idx, 1500.0)
                if max_elo == min_elo:
                    h_elo_n, a_elo_n = 0.0, 0.0
                else:
                    h_elo_n = (h_elo_val - min_elo) / (max_elo - min_elo)
                    a_elo_n = (a_elo_val - min_elo) / (max_elo - min_elo)
                
                h_elo_tensor = torch.tensor([h_elo_n], dtype=torch.float32).to(DEVICE)
                a_elo_tensor = torch.tensor([a_elo_n], dtype=torch.float32).to(DEVICE)

                # Extract Internal Features from the Main Model
                # This gives the PPO Agent the "Context" (Confidence, Confusion, etc.)
                state = model_current.extract_features(h_seq_tensor, a_seq_tensor, h_id_tensor, a_id_tensor, l_id_tensor, h_elo_tensor, a_elo_tensor)
                
                # Get Action Probabilities from Actor
                with torch.no_grad():
                     probs = policy_agent.actor(state)
                     action = torch.argmax(probs).item()
                     conf = probs[0, action].item() * 100.0
                     
                     if action == 0: rec = "BET HOME"
                     elif action == 1: rec = "BET DRAW"
                     elif action == 2: rec = "BET AWAY"
                     else: rec = "PASS (Uncertain)"
                     
                     rl_data = {'recommendation': rec, 'confidence': round(conf, 1)}
            except Exception as e:
                print(f"RL Agent Error: {e}")

        return jsonify({
            'status': 'success',
            'model_final': res_final,
            'model_loss': res_loss,
            'model_acc': res_acc,
            'average': res_avg,
            'technical_analysis': tech_report,
            'home_recent': home_recent,
            'away_recent': away_recent,
            'rl_agent': rl_data,
            'odds_source': predict_odds_source,
            'odds_used': [round(o1, 2), round(ox, 2), round(o2, 2)]
        })

    except Exception as e:
        print(f"PREDICTION ERROR: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})


# --- ROUTE 3: TRAINING HISTORY API ---
@app.route('/get_training_history', methods=['GET'])
def get_training_history():
    return jsonify(training_history)


# --- ROUTE 4: CALIBRATION API (Dual Model Support) ---
@app.route('/get_calibration_deprecated', methods=['GET'])
def get_calibration_deprecated():
    print("[CALIBRATION] Analysis Started...")

    # 1. Create Test Set (Last 20% of each league)
    test_rows = []
    leagues = master_df['league_name'].unique()
    for lg in leagues:
        lg_df = master_df[master_df['league_name'] == lg].sort_values('date_obj')
        if len(lg_df) < 10: continue
        cutoff = int(len(lg_df) * 0.8)
        test_chunk = lg_df.iloc[cutoff:]
        test_rows.append(test_chunk)

    if not test_rows:
        return jsonify({'status': 'error', 'message': 'Not enough data for calibration'})

    test_df = pd.concat(test_rows)
    print(f"[CALIBRATION] Test Set Size: {len(test_df)} matches")

    # Helper to evaluate a model
    def evaluate_model(model_instance):
        if model_instance is None: return []
        model_instance.eval()
        res_list = []
        with torch.no_grad():
            for idx, row in test_df.iterrows():
                h_id, a_id, l_id = row['home_id'], row['away_id'], row['league_id']
                date = row['date_obj']
                # Basic History features (same as before)
                h_seq = torch.from_numpy(pm.get_team_history(h_id, date, master_df)).float().unsqueeze(0).to(DEVICE)
                a_seq = torch.from_numpy(pm.get_team_history(a_id, date, master_df)).float().unsqueeze(0).to(DEVICE)
                
                # Extract Odds from Master DF (ensure columns exist or default)
                o1 = float(row.get('odds_1', 0.0))
                ox = float(row.get('odds_x', 0.0))
                o2 = float(row.get('odds_2', 0.0))
                odds_t = torch.tensor([o1, ox, o2], dtype=torch.float32).unsqueeze(0).to(DEVICE)

                try:
                    # ELO
                    h_elo_val = elo_ratings.get(h_id, 1500.0)
                    a_elo_val = elo_ratings.get(a_id, 1500.0)
                    if max_elo == min_elo:
                        h_elo_norm = 0.0; a_elo_norm = 0.0
                    else:
                        h_elo_norm = (h_elo_val - min_elo) / (max_elo - min_elo)
                        a_elo_norm = (a_elo_val - min_elo) / (max_elo - min_elo)
                    h_elo_t = torch.tensor([h_elo_norm], dtype=torch.float32).to(DEVICE)
                    a_elo_t = torch.tensor([a_elo_norm], dtype=torch.float32).to(DEVICE)
                    
                    pred = model_instance(h_seq, a_seq, torch.tensor([h_id], device=DEVICE),
                                     torch.tensor([a_id], device=DEVICE), torch.tensor([l_id], device=DEVICE),
                                     odds_t, h_elo_t, a_elo_t)
                    lambdas = pred[0]
                    h_lam, a_lam = lambdas[0, 0].item(), lambdas[0, 1].item()
                    stats = calculate_advanced_stats(h_lam, a_lam) 
                    
                    hg, ag = row['home team total goal'], row['away team total goal']
                    actual_res = 'Draw'
                    if hg > ag: actual_res = 'Home Win'
                    elif ag > hg: actual_res = 'Away Win'
                    actual_total = hg + ag
                    actual_btts = (hg > 0 and ag > 0)
                    
                    current_league_name = str(row['league_name']).upper()

                    res_list.append({
                        'League': current_league_name,
                        'Home Win': (stats['win'], 1 if actual_res == 'Home Win' else 0),
                        'Away Win': (stats['loss'], 1 if actual_res == 'Away Win' else 0),
                        'Draw': (stats['draw'], 1 if actual_res == 'Draw' else 0),
                        'Over 2.5': (stats['over_2_5'], 1 if actual_total > 2.5 else 0),
                        'BTTS': (stats['btts_yes'], 1 if actual_btts else 0)
                    })
                except Exception as e:
                    pass
        
        # --- RL EVALUATION (Only if checking Model Current) ---
        rl_data = {'bets': 0, 'wins': 0, 'losses': 0, 'passes': 0, 'yield': 0.0}
        
        if model_instance == model_current and policy_agent is not None:
             with torch.no_grad():
                for idx, row in test_df.iterrows():
                    try:
                        h_id, a_id, l_id = row['home_id'], row['away_id'], row['league_id']
                        date = row['date_obj']
                        h_seq = torch.from_numpy(pm.get_team_history(h_id, date, master_df)).float().unsqueeze(0).to(DEVICE)
                        a_seq = torch.from_numpy(pm.get_team_history(a_id, date, master_df)).float().unsqueeze(0).to(DEVICE)
                        
                        # Odds
                        o1 = float(row.get('odds_1', 0.0))
                        ox = float(row.get('odds_x', 0.0))
                        o2 = float(row.get('odds_2', 0.0))
                        odds_t = torch.tensor([o1, ox, o2], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        
                        # ELO
                        h_elo_val = elo_ratings.get(h_id, 1500.0)
                        a_elo_val = elo_ratings.get(a_id, 1500.0)
                        if max_elo == min_elo:
                            h_elo_norm = 0.0; a_elo_norm = 0.0
                        else:
                            h_elo_norm = (h_elo_val - min_elo) / (max_elo - min_elo)
                            a_elo_norm = (a_elo_val - min_elo) / (max_elo - min_elo)
                        h_elo_t = torch.tensor([h_elo_norm], dtype=torch.float32).to(DEVICE)
                        a_elo_t = torch.tensor([a_elo_norm], dtype=torch.float32).to(DEVICE)
                        
                        # Main Model Pred
                        pred = model_instance(h_seq, a_seq, torch.tensor([h_id], device=DEVICE),
                                         torch.tensor([a_id], device=DEVICE), torch.tensor([l_id], device=DEVICE),
                                         odds_t, h_elo_t, a_elo_t)
                        lambdas = pred[0]
                        h_lam, a_lam = lambdas[0, 0].item(), lambdas[0, 1].item()
                        stats = calculate_advanced_stats(h_lam, a_lam)
                        
                        # Feed to RL (PPO) - Only if it's the Current Model (which trained the PPO)
                        # PPO Agent needs the Internal State of the model
                        state = model_instance.extract_features(h_seq, a_seq, torch.tensor([h_id], device=DEVICE),
                                         torch.tensor([a_id], device=DEVICE), torch.tensor([l_id], device=DEVICE),
                                         h_elo_t, a_elo_t)
                        
                        probs = policy_agent.actor(state)
                        action = torch.argmax(probs).item() # 0=Home, 1=Draw, 2=Away, 3=Pass
                        
                        # Check Result
                        hg, ag = row['home team total goal'], row['away team total goal']
                        actual = 'Draw'
                        if hg > ag: actual = 'Home'
                        elif ag > hg: actual = 'Away'
                        
                        if action == 3: # Pass
                            rl_data['passes'] += 1
                        else:
                            rl_data['bets'] += 1
                            won = False
                            if action == 0 and actual == 'Home': won = True
                            elif action == 1 and actual == 'Draw': won = True
                            elif action == 2 and actual == 'Away': won = True
                            
                            if won:
                                rl_data['wins'] += 1
                                rl_data['yield'] += 1.0 # Flat 1 unit win
                            else:
                                rl_data['losses'] += 1
                                rl_data['yield'] -= 1.0 # Flat 1 unit loss
                    except: pass

        return res_list, rl_data

    # Evaluate BOTH
    results_curr, rl_stats = evaluate_model(model_current)
    results_prev, _ = evaluate_model(model_prev) # Ignore RL for prev

    bucket_defs = {
        'Home Win': [(50, 60), (60, 70), (70, 80), (80, 90), (90, 101)],
        'Away Win': [(50, 60), (60, 70), (70, 80), (80, 90), (90, 101)],
        'Draw': [(30, 40), (40, 50), (50, 60), (60, 70)],
        'Over 2.5': [(50, 60), (60, 70), (70, 80), (80, 90), (90, 101)],
        'BTTS': [(50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]
    }

    def process_results_to_table(raw_results):
        if not raw_results: return []
        table = []
        for p_type, buckets in bucket_defs.items():
            for (low, high) in buckets:
                relevant = [x for x in raw_results if low <= x[p_type][0] < high]
                if relevant:
                    total = len(relevant)
                    hits = sum(r[p_type][1] for r in relevant)
                    hit_rate = (hits / total) * 100
                    mid = (low + high) / 2
                    roi = ((hit_rate/100) * (100/mid) - 1) * 100
                    if np.isnan(roi) or np.isinf(roi): roi = 0.0
                    
                    label = f"{low}-{high}%" if high <= 90 else f"{low}%+"
                    status = "Value" if roi > 5 else ("Accurate" if roi > -5 else "Check")
                    
                    table.append({
                        'Type': p_type, 'Confidence': label,
                        'HitRate': f"{hit_rate:.1f}%", 
                        'Profit': f"{roi:+.1f}%",
                        'Status': status
                    })
        return table
    
    table_curr = process_results_to_table(results_curr)
    table_prev = process_results_to_table(results_prev)
    
    # Generate graph only for current to keep UI simple or we can omit
    # Re-using existing graph logic just for current so logic doesn't break
    graph_data_points = {k: [] for k in bucket_defs.keys()}
    for p_type, buckets in bucket_defs.items():
         for (low, high) in buckets:
            relevant = [x for x in results_curr if low <= x[p_type][0] < high]
            if relevant:
                 hits = sum(r[p_type][1] for r in relevant)
                 total = len(relevant)
                 midpoint = (low + high) / 2
                 graph_data_points[p_type].append((midpoint / 100.0, hits / total))

    plot_url = generate_calibration_plot(graph_data_points)
    
    return jsonify({
        'status': 'success', 
        'current_table': table_curr,
        'prev_table': table_prev,
        'graph_image': plot_url,
        'rl_stats': rl_stats
    })


# --- ROUTE 5: ADD DATA API ---
@app.route('/add_match', methods=['POST'])
def add_match():
    data = request.json
    try:
        date_str = data['date']
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        if dt.month >= 8:
            s_start, s_end = dt.year, dt.year + 1
        else:
            s_start, s_end = dt.year - 1, dt.year
        league_tag = data['league'].replace(' ', '_').upper()
        # Search in both current and old_matches directories
        pattern_current = f"*{league_tag}*{s_start}_{s_end}*.csv"
        pattern_old = f"old_matches/*{league_tag}*{s_start}_{s_end}*.csv"
        
        matching_files = glob.glob(pattern_current) + glob.glob(pattern_old)
        
        if matching_files:
            target_file = matching_files[0]
            df = pd.read_csv(target_file)
            new_row = {}
            input_map = {'date': data['date'], 'home team': data['home'], 'away team': data['away'],
                         'home team total goal': data['hg'], 'away team total goal': data['ag'],
                         'league': data['league']}
            for col in df.columns:
                col_clean = col.strip().lower()
                if col_clean in input_map:
                    new_row[col] = input_map[col_clean]
                else:
                    new_row[col] = pd.NA
            pd.DataFrame([new_row]).to_csv(target_file, mode='a', header=False, index=False)
            msg = f"Added to {target_file}."
        else:
            target_file = f"FOOTBALL_{league_tag}_{s_start}_{s_end}_RESULTS.csv"
            new_row = {'Date': data['date'], 'League': data['league'], 'Home Team': data['home'],
                       'Away Team': data['away'], 'Home Team Total Goal': data['hg'],
                       'Away Team Total Goal': data['ag']}
            pd.DataFrame([new_row]).to_csv(target_file, index=False)
            msg = f"Created {target_file}."
        return jsonify({'status': 'success', 'message': msg})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})




# --- ROUTE 6: POWER RANKINGS API ---
@app.route('/get_power_rankings', methods=['POST'])
def get_power_rankings():
    try:
        req = request.json
        target_league = req.get('league')
        
        _, ranking_data = calculate_power_rankings(target_league)
        
        return jsonify({'status': 'success', 'rankings': ranking_data})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})


# --- ROUTE 7: DAILY CALENDAR API ---
@app.route('/get_daily_matches', methods=['POST'])
def get_daily_matches():
    try:
        req = request.json
        start_date_str = req.get('start_date') or req.get('date')
        end_date_str = req.get('end_date') or start_date_str
        
        # Load upcoming matches
        upcoming_file = 'UPCOMING_MATCHES.csv'
        if not os.path.exists(upcoming_file):
             # Fallback to parent directory
             if os.path.exists(os.path.join('..', upcoming_file)):
                 upcoming_file = os.path.join('..', upcoming_file)
             else:
                 # Graceful fallback: return empty list rather than error
                 return jsonify({'status': 'success', 'matches': []})
             
        df = pd.read_csv(upcoming_file)
        
        # Filter by date range
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        daily_matches = df[(df['Date'] >= start_date_str) & (df['Date'] <= end_date_str)].copy()
        
        if daily_matches.empty:
             return jsonify({'status': 'success', 'matches': []})

        # Sort by Time then League
        if 'Time' in daily_matches.columns:
            daily_matches.sort_values(by=['Time', 'League'], inplace=True)
             
        results = []
        
        for _, row in daily_matches.iterrows():
            league_full = row['League']
            h_team = row['Home']
            a_team = row['Away']
            time_val = row.get('Time', '00:00')
            
            # Clean league name
            league_clean = pm.normalize_league_name(league_full)
            
            try:
                # Find IDs using Global Encoders
                # We need to handle teams not in training set safely
                if h_team not in le_team.classes_ or a_team not in le_team.classes_:
                    continue # Skip unknown teams
                    
                h_id = le_team.transform([h_team])[0]
                a_id = le_team.transform([a_team])[0]
                
                # Try to fuzzy match league or skip if not found
                if league_clean not in le_league.classes_:
                    # Fallback: Try mapping if defined
                    for key, val in LEAGUE_MAPPING.items():
                        if key in league_clean:
                            league_clean = val # transform will need exact internal code if we used that?
                            # Actually encoder uses raw names usually.
                            pass
                    if league_clean not in le_league.classes_:
                        continue
                        
                l_id = le_league.transform([league_clean])[0]
                
                # Get Odds (default to 0.0 if not available)
                o1 = float(row.get('Odds_1', 0.0)) if pd.notna(row.get('Odds_1')) else 0.0
                ox = float(row.get('Odds_X', 0.0)) if pd.notna(row.get('Odds_X')) else 0.0
                o2 = float(row.get('Odds_2', 0.0)) if pd.notna(row.get('Odds_2')) else 0.0
                odds_vals = (o1, ox, o2)

                # Predict with 3 models
                res_final = predict_match_internal(model_final, h_id, a_id, l_id, row['Date'], master_df, odds_vals)
                res_loss = predict_match_internal(model_current, h_id, a_id, l_id, row['Date'], master_df, odds_vals)
                res_acc = predict_match_internal(model_acc, h_id, a_id, l_id, row['Date'], master_df, odds_vals)
                
                # Average
                res_avg = {}
                keys = ['win', 'draw', 'loss', 'over_2_5', 'btts_yes', 'projected_home_xg', 'projected_away_xg']
                
                valid_res = [r for r in [res_final, res_loss, res_acc] if r]
                
                if valid_res:
                    for k in keys:
                        # Handle average for xG separately if needed, or just sum/len like others
                        # Since they are floats, the generic round logic works fine
                        vals = [r.get(k, 0) for r in valid_res]
                        res_avg[k] = round(sum(vals) / len(vals), 2)
                    # Use Loss model score as primary, or Final? Let's use Final if avail
                    score_res = res_final if res_final else (res_loss if res_loss else res_acc)
                    res_avg['predicted_score'] = score_res.get('predicted_score', '-')
                else:
                    res_avg = {k: 0 for k in keys}
                    res_avg['predicted_score'] = '-'

                # Get Detailed Last 9 Matches Form (Optimal AI Found Value)
                def get_detailed_form(team_id):
                    # Uses global helper defined earlier in file
                    matches = get_recent_matches_info(team_id, pd.Timestamp(row['Date']), master_df, le_team, n=9)
                    return matches

                h_form_details = get_detailed_form(h_id)
                a_form_details = get_detailed_form(a_id)
                
                # Helper to safe-get
                def sget(r, k): return r.get(k, 0) if r else 0

                # RL AGENT ADVICE
                rl_label = "N/A"
                rl_conf = 0.0
                rl_color = "secondary"
                rl_icon = "❓"
                rl_desc = "No Advice"
                
                if policy_agent:
                    try:
                        # Prepare Tensors for Feature Extraction
                        date_obj = pd.to_datetime(row['Date'])
                        h_seq = torch.from_numpy(pm.get_team_history(h_id, date_obj, master_df)).float().unsqueeze(0).to(DEVICE)
                        a_seq = torch.from_numpy(pm.get_team_history(a_id, date_obj, master_df)).float().unsqueeze(0).to(DEVICE)
                        h_id_t = torch.tensor([h_id], device=DEVICE)
                        a_id_t = torch.tensor([a_id], device=DEVICE)
                        l_id_t = torch.tensor([l_id], device=DEVICE)

                        if model_current:
                            with torch.no_grad():
                                # ELO
                                h_elo_val = elo_ratings.get(h_id, 1500.0)
                                a_elo_val = elo_ratings.get(a_id, 1500.0)
                                if max_elo == min_elo:
                                    h_elo_norm = 0.0; a_elo_norm = 0.0
                                else:
                                    h_elo_norm = (h_elo_val - min_elo) / (max_elo - min_elo)
                                    a_elo_norm = (a_elo_val - min_elo) / (max_elo - min_elo)
                                h_elo_t = torch.tensor([h_elo_norm], dtype=torch.float32).to(DEVICE)
                                a_elo_t = torch.tensor([a_elo_norm], dtype=torch.float32).to(DEVICE)

                                # Extract Context Features [1, 1056]
                                state = model_current.extract_features(h_seq, a_seq, h_id_t, a_id_t, l_id_t, h_elo_t, a_elo_t)
                                
                                # Use .actor() to get policy probabilities
                                rl_probs = policy_agent.actor(state)
                                rl_act = torch.argmax(rl_probs).item()
                                rl_conf = rl_probs[0, rl_act].item() * 100
                                
                                if rl_act == 0:
                                    rl_label = "HOME"
                                    rl_color = "success"
                                    rl_icon = "🏠"
                                    rl_desc = "High Confidence Home Win"
                                elif rl_act == 1:
                                    rl_label = "DRAW"
                                    rl_color = "warning"
                                    rl_icon = "🤝"
                                    rl_desc = "Value in Draw"
                                elif rl_act == 2:
                                    rl_label = "AWAY"
                                    rl_color = "success"
                                    rl_icon = "✈️"
                                    rl_desc = "High Confidence Away Win"
                                else: # 3 = Pass
                                    rl_label = "PASS (Uncertain)"
                                    rl_color = "warning"
                                    rl_icon = "🛑"
                                    rl_desc = "Risk too high / No Value"
                    except Exception as e:
                        print(f"RL Error: {e}")

                # Calculate Ranks for Display
                rank_map, _ = calculate_power_rankings(league_clean) # Use the same clean name logic
                h_rank = rank_map.get(h_team, '-')
                a_rank = rank_map.get(a_team, '-')
                
                h_team_display = f"{h_team} (Rank: {h_rank})"
                a_team_display = f"{a_team} (Rank: {a_rank})"

                # RECOMMENDATION LOGIC
                rec_label = "-"
                rec_conf = 0.0
                rec_color = "secondary"
                
                candidates = []
                avg_win = res_avg.get('win', 0)
                avg_loss = res_avg.get('loss', 0)
                avg_o25 = res_avg.get('over_2_5', 0)
                avg_btts = res_avg.get('btts_yes', 0)
                
                if avg_win > 45: candidates.append((avg_win, "Home Win", "success"))
                if avg_loss > 45: candidates.append((avg_loss, "Away Win", "danger"))
                if avg_o25 > 58: candidates.append((avg_o25, "Over 2.5 Goals", "warning text-dark"))
                if avg_btts > 58: candidates.append((avg_btts, "BTTS Yes", "info text-dark"))
                
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    best = candidates[0]
                    rec_conf = best[0]
                    rec_label = best[1]
                    rec_color = best[2]

                results.append({
                    'date': row['Date'],
                    'time': time_val,
                    'league': league_clean,
                    'home': h_team_display,
                    'away': a_team_display,
                    'match': f"{h_team} vs {a_team}",
                    'home_form': h_form_details,
                    'away_form': a_form_details,
                    
                    # Recommendation
                    'rec_label': rec_label,
                    'rec_conf': rec_conf,
                    'rec_color': rec_color,
                    
                    # RL Advice
                    'rl_label': rl_label,
                    'rl_conf': f"{rl_conf:.1f}",
                    'rl_color': rl_color,
                    'rl_icon': rl_icon,
                    'rl_desc': rl_desc,
                    
                    # Final Model
                    'final_win': sget(res_final, 'win'),
                    'final_draw': sget(res_final, 'draw'),
                    'final_loss': sget(res_final, 'loss'),
                    'final_o25': sget(res_final, 'over_2_5'),
                    'final_btts': sget(res_final, 'btts_yes'),
                    
                    # Best Loss Model (Current)
                    'loss_win': sget(res_loss, 'win'),
                    'loss_draw': sget(res_loss, 'draw'),
                    'loss_loss': sget(res_loss, 'loss'),
                    'loss_o25': sget(res_loss, 'over_2_5'),
                    'loss_btts': sget(res_loss, 'btts_yes'),

                    # Best Acc Model
                    'acc_win': sget(res_acc, 'win'),
                    'acc_draw': sget(res_acc, 'draw'),
                    'acc_loss': sget(res_acc, 'loss'),
                    'acc_o25': sget(res_acc, 'over_2_5'),
                    'acc_btts': sget(res_acc, 'btts_yes'),

                    # Average
                    'avg_win': res_avg.get('win', 0),
                    'avg_draw': res_avg.get('draw', 0),
                    'avg_loss': res_avg.get('loss', 0),
                    'avg_o25': res_avg.get('over_2_5', 0),
                    'avg_btts': res_avg.get('btts_yes', 0),
                    'avg_home_xg': res_avg.get('projected_home_xg', 0),
                    'avg_away_xg': res_avg.get('projected_away_xg', 0),
                    'pred_score': res_avg.get('predicted_score', '-'),
                    # Odds transparency: 'real' = bookmaker odds, 'implied_elo' = ELO fallback
                    'odds_source': (res_loss or res_final or res_acc or {}).get('odds_source', 'implied_elo'),
                    'odds_used': (res_loss or res_final or res_acc or {}).get('odds_used', [0, 0, 0])
                })
            except Exception as e:
                print(f"Skipping match {h_team} vs {a_team}: {e}")
                traceback.print_exc()
                continue
            
        return jsonify({'status': 'success', 'matches': results})
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# --- HELPER FOR INTERNAL PREDICTION ---
def predict_match_internal(model, h_id, a_id, l_id, date_str, df, odds_tuple=(0.0, 0.0, 0.0)):
    if model is None: return None
    model.eval()
    
    try:
        date_obj = pd.to_datetime(date_str)
    except:
        date_obj = datetime.now()

    h_seq = torch.from_numpy(pm.get_team_history(h_id, date_obj, df)).float().unsqueeze(0).to(DEVICE)
    a_seq = torch.from_numpy(pm.get_team_history(a_id, date_obj, df)).float().unsqueeze(0).to(DEVICE)
    
    # Safety net: impute missing odds using ELO to prevent lambda collapse
    o1, ox, o2 = float(odds_tuple[0]), float(odds_tuple[1]), float(odds_tuple[2])
    if o1 == 0.0 or ox == 0.0 or o2 == 0.0:
        h_elo_val = elo_ratings.get(h_id, 1500.0)
        a_elo_val = elo_ratings.get(a_id, 1500.0)
        o1, ox, o2 = get_implied_odds(h_elo_val, a_elo_val)
        odds_source = 'implied_elo'
    else:
        odds_source = 'real'
    odds_t = torch.tensor([o1, ox, o2], dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # ELO Lookup & Normalization
    h_elo_val = elo_ratings.get(h_id, 1500.0)
    a_elo_val = elo_ratings.get(a_id, 1500.0)
    
    if max_elo == min_elo:
        h_elo_norm = 0.0
        a_elo_norm = 0.0
    else:
        h_elo_norm = (h_elo_val - min_elo) / (max_elo - min_elo)
        a_elo_norm = (a_elo_val - min_elo) / (max_elo - min_elo)
        
    h_elo_t = torch.tensor([h_elo_norm], dtype=torch.float32).to(DEVICE)
    a_elo_t = torch.tensor([a_elo_norm], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred = model(h_seq, a_seq, 
                     torch.tensor([h_id], device=DEVICE), 
                     torch.tensor([a_id], device=DEVICE), 
                     torch.tensor([l_id], device=DEVICE),
                     odds_t, h_elo_t, a_elo_t)
        
        lambdas = pred[0]
        rho = pred[1]
        h_lam, a_lam = lambdas[0, 0].item(), lambdas[0, 1].item()
        rho_val = rho[0].item() if rho.ndim > 0 else rho.item()

    stats = calculate_advanced_stats(h_lam, a_lam)
    
    # --- HYBRID ENSEMBLE SYSTEM ---
    # We use XGBoost and LightGBM alongside the Deep Learning model 
    # to form a meta-calibrated consensus probability for Win/Draw/Loss.
    global hybrid_model, lgbm_model
    if hybrid_model or lgbm_model:
        try:
            # 1. Extract Embeddings from the new Transformer
            feats = model.extract_features(h_seq, a_seq, 
                         torch.tensor([h_id], device=DEVICE), 
                         torch.tensor([a_id], device=DEVICE), 
                         torch.tensor([l_id], device=DEVICE),
                         h_elo_t, a_elo_t)
            
            # 2. Concat Odds (cpu numpy)
            import numpy as np
            emb_np = feats.cpu().numpy()
            odds_np = np.array([list(odds_tuple)], dtype=np.float32)

            # 3. Calculate Feature Deltas (Model vs Market)
            model_probs = pm.calculate_probabilities(h_lam, a_lam, rho_val)
            p_home = model_probs['home_win']
            p_draw = model_probs['draw']
            p_away = model_probs['away_win']

            o1, ox, o2 = odds_tuple
            imp_home = (1.0 / o1) if o1 > 0 else 0.0
            imp_draw = (1.0 / ox) if ox > 0 else 0.0
            imp_away = (1.0 / o2) if o2 > 0 else 0.0

            d_home = p_home - imp_home
            d_draw = p_draw - imp_draw
            d_away = p_away - imp_away
            
            # [DeltaH, DeltaD, DeltaA, ProbH, ProbD, ProbA]
            deltas_np = np.array([[d_home, d_draw, d_away, p_home, p_draw, p_away]], dtype=np.float32)
            
            # 4. Construct Final Input [Embeddings + Odds + Deltas]
            X_in = np.hstack([emb_np, odds_np, deltas_np])
            
            # 5. Ensemble Voting
            probs_list = []
            
            # Include base DL model as a voter (converted to percentages)
            probs_list.append([p_home * 100, p_draw * 100, p_away * 100])
            
            # Voter 2: XGBoost
            if hybrid_model:
                 xgb_probs = hybrid_model.predict_proba(X_in)[0]
                 if calibrator:
                     xgb_probs = calibrator.predict_proba([xgb_probs])[0]
                 probs_list.append((xgb_probs * 100).tolist())
                 
            # Voter 3: LightGBM
            if lgbm_model:
                 lgb_probs = lgbm_model.predict_proba(X_in)[0] * 100
                 probs_list.append(lgb_probs.tolist())
                 
            # 6. Meta-Calibration (Average of all available models)
            ensemble_probs = np.mean(probs_list, axis=0)
            
            # 7. Integrate into Stats
            # Ensembles only give W/D/L. We keep score/btts/o2.5 from Deep Learning.
            stats['win'] = round(float(ensemble_probs[0]), 1)
            stats['draw'] = round(float(ensemble_probs[1]), 1)
            stats['loss'] = round(float(ensemble_probs[2]), 1)
            
            stats['is_hybrid'] = True
            
        except Exception as e:
            # Fallback to pure DL if ensemble fails
            # print(f"Ensemble Inf Error: {e}")
            pass
            
    stats['odds_source'] = odds_source
    stats['odds_used'] = [round(o1, 2), round(ox, 2), round(o2, 2)]
    return stats



# --- ROUTE 9: TRAIN RL AGENT ---
@app.route('/train_rl', methods=['POST'])
def train_rl():
    global policy_agent
    try:
        if model_current is None:
            return jsonify({'status': 'error', 'message': 'Main model must be trained first.'})

        print("Starting RL Policy Training (PPO)...")
        new_agent = pm.PPOAgent(state_dim=RL_STATE_DIM, action_dim=4, lr=0.0003, entropy_coef=0.15).to(DEVICE)
        policy_agent = pm.train_ppo_agent(model_current, new_agent, epochs=20)

        torch.save(policy_agent.state_dict(), os.path.join(_MODELS_DIR, 'ppo_agent.pth'))

        return jsonify({'status': 'success', 'message': 'PPO Agent trained and saved!'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})


# --- ROUTE 10: GET BETTING STRATEGY ---


@app.route('/scrape_upcoming', methods=['POST'])
def scrape_upcoming_route():
    try:
        import scrape_upcoming
        scrape_upcoming.scrape_fixtures(days=30)
        return jsonify({'status': 'success', 'message': 'Successfully scraped upcoming fixtures for 30 days!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})




# --- ROUTE 12: STRATEGY BACKTEST SIMULATION ---
@app.route('/run_strategy_backtest', methods=['POST'])
def run_strategy_backtest():
    try:
        req = request.json
        start_date = pd.Timestamp(req.get('start_date', (datetime.now() - pd.Timedelta(days=90)).strftime('%Y-%m-%d')))
        end_date = pd.Timestamp(req.get('end_date', datetime.now().strftime('%Y-%m-%d')))
        initial_bankroll = float(req.get('bankroll', 40.0))
        
        # Split Bankroll: 4 Strategies -> $10.0 each
        strat_capital = initial_bankroll / 4.0
        
        equity = {
            'model': [strat_capital],
            'rl_sniper': [strat_capital],
            'consensus': [strat_capital],
            'rl_draw': [strat_capital],
            'dates': [start_date.strftime('%Y-%m-%d')]
        }
        
        history_log = []
        
        # Filter Data
        if master_df is None or master_df.empty: 
            return jsonify({'status': 'error', 'message': 'No data loaded.'})
            
        mask = (master_df['date_obj'] >= start_date) & (master_df['date_obj'] <= end_date)
        sim_df = master_df[mask].sort_values('date_obj')
        
        # Group by Date to simulate daily P/L updates (for cleaner graph)
        grouped = sim_df.groupby('date_obj')
        
        curr_eq = {
            'model': strat_capital,
            'rl_sniper': strat_capital,
            'consensus': strat_capital,
            'rl_draw': strat_capital
        }
        
        for date_val, group in grouped:
            date_str = date_val.strftime('%Y-%m-%d')
            
            for _, row in group.iterrows():
                try:
                    # Parse Match Info
                    h_name, a_name = row['home team'], row['away team']
                    if h_name not in le_team.classes_ or a_name not in le_team.classes_: continue
                    
                    h_id = le_team.transform([h_name])[0]
                    a_id = le_team.transform([a_name])[0]
                    l_id = le_league.transform([row['league_name']])[0]
                    
                    # Odds — use implied odds from ELO if missing or 1.0 default
                    o1 = float(row['odds_1']) if pd.notna(row['odds_1']) and float(row['odds_1']) > 1.0 else 0.0
                    ox = float(row['odds_x']) if pd.notna(row['odds_x']) and float(row['odds_x']) > 1.0 else 0.0
                    o2 = float(row['odds_2']) if pd.notna(row['odds_2']) and float(row['odds_2']) > 1.0 else 0.0
                    if o1 == 0.0 or ox == 0.0 or o2 == 0.0:
                        h_elo_v = elo_ratings.get(h_id, 1500.0)
                        a_elo_v = elo_ratings.get(a_id, 1500.0)
                        o1, ox, o2 = get_implied_odds(h_elo_v, a_elo_v)
                    odds_vals = (o1, ox, o2)
                    
                    # Result
                    hg, ag = row['home team total goal'], row['away team total goal']
                    res_idx = 0 # Default Home
                    if ag > hg: res_idx = 2
                    elif ag == hg: res_idx = 1
                    
                    # --- PREDICTIONS ---
                    
                    # 1. Model Prediction (Best Loss Model)
                    stats = predict_match_internal(model_current, h_id, a_id, l_id, row['date_obj'], master_df, odds_vals)
                    if not stats: continue
                    
                    # Determine Model Pick (Highest Prob from Stats)
                    # stats has 'win', 'draw', 'loss' in %
                    p_win, p_draw, p_loss = stats['win'], stats['draw'], stats['loss']
                    model_pick_idx = -1
                    if p_win > p_draw and p_win > p_loss: model_pick_idx = 0
                    elif p_draw > p_win and p_draw > p_loss: model_pick_idx = 1
                    elif p_loss > p_win and p_loss > p_draw: model_pick_idx = 2
                    
                    # 2. RL Agent Prediction
                    rl_pick_idx = 3 # Pass
                    rl_conf = 0.0
                    if policy_agent:
                        date_obj = row['date_obj']
                        h_seq = torch.from_numpy(pm.get_team_history(h_id, date_obj, master_df)).float().unsqueeze(0).to(DEVICE)
                        a_seq = torch.from_numpy(pm.get_team_history(a_id, date_obj, master_df)).float().unsqueeze(0).to(DEVICE)
                        h_id_t = torch.tensor([h_id], device=DEVICE)
                        a_id_t = torch.tensor([a_id], device=DEVICE)
                        l_id_t = torch.tensor([l_id], device=DEVICE)

                        with torch.no_grad():
                            # ELO
                            h_elo_val = elo_ratings.get(h_id, 1500.0)
                            a_elo_val = elo_ratings.get(a_id, 1500.0)
                            if max_elo == min_elo:
                                h_elo_norm = 0.0; a_elo_norm = 0.0
                            else:
                                h_elo_norm = (h_elo_val - min_elo) / (max_elo - min_elo)
                                a_elo_norm = (a_elo_val - min_elo) / (max_elo - min_elo)
                            h_elo_t = torch.tensor([h_elo_norm], dtype=torch.float32).to(DEVICE)
                            a_elo_t = torch.tensor([a_elo_norm], dtype=torch.float32).to(DEVICE)

                            state = model_current.extract_features(h_seq, a_seq, h_id_t, a_id_t, l_id_t, h_elo_t, a_elo_t)
                            probs = policy_agent.actor(state)
                            rl_pick_idx = torch.argmax(probs).item() # 0,1,2,3
                            rl_conf = probs[0, rl_pick_idx].item() * 100
                            
                    # --- EXECUTE STRATEGIES ---
                    stake = 1.0
                    match_lbl = f"{h_name} vs {a_name}"
                    
                    # Strategy 1: Pure Model (Bet on Model Pick)
                    if model_pick_idx != -1:
                        payout = -stake
                        if model_pick_idx == res_idx:
                            odd = [o1, ox, o2][model_pick_idx]
                            payout = (stake * odd) - stake
                        curr_eq['model'] += payout
                        # history_log.append({'date': date_str, 'match': match_lbl, 'strat': 'Model', 'res': payout})
                        
                    # Strategy 2: RL 100% (High Conf > 90%)
                    if rl_pick_idx != 3 and rl_conf > 90.0:
                         payout = -stake
                         if rl_pick_idx == res_idx:
                             odd = [o1, ox, o2][rl_pick_idx]
                             payout = (stake * odd) - stake
                         curr_eq['rl_sniper'] += payout
                         history_log.append({'date': date_str, 'match': match_lbl, 'strat': 'RL Sniper', 'pick': ['H','D','A'][rl_pick_idx], 'res': payout})
                         
                    # Strategy 3: Consensus (Model == RL)
                    if rl_pick_idx != 3 and rl_pick_idx == model_pick_idx:
                         payout = -stake
                         if rl_pick_idx == res_idx:
                             odd = [o1, ox, o2][rl_pick_idx]
                             payout = (stake * odd) - stake
                         curr_eq['consensus'] += payout
                         history_log.append({'date': date_str, 'match': match_lbl, 'strat': 'Consensus', 'pick': ['H','D','A'][rl_pick_idx], 'res': payout})

                    # Strategy 4: RL Draw (RL Pick == Draw)
                    if rl_pick_idx == 1:
                         payout = -stake
                         if 1 == res_idx:
                             odd = ox
                             payout = (stake * odd) - stake
                         curr_eq['rl_draw'] += payout
                         history_log.append({'date': date_str, 'match': match_lbl, 'strat': 'RL Draw', 'pick': 'Draw', 'res': payout})
                    
                except: continue
            
            # End of Day Update
            equity['dates'].append(date_str)
            equity['model'].append(round(curr_eq['model'], 2))
            equity['rl_sniper'].append(round(curr_eq['rl_sniper'], 2))
            equity['consensus'].append(round(curr_eq['consensus'], 2))
            equity['rl_draw'].append(round(curr_eq['rl_draw'], 2))
            
        return jsonify({
            'status': 'success',
            'equity': equity,
            'log': history_log[-50:] # Return last 50 bets for detailed table
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/strategy/weekly_report')
def get_weekly_report():
    # 1. Date Range: Jan 1st 2026 onwards
    start_date = pd.Timestamp("2026-01-01")
    
    # 2. Get Matches from History (Simulation) + Upcoming (if any)
    # For "Profit so far", we strictly look at historical results in master_df first.
    # We can also start from the csv, but master_df is the source of truth for "Past".
    
    # Let's filter master_df for 2026+
    mask = master_df['date_obj'] >= start_date
    df = master_df[mask].copy()
    
    if df.empty:
        # Fallback for demo if no 2026 data: use 2024/2025
        # mask = master_df['date_obj'] >= pd.Timestamp("2023-08-01")
        # df = master_df[mask].copy()
        pass

    # 3. Simulate Predictions for each match
    # This is expensive. We should really rely on logged predictions, but we'll re-infer for the "Lab".
    
    report_data = {} # Key: "Week X" -> { "Interval 1": [matches], ... }
    
    # Group by Week Number
    # ISO Calendar: Mon=1, Sun=7.
    # Interval 1: Tue(2)-Wed(3)
    # Interval 2: Thu(4)-Fri(5)
    # Interval 3: Sat(6)-Sun(7)
    # Mon(1) Excluded.
    
    # Sort by date
    df = df.sort_values('date_obj')
    
    for _, row in df.iterrows():
        dt = row['date_obj']
        week_num = dt.isocalendar()[1]
        day_num = dt.isocalendar()[2] # 1=Mon...7=Sun
        year = dt.year
        
        week_key = f"{year}-W{week_num}"
        
        interval = None
        limit = 3
        if day_num in [2, 3]: interval = "Interval 1 (Tue-Wed)"; limit=3
        elif day_num in [4, 5]: interval = "Interval 2 (Thu-Fri)"; limit=3
        elif day_num in [6, 7]: interval = "Interval 3 (Sat-Sun)"; limit=4
        else: continue # Skip Monday
        
        # PREDICTION LOGIC (Strict)
        # We need Model Prob > 72% AND RL > 90%
        # Let's re-run inference or use cached values if we had them.
        # For speed in this verified environment, I will use the 'odds' to infer difficulty 
        # but realistically we must run the model.
        
        # Re-construct Tensors (Single Batch)
        h_id = row['home_id']
        a_id = row['away_id']
        l_id = row['league_id']
        
        # Helper: Get cached prediction if possible or run minimal
        # (Running full DL model 1000 times will timeout. We will use a proxy of stored stats or simple heuristic for this demo
        # UNLESS we are sure we can run it fast. Let's try to run it but only for matches with "decent" odds to save time?)
        
        # Optimization: Only check if odds imply it's even plausible?
        # No, user wants model truth.
        
        # Fast Inference Context
        h_seq = torch.from_numpy(pm.get_team_history(h_id, dt, master_df)).float().unsqueeze(0).to(DEVICE)
        a_seq = torch.from_numpy(pm.get_team_history(a_id, dt, master_df)).float().unsqueeze(0).to(DEVICE)
        h_id_t = torch.tensor([h_id], device=DEVICE)
        a_id_t = torch.tensor([a_id], device=DEVICE)
        l_id_t = torch.tensor([l_id], device=DEVICE)
        
        o1 = float(row.get('odds_1', 0.0))
        ox = float(row.get('odds_x', 0.0))
        o2 = float(row.get('odds_2', 0.0))
        if o1 == 0.0 or ox == 0.0 or o2 == 0.0:
            h_elo_val = elo_ratings.get(h_id, 1500.0)
            a_elo_val = elo_ratings.get(a_id, 1500.0)
            o1, ox, o2 = get_implied_odds(h_elo_val, a_elo_val)
        odds_t = torch.tensor([[o1, ox, o2]], device=DEVICE)

        # ELO
        h_elo_val = elo_ratings.get(h_id, 1500.0)
        a_elo_val = elo_ratings.get(a_id, 1500.0)
        if max_elo == min_elo:
            h_elo_norm = 0.0; a_elo_norm = 0.0
        else:
            h_elo_norm = (h_elo_val - min_elo) / (max_elo - min_elo)
            a_elo_norm = (a_elo_val - min_elo) / (max_elo - min_elo)
        h_elo_t = torch.tensor([h_elo_norm], dtype=torch.float32).to(DEVICE)
        a_elo_t = torch.tensor([a_elo_norm], dtype=torch.float32).to(DEVICE)
        
        if model_final is None or policy_agent is None:
            continue

        try:
            # DL Model
            with torch.no_grad():
                preds = model_final(h_seq, a_seq, h_id_t, a_id_t, l_id_t, odds_t, h_elo_t, a_elo_t)
                lambdas = preds[0]
                h_lam, a_lam = lambdas[0, 0].item(), lambdas[0, 1].item()
                # Poisson Prob
                # Simple Win Prob
                grid = pm.generate_score_grid(h_lam, a_lam)
                h_prob = np.sum(np.tril(grid, -1)) * 100
                a_prob = np.sum(np.triu(grid, 1)) * 100
                
                # RL Agent
                state = model_final.extract_features(h_seq, a_seq, h_id_t, a_id_t, l_id_t, h_elo_t, a_elo_t)
                rl_probs = policy_agent.actor(state)
                rl_act = torch.argmax(rl_probs).item()
                rl_conf = rl_probs[0, rl_act].item() * 100
                
            # THRESHOLD CHECK
            # Model > 60% (Lowered from 72)
            # RL > 75% (Lowered from 90) and Matches Model
            
            pick = None
            pick_prob = 0
            is_hit = False
            odds_val = 0.0
            
            # Map Result (1=H, 0=D, 2=A)
            # Master DF usually has 1, 0, 2 for Home, Draw, Away OR strings 'H', 'D', 'A'
            res_input = row.get('result', None)
            actual_res = 'UNKNOWN'
            
            # Normalize to 'H', 'D', 'A'
            if str(res_input).strip().upper() in ['H', 'HOME', '1', '1.0']: actual_res = 'H'
            elif str(res_input).strip().upper() in ['D', 'DRAW', '0', '0.0', 'X']: actual_res = 'D'
            elif str(res_input).strip().upper() in ['A', 'AWAY', '2', '2.0']: actual_res = 'A'
            
            # Check Home
            if h_prob > 65 and rl_act == 0 and rl_conf > 75:
                pick = "HOME"
                pick_prob = h_prob
                is_hit = (actual_res == 'H')
                odds_val = float(row.get('odds_1', 0.0))
            # Check Away
            elif a_prob > 65 and rl_act == 2 and rl_conf > 75:
                pick = "AWAY"
                pick_prob = a_prob
                is_hit = (actual_res == 'A')
                odds_val = float(row.get('odds_2', 0.0))
                
            if pick:
                # Add to candidate list
                if week_key not in report_data: report_data[week_key] = {}
                if interval not in report_data[week_key]: 
                    report_data[week_key][interval] = {'matches': [], 'limit': limit}
                
                report_data[week_key][interval]['matches'].append({
                    'match': f"{row['home team']} vs {row['away team']}",
                    'pick': pick,
                    'prob': pick_prob,
                    'odds': odds_val,
                    'result': 'HIT' if is_hit else 'MISS'
                })
                
        except Exception:
            continue

    # 4. Post-Process: Sort by Confidence and Cut to Pool Size
    formatted_report = []
    
    # Sort weeks
    sorted_weeks = sorted(report_data.keys())
    
    for week in sorted_weeks:
        intervals_data = report_data[week]
        # Sort intervals (1, 2, 3)
        sorted_intervals = sorted(intervals_data.keys())
        
        display_week = {
            'week': week,
            'intervals': []
        }
        
        for intv_name in sorted_intervals:
            data = intervals_data[intv_name]
            # Sort by Probability Desc
            all_picks = sorted(data['matches'], key=lambda x: x['prob'], reverse=True)
            
            # Cut to Limit
            pool = all_picks[:data['limit']]
            
            if not pool: continue
            
            # Calculate Pool Stats
            total_odds = 1.0
            hits = 0
            for p in pool:
                if p['result'] == 'HIT':
                    total_odds *= p['odds'] # Accumulator math
                    hits += 1
                else:
                    total_odds = 0 # Acca busts
            
            display_week['intervals'].append({
                'name': intv_name,
                'pool': pool,
                'pool_status': 'WIN' if total_odds > 0 else 'LOSS',
                'total_odds': round(total_odds, 2) if total_odds > 0 else 0
            })
            
        formatted_report.append(display_week)

    return jsonify({'status': 'success', 'report': formatted_report})

@app.route('/gnn-graph/<league>')
def get_gnn_graph(league):
    # Filter for League
    league_matches = master_df[master_df['league_name'] == league].copy()
    
    # --- FILTER FOR CURRENT SEASON ONLY (User Request) ---
    # Keeps graph relevant to "this season" logic
    today = datetime.now()
    if today.month >= 7: # July/Aug Start
        season_start = pd.Timestamp(year=today.year, month=7, day=15)
    else:
        season_start = pd.Timestamp(year=today.year - 1, month=7, day=15)
        
    league_matches = league_matches[league_matches['date_obj'] >= season_start]
    # -----------------------------------------------------
    
    # Calculate Power Index for Node Size
    team_stats = {}
    teams = pd.concat([league_matches['home team'], league_matches['away team']]).unique()
    
    for team in teams:
        team_stats[team] = {'pts': 0, 'form': 0}
        
    # Quick Points Calc & Edge Building
    edge_counts = {}  # (winner, loser) -> count
    
    for _, row in league_matches.iterrows():
        h, a = row['home team'], row['away team']
        
        # Robustly calculate result from goals
        try:
            hg = float(row['home team total goal'])
            ag = float(row['away team total goal'])
        except:
            continue # Skip invalid rows
            
        res = 'D'
        if hg > ag: res = 'H'
        elif ag > hg: res = 'A'
        
        # Points
        if res == 'H':
            if h in team_stats: team_stats[h]['pts'] += 3
        elif res == 'A':
            if a in team_stats: team_stats[a]['pts'] += 3
        else:
            if h in team_stats: team_stats[h]['pts'] += 1
            if a in team_stats: team_stats[a]['pts'] += 1
            
        # Edges
        if res == 'H':
            key = (h, a)
            edge_counts[key] = edge_counts.get(key, 0) + 1
        elif res == 'A':
            key = (a, h)
            edge_counts[key] = edge_counts.get(key, 0) + 1
            
    # Calculate Point Range for Normalization
    all_pts = [stats['pts'] for stats in team_stats.values()]
    if all_pts:
        min_pts = min(all_pts)
        max_pts = max(all_pts)
    else:
        min_pts, max_pts = 0, 0
        
    # Nodes
    nodes = []
    for team, stats in team_stats.items():
        # Normalize size to range 15-60 for better visual distinction
        if max_pts > min_pts:
            norm_size = 15 + ((stats['pts'] - min_pts) / (max_pts - min_pts)) * 45
        else:
            norm_size = 30 # Default if equal
            
        # Color by "Tier" (Relative Percentiles)
        # Top 20% = Gold, Bottom 20% = Red, Rest = Blue
        rank_pct = 0.5
        if max_pts > min_pts:
            rank_pct = (stats['pts'] - min_pts) / (max_pts - min_pts)
        
        color = '#97c2fc' # Default Blue
        if rank_pct >= 0.8: color = '#ffd700' # Top Tier (Gold)
        elif rank_pct <= 0.2: color = '#ffcccc' # Relegation (Red)
        
        nodes.append({
            'id': team,
            'label': team,
            'value': norm_size, # Vis.js scaling
            'size': norm_size,  # Explicit size
            'color': color,
            'title': f"Points: {stats['pts']}" # Tooltip
        })
        
    print(f"[GNN DEBUG] League: {league}, Nodes: {len(nodes)}, MinPts: {min_pts}, MaxPts: {max_pts}")
    
    edges = []
    for (winner, loser), count in edge_counts.items():
        edges.append({
            'from': winner, 
            'to': loser, 
            'arrows': 'to',           # EXPLICITLY RE-ADDED ARROWS
            'width': min(count * 2, 8),
            'color': '#2d6a4f',       # Dark green
            'title': f'{winner} beat {loser} {count} time(s)'
        })
            
    return jsonify({'nodes': nodes, 'edges': edges})


# --- OPTIMAL PERCENTAGE CALCULATION (STREAMING) ---
@app.route('/get_calibration', methods=['GET'])
def get_calibration():
    def generate():
        yield json.dumps({'progress': 0, 'log': 'Starting Model Calibration...', 'status': 'running'}) + '\n'
        
        # Select 400 most recent matches globally
        test_df = master_df.sort_values('date_obj').tail(400)

        if len(test_df) < 10:
            yield json.dumps({'progress': 100, 'log': 'Error: Not enough data.', 'status': 'error'}) + '\n'
            return
        total_matches = len(test_df)
        yield json.dumps({'progress': 5, 'log': f'Found {total_matches} matches for analysis.', 'status': 'running'}) + '\n'

        results = []
        model_instance = model_current
        
        count = 0
        errors = 0
        import sys
        with torch.no_grad():
            for idx, row in test_df.iterrows():
                count += 1
                try:
                    # print(f"DEBUG: Processing match {count}...", file=sys.stdout)
                    h_id, a_id = row['home_id'], row['away_id']
                    l_id = row['league_id']
                    date = row['date_obj']
                    h_name, a_name = row['home team'], row['away team']
                    
                    # Actual Result
                    hg, ag = row['home team total goal'], row['away team total goal']
                    actual = 'Draw'
                    if hg > ag: actual = 'Home'
                    elif ag > hg: actual = 'Away'

                    h_seq = torch.from_numpy(pm.get_team_history(h_id, date, master_df)).float().unsqueeze(0).to(DEVICE)
                    a_seq = torch.from_numpy(pm.get_team_history(a_id, date, master_df)).float().unsqueeze(0).to(DEVICE)
                    o1 = float(row.get('odds_1', 0.0))
                    ox = float(row.get('odds_x', 0.0))
                    o2 = float(row.get('odds_2', 0.0))
                    if o1 == 0.0 or ox == 0.0 or o2 == 0.0:
                        h_elo_val = elo_ratings.get(h_id, 1500.0)
                        a_elo_val = elo_ratings.get(a_id, 1500.0)
                        o1, ox, o2 = get_implied_odds(h_elo_val, a_elo_val)
                    odds_t = torch.tensor([o1, ox, o2], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    
                    # ELO
                    h_elo_val = elo_ratings.get(h_id, 1500.0)
                    a_elo_val = elo_ratings.get(a_id, 1500.0)
                    if max_elo == min_elo:
                        h_elo_norm = 0.0; a_elo_norm = 0.0
                    else:
                        h_elo_norm = (h_elo_val - min_elo) / (max_elo - min_elo)
                        a_elo_norm = (a_elo_val - min_elo) / (max_elo - min_elo)
                    h_elo_t = torch.tensor([h_elo_norm], dtype=torch.float32).to(DEVICE)
                    a_elo_t = torch.tensor([a_elo_norm], dtype=torch.float32).to(DEVICE)

                    pred = model_instance(h_seq, a_seq, torch.tensor([h_id], device=DEVICE),
                                     torch.tensor([a_id], device=DEVICE), torch.tensor([l_id], device=DEVICE), 
                                     odds_t, h_elo_t, a_elo_t)
                    lambdas = pred[0]
                    h_lam, a_lam = lambdas[0, 0].item(), lambdas[0, 1].item()
                    
                    grid = pm.generate_score_grid(h_lam, a_lam)
                    h_prob = np.sum(np.tril(grid, -1))
                    d_prob = np.sum(np.diag(grid))
                    a_prob = np.sum(np.triu(grid, 1))
                    
                    probs = {'Home': h_prob, 'Draw': d_prob, 'Away': a_prob}
                    pick = max(probs, key=probs.get)
                    conf = probs[pick] * 100
                    is_hit = (pick == actual)
                    
                    results.append((conf, is_hit))
                    
                    # Log Update
                    # if count % 5 == 0: 
                    # Always log for debug
                    prog = 5 + int((count / total_matches) * 80) # Scale to 85%
                    res_str = "SUCCESS" if is_hit else "FAIL"
                    log_msg = f"{date.strftime('%b %d %Y')} {h_name} vs {a_name}: Model predicts {pick} ({conf:.0f}%) -> {res_str}"
                    yield json.dumps({'progress': prog, 'log': log_msg, 'status': 'running'}) + '\n'

                except Exception as e:
                    errors += 1
                    print(f"Error in CALIB loop: {e}", file=sys.stdout)
                    sys.stdout.flush()
                    if errors < 3: 
                         yield json.dumps({'log': f"[ERR] Match {count} failed: {str(e)}", 'status': 'running'}) + '\n'
                    pass

        # Optimization Phase
        yield json.dumps({'progress': 90, 'log': 'Optimizing Thresholds...', 'status': 'running'}) + '\n'
        
        best_thresh = 0
        best_acc = 0.0
        
        for t in range(50, 95):
            subset = [x for x in results if x[0] >= t]
            if len(subset) < 20: continue 
            hits = sum(1 for x in subset if x[1])
            acc = (hits / len(subset)) * 100
            if acc >= best_acc:
                best_acc = acc
                best_thresh = t

        global optimal_pred_threshold
        optimal_pred_threshold = best_thresh

        details = f"Accuracy {best_acc:.1f}% on >{best_thresh}% confidence ({len(results)} matches)"
        yield json.dumps({
            'progress': 100, 
            'log': f'Optimization Complete. Optimal Threshold: {best_thresh}%', 
            'status': 'complete',
            'result': {'optimal_threshold': best_thresh, 'details': details}
        }) + '\n'

    return Response(stream_with_context(generate()), mimetype='application/json')

@app.route('/get_rl_optimal', methods=['GET'])
def get_rl_optimal():
    def generate():
        yield json.dumps({'progress': 0, 'log': 'Starting RL Agent Optimization...', 'status': 'running'}) + '\n'
        
        if policy_agent is None:
             yield json.dumps({'progress': 100, 'log': 'Error: RL Agent not loaded.', 'status': 'error'}) + '\n'
             return

        # Select 400 most recent matches globally
        test_df = master_df.sort_values('date_obj').tail(400)

        if len(test_df) < 10:
            yield json.dumps({'progress': 100, 'log': 'Error: Not enough data.', 'status': 'error'}) + '\n'
            return
        total_matches = len(test_df)
        yield json.dumps({'progress': 5, 'log': f'Analyzing {total_matches} matches with Policy Network...', 'status': 'running'}) + '\n'
        
        results = []
        model_instance = model_current
        passes = 0
        errors = 0
        count = 0
        
        with torch.no_grad():
            for idx, row in test_df.iterrows():
                count += 1
                try:
                    h_id, a_id = row['home_id'], row['away_id']
                    l_id = row['league_id']
                    date = row['date_obj']
                    h_name, a_name = row['home team'], row['away team']
                    
                    h_seq = torch.from_numpy(pm.get_team_history(h_id, date, master_df)).float().unsqueeze(0).to(DEVICE)
                    a_seq = torch.from_numpy(pm.get_team_history(a_id, date, master_df)).float().unsqueeze(0).to(DEVICE)
                    o1 = float(row.get('odds_1', 0.0)); ox = float(row.get('odds_x', 0.0)); o2 = float(row.get('odds_2', 0.0))
                    if o1 == 0.0 or ox == 0.0 or o2 == 0.0:
                        h_elo_val = elo_ratings.get(h_id, 1500.0)
                        a_elo_val = elo_ratings.get(a_id, 1500.0)
                        o1, ox, o2 = get_implied_odds(h_elo_val, a_elo_val)
                    odds_t = torch.tensor([o1, ox, o2], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    
                    # ELO
                    h_elo_val = elo_ratings.get(h_id, 1500.0)
                    a_elo_val = elo_ratings.get(a_id, 1500.0)
                    if max_elo == min_elo:
                        h_elo_norm = 0.0; a_elo_norm = 0.0
                    else:
                        h_elo_norm = (h_elo_val - min_elo) / (max_elo - min_elo)
                        a_elo_norm = (a_elo_val - min_elo) / (max_elo - min_elo)
                    h_elo_t = torch.tensor([h_elo_norm], dtype=torch.float32).to(DEVICE)
                    a_elo_t = torch.tensor([a_elo_norm], dtype=torch.float32).to(DEVICE)

                    state = model_instance.extract_features(h_seq, a_seq, torch.tensor([h_id], device=DEVICE),
                                             torch.tensor([a_id], device=DEVICE), torch.tensor([l_id], device=DEVICE), 
                                             h_elo_t, a_elo_t)
                    
                    probs = policy_agent.actor(state)
                    action = torch.argmax(probs).item()
                    conf = probs[0, action].item() * 100
                    
                    hg, ag = row['home team total goal'], row['away team total goal']
                    actual = 3 # Pass
                    if hg > ag: actual = 0 # Home
                    elif ag > hg: actual = 2 # Away
                    elif hg == ag: actual = 1 # Draw
                    
                    actions = ['HOME', 'DRAW', 'AWAY', 'PASS']
                    act_str = actions[action]
                    
                    if action == 3: 
                        passes += 1
                        if count % 10 == 0:
                             prog = 5 + int((count / total_matches) * 85)
                             yield json.dumps({'progress': prog, 'log': f"{h_name} vs {a_name}: Agent PASSED (Conf {conf:.1f}%)", 'status': 'running'}) + '\n'
                        continue 
                        
                    is_hit = (action == actual)
                    results.append((conf, is_hit))
                    
                    res_str = "SUCCESS" if is_hit else "FAIL"
                    if count % 5 == 0:
                        prog = 5 + int((count / total_matches) * 85)
                        log_msg = f"{date.strftime('%b %d %Y')} {h_name} vs {a_name}: Agent picks {act_str} ({conf:.0f}%) -> {res_str}"
                        yield json.dumps({'progress': prog, 'log': log_msg, 'status': 'running'}) + '\n'
                        
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                         yield json.dumps({'log': f"[ERR] RL Match failed: {str(e)}", 'status': 'running'}) + '\n'
                    pass

        # Find optimal
        best_acc = 0
        best_thresh = 0
        for t in range(50, 95):
            subset = [x for x in results if x[0] >= t]
            if len(subset) < 10: continue
            hits = sum(1 for x in subset if x[1])
            acc = (hits / len(subset)) * 100
            
            if acc >= best_acc:
                best_acc = acc
                best_thresh = t

        global optimal_rl_threshold
        optimal_rl_threshold = best_thresh
        
        yield json.dumps({
            'progress': 100, 
            'log': f'Optimal RL Threshold: {best_thresh}% (Accuracy: {best_acc:.1f}%)', 
            'status': 'complete',
            'result': {'optimal_threshold': best_thresh, 'details': f'Accuracy {best_acc:.1f}% at >{best_thresh}% confidence'}
        }) + '\n'

    return Response(stream_with_context(generate()), mimetype='application/json')


# --- HELPERS FOR DYNAMIC STRATEGY ---
def calc_optimal_pred_threshold():
    """
    Calculates the 'Optimal Prediction Percentage' based on last 400 matches.
    Returns: threshold (float)
    """
    test_df = master_df.sort_values('date_obj').tail(400)
    if len(test_df) < 10: return 60.0 # Default
    
    results = []
    with torch.no_grad():
        for idx, row in test_df.iterrows():
            try:
                h_id, a_id = row['home_id'], row['away_id']
                l_id = row['league_id']
                date = row['date_obj']
                # Actual
                hg, ag = row['home team total goal'], row['away team total goal']
                actual = 'Draw'
                if hg > ag: actual = 'Home'
                elif ag > hg: actual = 'Away'
                
                h_seq = torch.from_numpy(pm.get_team_history(h_id, date, master_df)).float().unsqueeze(0).to(DEVICE)
                a_seq = torch.from_numpy(pm.get_team_history(a_id, date, master_df)).float().unsqueeze(0).to(DEVICE)
                o1 = float(row.get('odds_1', 0.0)); ox = float(row.get('odds_x', 0.0)); o2 = float(row.get('odds_2', 0.0))
                if o1 == 0.0 or ox == 0.0 or o2 == 0.0:
                    h_elo_val = elo_ratings.get(h_id, 1500.0)
                    a_elo_val = elo_ratings.get(a_id, 1500.0)
                    o1, ox, o2 = get_implied_odds(h_elo_val, a_elo_val)
                odds_t = torch.tensor([o1, ox, o2], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                # ELO
                h_elo_val = elo_ratings.get(h_id, 1500.0)
                a_elo_val = elo_ratings.get(a_id, 1500.0)
                if max_elo == min_elo:
                    h_elo_norm = 0.0; a_elo_norm = 0.0
                else:
                    h_elo_norm = (h_elo_val - min_elo) / (max_elo - min_elo)
                    a_elo_norm = (a_elo_val - min_elo) / (max_elo - min_elo)
                h_elo_t = torch.tensor([h_elo_norm], dtype=torch.float32).to(DEVICE)
                a_elo_t = torch.tensor([a_elo_norm], dtype=torch.float32).to(DEVICE)

                pred = model_current(h_seq, a_seq, torch.tensor([h_id], device=DEVICE),
                                     torch.tensor([a_id], device=DEVICE), torch.tensor([l_id], device=DEVICE), 
                                     odds_t, h_elo_t, a_elo_t)
                h_lam, a_lam = pred[0][0,0].item(), pred[0][0,1].item()
                grid = pm.generate_score_grid(h_lam, a_lam)
                
                probs = {'Home': np.sum(np.tril(grid, -1)), 'Draw': np.sum(np.diag(grid)), 'Away': np.sum(np.triu(grid, 1))}
                pick = max(probs, key=probs.get)
                conf = probs[pick] * 100
                is_hit = (pick == actual)
                results.append((conf, is_hit))
            except: pass
            
    best_acc = 0
    best_thresh = 60.0
    for t in range(50, 95):
        subset = [x for x in results if x[0] >= t]
        if len(subset) < 15: continue
        hits = sum(1 for x in subset if x[1])
        acc = hits / len(subset)
        if acc >= best_acc:
            best_acc = acc
            best_thresh = t
    return float(best_thresh)

def calc_optimal_rl_threshold():
    """
    Calculates the 'Optimal RL Confidence' based on last 400 matches.
    Returns: threshold (float)
    """
    if policy_agent is None: return 60.0
    test_df = master_df.sort_values('date_obj').tail(400)
    if len(test_df) < 10: return 60.0
    
    results = []
    with torch.no_grad():
        for idx, row in test_df.iterrows():
            try:
                h_id, a_id = row['home_id'], row['away_id']
                l_id = row['league_id']
                date = row['date_obj']
                hg, ag = row['home team total goal'], row['away team total goal']
                actual = 3 # Pass
                if hg > ag: actual = 0
                elif ag > hg: actual = 2
                elif hg == ag: actual = 1
                
                h_seq = torch.from_numpy(pm.get_team_history(h_id, date, master_df)).float().unsqueeze(0).to(DEVICE)
                a_seq = torch.from_numpy(pm.get_team_history(a_id, date, master_df)).float().unsqueeze(0).to(DEVICE)
                o1 = float(row.get('odds_1', 0.0)); ox = float(row.get('odds_x', 0.0)); o2 = float(row.get('odds_2', 0.0))
                if o1 == 0.0 or ox == 0.0 or o2 == 0.0:
                    h_elo_val = elo_ratings.get(h_id, 1500.0)
                    a_elo_val = elo_ratings.get(a_id, 1500.0)
                    o1, ox, o2 = get_implied_odds(h_elo_val, a_elo_val)
                # ELO
                h_elo_val = elo_ratings.get(h_id, 1500.0)
                a_elo_val = elo_ratings.get(a_id, 1500.0)
                if max_elo == min_elo:
                    h_elo_norm = 0.0; a_elo_norm = 0.0
                else:
                    h_elo_norm = (h_elo_val - min_elo) / (max_elo - min_elo)
                    a_elo_norm = (a_elo_val - min_elo) / (max_elo - min_elo)
                h_elo_t = torch.tensor([h_elo_norm], dtype=torch.float32).to(DEVICE)
                a_elo_t = torch.tensor([a_elo_norm], dtype=torch.float32).to(DEVICE)

                state = model_current.extract_features(h_seq, a_seq, torch.tensor([h_id], device=DEVICE),
                                         torch.tensor([a_id], device=DEVICE), torch.tensor([l_id], device=DEVICE), 
                                         h_elo_t, a_elo_t)
                probs = policy_agent.actor(state)
                action = torch.argmax(probs).item()
                conf = probs[0, action].item() * 100
                
                if action != 3: # Not Pass
                    is_hit = (action == actual)
                    results.append((conf, is_hit))
            except: pass
            
    best_acc = 0
    best_thresh = 60.0
    for t in range(50, 95):
        subset = [x for x in results if x[0] >= t]
        if len(subset) < 15: continue
        hits = sum(1 for x in subset if x[1])
        acc = hits / len(subset)
        if acc >= best_acc:
            best_acc = acc
            best_thresh = t
    return float(best_thresh)

@app.route('/strategy/dynamic_report', methods=['POST'])
def get_strategy_dynamic():
    try:
        data = request.json
        start_date = pd.Timestamp(data['start_date'])
        end_date = pd.Timestamp(data['end_date'])
        
        # 1. Calculate Dynamic Thresholds (Recommendations)
        opt_pred_thresh = calc_optimal_pred_threshold()
        opt_rl_thresh = calc_optimal_rl_threshold()
        
        # 2. Filter Matches
        mask = (master_df['date_obj'] >= start_date) & (master_df['date_obj'] <= end_date)
        df = master_df[mask].copy()
        
        candidates = []
        
        if df.empty:
             # FALLBACK: Try loading UPCOMING_MATCHES.csv manually
             try:
                 upcoming_csv = 'UPCOMING_MATCHES.csv'
                 if os.path.exists(upcoming_csv):
                     cdf = pd.read_csv(upcoming_csv)
                     cdf['date_obj'] = pd.to_datetime(cdf['Date'])
                     
                     # Map Columns to match master_df
                     cdf['home team'] = cdf['Home']
                     cdf['away team'] = cdf['Away']
                     cdf['odds_1'] = cdf['Odds_1']
                     cdf['odds_x'] = cdf['Odds_X']
                     cdf['odds_2'] = cdf['Odds_2']
                     
                     # Create IDs (Transform)
                     # We need to filter by date first to avoid processing everything
                     mask_u = (cdf['date_obj'] >= start_date) & (cdf['date_obj'] <= end_date)
                     df_u = cdf[mask_u].copy()
                     
                     if not df_u.empty:
                         # Use Copy to avoid SettingWithCopy warning/issues
                         df = df_u.copy()
                         
                         # Safe Transform Helper
                         def safe_team_id(name):
                             if not le_team: return 0
                             try: return le_team.transform([str(name)])[0]
                             except: return 0
                             
                         # Add missing columns
                         if 'league_id' not in df.columns: df['league_id'] = 0 # Default/Unknown
                         if 'home_id' not in df.columns: 
                             df['home_id'] = df['home team'].apply(safe_team_id)
                         if 'away_id' not in df.columns: 
                             df['away_id'] = df['away team'].apply(safe_team_id)
                         
                         # League Map Attempt
                         def get_league_id(row):
                             lname = row.get('League', 'Unknown')
                             short_name = 'Unknown'
                             for k, v in LEAGUE_MAPPING.items():
                                 if k in lname: short_name = v; break
                             if le_league:
                                 try: return le_league.transform([short_name])[0]
                                 except: return 0
                             return 0
                             
                         df['league_id'] = df.apply(get_league_id, axis=1)
                         
             except Exception as e:
                 print(f"Fallback CSV Error: {e}")
        
        candidates = []
        
        if df.empty:
             return jsonify({
                'status': 'success', 
                'thresholds': {'pred': opt_pred_thresh, 'rl': opt_rl_thresh},
                'matches': [],
                'message': 'No matches found in database or upcoming file.'
            })
            
        last_error = None
        error_count = 0
            
        for idx, row in df.iterrows():
            try:
                if model_current is None or policy_agent is None:
                    continue

                date = row['date_obj']
                h_id, a_id = row['home_id'], row['away_id']
                l_id = row['league_id']

                # Model Inference
                h_seq = torch.from_numpy(pm.get_team_history(h_id, date, master_df)).float().unsqueeze(0).to(DEVICE)
                a_seq = torch.from_numpy(pm.get_team_history(a_id, date, master_df)).float().unsqueeze(0).to(DEVICE)
                o1 = float(row.get('odds_1', 0.0)); ox = float(row.get('odds_x', 0.0)); o2 = float(row.get('odds_2', 0.0))
                if o1 == 0.0 or ox == 0.0 or o2 == 0.0:
                    h_elo_val = elo_ratings.get(h_id, 1500.0)
                    a_elo_val = elo_ratings.get(a_id, 1500.0)
                    o1, ox, o2 = get_implied_odds(h_elo_val, a_elo_val)
                    strat_odds_source = 'implied_elo'
                else:
                    strat_odds_source = 'real'
                odds_t = torch.tensor([o1, ox, o2], dtype=torch.float32).unsqueeze(0).to(DEVICE)
                
                # ELO
                h_elo_val = elo_ratings.get(h_id, 1500.0)
                a_elo_val = elo_ratings.get(a_id, 1500.0)
                if max_elo == min_elo:
                    h_elo_norm = 0.0; a_elo_norm = 0.0
                else:
                    h_elo_norm = (h_elo_val - min_elo) / (max_elo - min_elo)
                    a_elo_norm = (a_elo_val - min_elo) / (max_elo - min_elo)
                h_elo_t = torch.tensor([h_elo_norm], dtype=torch.float32).to(DEVICE)
                a_elo_t = torch.tensor([a_elo_norm], dtype=torch.float32).to(DEVICE)

                # Prediction
                with torch.no_grad():
                    pred = model_current(h_seq, a_seq, torch.tensor([h_id], device=DEVICE), 
                                         torch.tensor([a_id], device=DEVICE), torch.tensor([l_id], device=DEVICE), 
                                         odds_t, h_elo_t, a_elo_t)
                    h_lam, a_lam = pred[0][0,0].item(), pred[0][0,1].item()
                    grid = pm.generate_score_grid(h_lam, a_lam)
                    h_prob = np.sum(np.tril(grid, -1)) * 100
                    d_prob = np.sum(np.diag(grid)) * 100
                    a_prob = np.sum(np.triu(grid, 1)) * 100
                    
                    # RL
                    state = model_current.extract_features(h_seq, a_seq, torch.tensor([h_id], device=DEVICE), 
                                            torch.tensor([a_id], device=DEVICE), torch.tensor([l_id], device=DEVICE), 
                                            h_elo_t, a_elo_t)
                    rl_probs = policy_agent.actor(state)
                    rl_act = torch.argmax(rl_probs).item()
                    rl_conf = rl_probs[0, rl_act].item() * 100
                
                probs = {'HOME': h_prob, 'DRAW': d_prob, 'AWAY': a_prob}
                model_pick = max(probs, key=probs.get)
                model_conf = probs[model_pick]
                
                rl_pick_str = ['HOME', 'DRAW', 'AWAY', 'PASS'][rl_act]
                
                match_lbl = f"{row['home team']} vs {row['away team']}"
                
                # Return ALL Matches > 0% (Client Side Filtering)
                if model_conf >= 0:
                     candidates.append({
                        'match': match_lbl,
                        'date': date.strftime('%Y-%m-%d'),
                        'time': row.get('Time', ''),
                        'league': row.get('league_name', row.get('League', 'Unknown')),
                        'home': row['home team'],
                        'away': row['away team'],
                        'pred_pick': model_pick,
                        'pred_conf': round(model_conf, 1),
                        'rl_pick': rl_pick_str,
                        'rl_conf': round(rl_conf, 1),
                        'odds_1': row.get('odds_1', 0.0),
                        'odds_x': row.get('odds_x', 0.0),
                        'odds_2': row.get('odds_2', 0.0),
                        'odds_source': strat_odds_source,
                        'odds_used': [round(o1, 2), round(ox, 2), round(o2, 2)]
                     })

            except Exception as e:
                # Store first error for debugging
                if last_error is None: last_error = str(e)
                error_count += 1
                pass
            
        debug_msg = f"Found {len(df)} raw matches. Candidates: {len(candidates)}. Source: {'DB' if not df.empty else 'CSV'}"
        if error_count > 0:
            debug_msg += f" | Errors: {error_count}. First Error: {last_error}"

        return jsonify({
            'status': 'success',
            'thresholds': {'pred': opt_pred_thresh, 'rl': opt_rl_thresh},
            'matches': candidates,
            'debug': debug_msg
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_optimal_thresholds', methods=['GET'])
def get_optimal_thresholds():
    return jsonify({
        'status': 'success',
        'pred': optimal_pred_threshold,
        'rl': optimal_rl_threshold
    })

# --- BET HISTORY API ---
BET_HISTORY_FILE = os.path.join(_DIR, 'bet_history.json')

def load_bet_history():
    if os.path.exists(BET_HISTORY_FILE):
        try:
            with open(BET_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_bet_history(history):
    with open(BET_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

@app.route('/api/bet_history', methods=['GET'])
def get_bet_history():
    history = load_bet_history()
    # Sort history by added_at descending
    history.sort(key=lambda x: x.get('added_at', ''), reverse=True)
    
    # Optional: Attach outcomes from master_df if available
    for bet in history:
        # Evaluate Outcome
        # Preserve original outcome if it's already graded (e.g., from dummy data script)
        if bet.get('outcome') not in ['Hit', 'Miss', 'Passed']:
            bet['outcome'] = 'Pending'
        if bet.get('actual_score') is None or bet.get('actual_score') == '':
            bet['actual_score'] = 'N/A'
        
        try:
             # Find corresponding match in master_df
             b_date = pd.to_datetime(bet['match_date'])
             b_home = bet['home']
             b_away = bet['away']
             user_choice = bet['user_choice']
             
             # Locate match exactly or loosely
             match = master_df[(master_df['date_obj'] == b_date) & 
                               (master_df['home team'] == b_home) & 
                               (master_df['away team'] == b_away)]
                               
             if not match.empty:
                 row = match.iloc[0]
                 hg = row['home team total goal']
                 ag = row['away team total goal']
                 
                 bet['actual_score'] = f"{hg} - {ag}"
                 
                 actual_res = 'Draw'
                 if hg > ag: actual_res = 'Home'
                 elif ag > hg: actual_res = 'Away'
                 
                 # Resolve user choice 
                 # User choices typically match "Home", "Away", "Draw", etc. Convert if needed
                 choice_norm = str(user_choice).title()
                 
                 if choice_norm in ['Home', 'Away', 'Draw']:
                     if choice_norm == actual_res: bet['outcome'] = 'Hit'
                     else: bet['outcome'] = 'Miss'
                 elif choice_norm == "Pass":
                     bet['outcome'] = 'Passed'
        except:
             pass

    return jsonify({'status': 'success', 'data': history})

@app.route('/api/bet_history/add', methods=['POST'])
def add_bet_history():
    data = request.json
    if not data:
        return jsonify({'status': 'error', 'message': 'No data provided'}), 400

    history = load_bet_history()
    import uuid
    new_bet = {
        'id': str(uuid.uuid4()),
        'home': data.get('home', 'Unknown'),
        'away': data.get('away', 'Unknown'),
        'match_date': data.get('date', 'Unknown'),
        'pred_score': data.get('pred_score', '-'),
        'pred_win': data.get('pred_win', 0),
        'pred_draw': data.get('pred_draw', 0),
        'pred_loss': data.get('pred_loss', 0),
        'rl_pick': data.get('rl_pick', 'N/A'),
        'user_choice': data.get('user_choice', 'Unknown'),
        'odds_1': data.get('odds_1', 0.0),
        'odds_x': data.get('odds_x', 0.0),
        'odds_2': data.get('odds_2', 0.0),
        'added_at': datetime.now().isoformat()
    }
    
    history.append(new_bet)
    save_bet_history(history)
    return jsonify({'status': 'success', 'message': 'Bet added to history', 'id': new_bet['id']})

@app.route('/api/bet_history/add_batch', methods=['POST'])
def add_bet_history_batch():
    data = request.json
    matches = data.get('matches', [])
    if not matches:
        return jsonify({'status': 'error', 'message': 'No matches provided'}), 400

    history = load_bet_history()
    import uuid
    added_count = 0
    
    for mdata in matches:
         new_bet = {
             'id': str(uuid.uuid4()),
             'home': mdata.get('home', 'Unknown'),
             'away': mdata.get('away', 'Unknown'),
             'match_date': mdata.get('date', 'Unknown'),
             'pred_score': mdata.get('pred_score', '-'),
             'pred_win': mdata.get('pred_win', 0),
             'pred_draw': mdata.get('pred_draw', 0),
             'pred_loss': mdata.get('pred_loss', 0),
             'rl_pick': mdata.get('rl_pick', 'N/A'),
             'user_choice': mdata.get('user_choice', 'Unknown'),
             'odds_1': mdata.get('odds_1', 0.0),
             'odds_x': mdata.get('odds_x', 0.0),
             'odds_2': mdata.get('odds_2', 0.0),
             'added_at': datetime.now().isoformat()
         }
         history.append(new_bet)
         added_count += 1
         
    save_bet_history(history)
    return jsonify({'status': 'success', 'message': f'{added_count} bets added to history'})

@app.route('/api/bet_history/remove', methods=['POST'])
def remove_bet_history():
    data = request.json
    ids_to_remove = data.get('ids', [])
    if not ids_to_remove:
        return jsonify({'status': 'error', 'message': 'No IDs provided'}), 400
        
    history = load_bet_history()
    new_history = [bet for bet in history if bet.get('id') not in ids_to_remove]
    
    if len(history) == len(new_history):
         return jsonify({'status': 'error', 'message': 'No matching bets found to remove'}), 404
         
    save_bet_history(new_history)
    return jsonify({'status': 'success', 'message': f'Removed {len(history) - len(new_history)} bets'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader=False)