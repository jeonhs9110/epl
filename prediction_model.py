import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import glob
import os
import re
from datetime import datetime
import warnings
from scipy.stats import poisson
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

import pickle

# ==========================================
# 0. CONFIGURATION
# ==========================================
warnings.filterwarnings("ignore")

ENCODER_FILE = 'encoders.pkl'

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"\n[CUDA STATUS] SUCCESS: CUDA is available. Using {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device('cpu')
    print("\n[CUDA STATUS] WARNING: CUDA NOT DETECTED. Using CPU. This will be slow.\n")


# HYPERPARAMETERS (Level 6 - Advanced)
BATCH_SIZE = 512
SEQ_LENGTH = 10 # Global maximum sequence length for the PyTorch tensor

import json

# DYNAMIC LEAGUE-SPECIFIC SEQUENCE LENGTHS
# Load optimized sequence lengths if available, otherwise fallback to defaults
LEAGUE_SEQ_LENGTHS = {
    'Premier League': 5,
    'Ligue 1': 3,
    'LaLiga': 7,
}

if os.path.exists('optimal_seq_lengths.json'):
    try:
        with open('optimal_seq_lengths.json', 'r') as f:
            valid_lengths = json.load(f)
            # Only update if valid dictionary loaded
            if isinstance(valid_lengths, dict) and valid_lengths:
                LEAGUE_SEQ_LENGTHS.update(valid_lengths)
    except Exception as e:
        print(f"[WARNING] Failed to load optimal_seq_lengths.json: {e}")


EMBED_DIM = 256
LEAGUE_EMBED_DIM = 32
NUM_HEADS = 8
NUM_LAYERS = 3
DROPOUT = 0.2
LEARNING_RATE = 1e-4

# GNN CONFIG
GAT_HEADS = 4
GAT_DROPOUT = 0.2


# ==========================================
# 1. DATA UTILITIES
# ==========================================
def parse_football_date(date_str, season_start_year):
    if pd.isna(date_str) or str(date_str).strip() == '': return pd.NaT
    s = str(date_str).strip().split(' ')[0].replace('.', '/').replace('-', '/').rstrip('/')
    try:
        parts = s.split('/')
        if len(parts) == 2:
            d, m = int(parts[0]), int(parts[1])
            y = season_start_year if m >= 8 else season_start_year + 1
            return pd.Timestamp(year=y, month=m, day=d)
        elif len(parts) == 3:
            d, m, y = int(parts[0]), int(parts[1]), int(parts[2])
            if y < 100: y += 2000
            return pd.Timestamp(year=y, month=m, day=d)
        return pd.NaT
    except:
        return pd.NaT


def get_league_from_filename(filename):
    base = os.path.basename(filename)
    match = re.search(r'FOOTBALL_(.*?)_(?:\d{4}|RESULTS)', base)
    if match: return match.group(1)
    return "UNKNOWN_LEAGUE"


def normalize_league_name(name):
    if not isinstance(name, str): return "Unknown"
    name = name.strip()
    
    # 1. Remove Years (e.g. "2023 2024")
    name = re.sub(r'\s*\d{4}\s*\d{4}', '', name)
    name = re.sub(r'\s*\d{4}', '', name)
    
    # 2. Remove Parentheses (e.g. "(Spain)", "(England)")
    name = re.sub(r'\s*\(.*?\)', '', name)
    
    # 3. Specific Mappings
    manual_map = {
        "Laliga": "La Liga",
        "Laliga 2": "La Liga 2",
        "QPR": "QPR" # Keep as is (Team name, not league, ensuring safety)
    }
    
    # Case insensitive check
    for k, v in manual_map.items():
        if name.lower() == k.lower():
            return v
            
    return name.strip()

def save_encoders(le_team, le_league):
    with open(ENCODER_FILE, 'wb') as f:
        pickle.dump({'le_team': le_team, 'le_league': le_league}, f)
    print(f"[SYSTEM] Scalable Encoders saved to {ENCODER_FILE}")

def load_encoders():
    if os.path.exists(ENCODER_FILE):
        with open(ENCODER_FILE, 'rb') as f:
            data = pickle.load(f)
        print(f"[SYSTEM] Loaded Encoders from {ENCODER_FILE}")
        return data['le_team'], data['le_league']
    return None, None

def get_master_data():
    all_files = glob.glob('*_RESULTS.csv') + glob.glob('old_matches/*_RESULTS.csv')
    if not all_files: return None, None, None

    today = datetime.now()
    current_season_start = today.year if today.month >= 8 else today.year - 1

    df_list = []
    for f in all_files:
        try:
            temp_df = pd.read_csv(f)
            temp_df.columns = temp_df.columns.str.strip().str.lower()
            required = ['date', 'home team', 'away team', 'home team total goal', 'away team total goal']
            if not all(c in temp_df.columns for c in required): continue

            # Ensure Odds columns exist, default to 0.0 if not
            if 'odds_1' not in temp_df.columns: temp_df['odds_1'] = 0.0
            if 'odds_x' not in temp_df.columns: temp_df['odds_x'] = 0.0
            if 'odds_2' not in temp_df.columns: temp_df['odds_2'] = 0.0

            raw_league = ""
            if 'league' in temp_df.columns:
                # Use the column, but handle if it's NaN
                temp_df['league'] = temp_df['league'].fillna('')
                # Vectorized normalization not easily done with complex function, using apply
                temp_df['league_name'] = temp_df['league'].astype(str).apply(normalize_league_name)
            else:
                # Filename based
                raw_league = get_league_from_filename(f)
                norm_league = normalize_league_name(raw_league)
                temp_df['league_name'] = norm_league
            
            # Ensure xG columns exist
            if 'home_xg' not in temp_df.columns: temp_df['home_xg'] = 0.0
            if 'away_xg' not in temp_df.columns: temp_df['away_xg'] = 0.0
            
            # --- CLEAN TEAM NAMES ---
            # User reported 'TeamName2' and 'Lazio 3' artifacts. 
            # Flashscore sometimes appends numbers for reserve/B-teams or duplicate entries.
            # Regex: Remove ' 2', ' 3', '2', '3' at end of string.
            # We use `r'\s*[23]$'` which matches optional space followed by 2 or 3 at the end of the line.
            # This safely avoids Schalke 04, Hannover 96, Mainz 05 (since they end in 4, 6, 5 or have '0').
            temp_df['home team'] = temp_df['home team'].astype(str).str.replace(r'\s*[23]$', '', regex=True).str.strip()
            temp_df['away team'] = temp_df['away team'].astype(str).str.replace(r'\s*[23]$', '', regex=True).str.strip()
            # ------------------------
            
            # Handle NaNs in existing columns
            temp_df['home_xg'] = temp_df['home_xg'].fillna(0.0).astype(float)
            temp_df['away_xg'] = temp_df['away_xg'].fillna(0.0).astype(float)
            temp_df['odds_1'] = temp_df['odds_1'].fillna(0.0).astype(float)
            temp_df['odds_x'] = temp_df['odds_x'].fillna(0.0).astype(float)
            temp_df['odds_2'] = temp_df['odds_2'].fillna(0.0).astype(float)

            file_year = current_season_start
            match = re.search(r'(\d{4})_\d{4}', f)
            if match: file_year = int(match.group(1))

            valid_dates = []
            valid_indices = []
            for idx, row in temp_df.iterrows():
                dt = parse_football_date(row['date'], file_year)
                if not pd.isna(dt):
                    valid_dates.append(dt)
                    valid_indices.append(idx)

            if len(valid_dates) > 0:
                clean_df = temp_df.loc[valid_indices].copy()
                clean_df['date_obj'] = valid_dates
                df_list.append(clean_df)

        except Exception as e:
            print(f'Skipping {f}: {e}')

    if not df_list: return None, None, None
    
    master_df = pd.concat(df_list, ignore_index=True)
    master_df = master_df.sort_values('date_obj').reset_index(drop=True)
    
    # --- LOAD OR CREATE ENCODERS ---
    le_team, le_league = load_encoders()
    
    all_teams = pd.concat([master_df['home team'], master_df['away team']]).astype(str).unique()
    all_leagues = master_df['league_name'].astype(str).unique()
    
    if le_team is None:
        print("[SYSTEM] No existing encoders found. Fitting new ones...")
        le_team = LabelEncoder()
        le_team.fit(all_teams)
        le_league = LabelEncoder()
        le_league.fit(all_leagues)
        save_encoders(le_team, le_league)
    else:
        # Check for new items and extend if necessary
        # LabelEncoder doesn't support partial_fit, so we must refit if new classes appear
        # BUT we must keep old mapping consistent. 
        # Actually, standard LabelEncoder refit breaks IDs.
        # Strategy: converting to classes list, adding new ones, and refitting IS safe if sorted?
        # No, 'fit' sorts alphabetically. So if we add "Aaron FC", indices shift.
        # CRITICAL: We need robust handling. For now, we RE-FIT if new teams found but this MIGHT shift IDs
        # which invalidates the model.
        # The user's problem IS shifting IDs.
        # Correct fix: We should check if current classes cover all data.
        
        known_teams = set(le_team.classes_)
        new_teams = set(all_teams) - known_teams

        known_leagues = set(le_league.classes_)
        new_leagues = set(all_leagues) - known_leagues

        if new_teams or new_leagues:
            print(f"[WARNING] New teams/leagues detected! {len(new_teams)} Teams, {len(new_leagues)} Leagues.")
            print(">> Appending new entries to END of encoder (existing IDs preserved — no retraining needed).")

            # STABLE APPEND: new classes go to the END so existing IDs never shift.
            # LabelEncoder.classes_ is just a numpy array; we can set it directly.
            if new_teams:
                le_team.classes_ = np.append(le_team.classes_, sorted(list(new_teams)))
            if new_leagues:
                le_league.classes_ = np.append(le_league.classes_, sorted(list(new_leagues)))

            save_encoders(le_team, le_league)
            
    master_df['home_id'] = le_team.transform(master_df['home team'].astype(str))
    master_df['away_id'] = le_team.transform(master_df['away team'].astype(str))
    master_df['league_id'] = le_league.transform(master_df['league_name'].astype(str))

    return master_df, le_team, le_league

def _elo_k_margin(k_factor, goal_diff):
    """
    Scale the ELO K-factor by goal margin using FIFA-style logarithmic weighting.
    Keeps large-margin wins from over-inflating ratings while still rewarding dominance.
        GD=1 → k×1.0,  GD=2 → k×1.5,  GD=3+ → k×1.75 + log(GD-2)
    """
    if goal_diff <= 1:
        return k_factor
    elif goal_diff == 2:
        return k_factor * 1.5
    else:
        return k_factor * (1.75 + np.log(goal_diff - 1))

def calculate_dynamic_elo(df, k_factor=30, default_elo=1500.0):
    """
    Calculates ELO ratings for all teams in the dataframe timeline.
    Returns:
        elo_dict: {team_id: final_elo}
        h_elos: list of home elos per match
        a_elos: list of away elos per match
    """
    elo_ratings = {} # {team_id: current_elo}
    
    # Initialize all teams
    all_teams = np.unique(np.hstack([df['home_id'].values, df['away_id'].values]))
    for t in all_teams: elo_ratings[t] = default_elo
    
    h_elos = np.zeros(len(df))
    a_elos = np.zeros(len(df))
    
    h_ids = df['home_id'].values
    a_ids = df['away_id'].values
    hgs = df['home team total goal'].values
    ags = df['away team total goal'].values
    
    for i in range(len(df)):
        hid, aid = h_ids[i], a_ids[i]
        
        # 1. Store current ELO (Before match)
        h_elos[i] = elo_ratings[hid]
        a_elos[i] = elo_ratings[aid]
        
        # 2. Calculate Expected Result
        # E_A = 1 / (1 + 10 ^ ((Rb - Ra) / 400))
        # Added +100 to Home ELO to reflect inherent home field advantage
        diff_h = (elo_ratings[aid] - (elo_ratings[hid] + 100.0)) / 400.0
        diff_a = ((elo_ratings[hid] + 100.0) - elo_ratings[aid]) / 400.0
        exp_h = 1.0 / (1.0 + 10.0 ** diff_h)
        exp_a = 1.0 / (1.0 + 10.0 ** diff_a)
        
        # 3. Actual Result (0, 0.5, 1)
        if hgs[i] > ags[i]: res_h, res_a = 1.0, 0.0
        elif hgs[i] < ags[i]: res_h, res_a = 0.0, 1.0
        else: res_h, res_a = 0.5, 0.5
        
        # 4. Update Ratings (goal-margin scaled K)
        k = _elo_k_margin(k_factor, int(abs(hgs[i] - ags[i])))
        elo_ratings[hid] += k * (res_h - exp_h)
        elo_ratings[aid] += k * (res_a - exp_a)
        
    return elo_ratings, h_elos, a_elos


# ==========================================
# 2. DATASET (Enriched with GNN Graph)
# ==========================================
class SoccerDataset(Dataset):
    def __init__(self, df, tail=None):
        self.df = df
        print(f"    [OPTIMIZATION] Pre-calculating extended history for {len(df)} matches (Fast Mode)...")
        self.samples = []
        
        # Fast Access Arrays
        h_ids = df['home_id'].values
        a_ids = df['away_id'].values
        l_ids = df['league_id'].values
        dates = df['date_obj'].tolist()
        hgs = df['home team total goal'].values.astype(np.float32)
        ags = df['away team total goal'].values.astype(np.float32)
        hxgs = df['home_xg'].values.astype(np.float32)
        axgs = df['away_xg'].values.astype(np.float32)
        
        # Odds
        def clean_odd(x):
            try: return float(x)
            except: return 0.0
        o1 = df['odds_1'].apply(clean_odd).values.astype(np.float32)
        ox = df['odds_x'].apply(clean_odd).values.astype(np.float32)
        o2 = df['odds_2'].apply(clean_odd).values.astype(np.float32)

        # --- NEW: ELO TRACKING ---
        self.final_elos, h_elos, a_elos = calculate_dynamic_elo(df)
        
        # Normalize ELOs to 0-1 range for Neural Net
        self.max_elo = max(self.final_elos.values()) if self.final_elos else 1500
        self.min_elo = min(self.final_elos.values()) if self.final_elos else 1500
        # Avoid div zero
        if self.max_elo == self.min_elo:
            h_elos_norm = np.zeros_like(h_elos)
            a_elos_norm = np.zeros_like(a_elos)
        else:
            h_elos_norm = (h_elos - self.min_elo) / (self.max_elo - self.min_elo)
            a_elos_norm = (a_elos - self.min_elo) / (self.max_elo - self.min_elo)

        # Date Weights
        date_23_24_start = pd.Timestamp(2023, 8, 1)
        date_24_25_start = pd.Timestamp(2024, 8, 1)
        date_25_curr_start = pd.Timestamp(2025, 8, 1)
        
        # League Position Variance
        league_points = {}
        
        # Style Vectors for Contrastive Learning (Avg xG, Goals)
        # {team_id: [sum_goals, sum_xg, count]}
        self.team_stats = {}
        
        # Team History
        team_history = {}
        
        # --- NEW: 6-CATEGORY xG VARIANCE TRACKING ---
        # Track MAE of (Actual - xG) per team per category
        # Categories: 
        # 0: Home vs Strong (ELO >= 1550)
        # 1: Home vs Mid   (1450 <= ELO < 1550)
        # 2: Home vs Weak  (ELO < 1450)
        # 3: Away vs Strong
        # 4: Away vs Mid
        # 5: Away vs Weak
        # Form: {team_id: {cat_id: [sum_error, count]}}
        self.team_xg_variance = {}
        
        lookback = SEQ_LENGTH
        skipped_zero_odds = 0
        
        def build_seq(t_id, curr_date, optimal_seq_length=None):
            hist = team_history.get(t_id, [])
            feats = []
            
            c_y = curr_date.year
            c_m = curr_date.month
            if c_m >= 8: seas_start = pd.Timestamp(year=c_y, month=8, day=1)
            else: seas_start = pd.Timestamp(year=c_y-1, month=8, day=1)
            
            season_matches = [m for m in hist if m[0] >= seas_start]
            if season_matches:
                seas_avg_xg = sum(m[3] for m in season_matches) / len(season_matches)
                seas_avg_xga = sum(m[4] for m in season_matches) / len(season_matches)
            else:
                seas_avg_xg, seas_avg_xga = 1.3, 1.3

            available = len(hist)
            start_idx = max(0, available - lookback)
            segment = hist[start_idx:]
            
            for i, match in enumerate(segment):
                m_date, m_gf, m_ga, m_xg, m_xga, m_is_home, m_l_var = match
                
                if i > 0:
                    delta = (m_date - segment[i-1][0]).days
                    rest_days = min(delta, 14)
                elif start_idx > 0:
                     delta = (m_date - hist[start_idx-1][0]).days
                     rest_days = min(delta, 14)
                else: rest_days = 7
                rest_norm = rest_days / 7.0
                
                
                abs_idx = start_idx + i
                form_start = max(0, abs_idx - 5)
                form_window = hist[form_start : abs_idx]
                
                # Default 6-cat variances (Fallback to 0.0)
                cat_vars = [0.0] * 6
                
                if form_window:
                    avg_gf = sum(m[1] for m in form_window) / len(form_window)
                    avg_ga = sum(m[2] for m in form_window) / len(form_window)
                    avg_xg = sum(m[3] for m in form_window) / len(form_window)
                    avg_xga = sum(m[4] for m in form_window) / len(form_window)
                    
                    # Compute rolling variances per category from history up to this point
                    # We look at all historical matches prior to this one to populate the 6 categories natively
                    # To keep it fast, we query the `self.team_xg_variance` which is being built chronologically in the outer loop
                    # However, since `build_seq` uses historical data, we need the snapshot of stats at that exact time.
                    # Since team_xg_variance is updated daily, this sequence will use the "latest" known variances.
                    team_var_dict = self.team_xg_variance.get(t_id, {})
                    for c in range(6):
                        if c in team_var_dict and team_var_dict[c][1] > 0:
                            cat_vars[c] = team_var_dict[c][0] / team_var_dict[c][1]
                            
                else:
                    avg_gf, avg_ga, avg_xg, avg_xga = 1.3, 1.3, 1.3, 1.3
                
                # Expand feat length: 14 base + 6 context variances = 20 dims
                feat = [
                    m_gf, m_ga, m_xg, m_xga, m_is_home, rest_norm, 
                    avg_gf/3.0, avg_ga/3.0, avg_xg/3.0, avg_xga/3.0,
                    seas_avg_xg/3.0, seas_avg_xga/3.0,
                    m_l_var/10.0, 
                    # 13 is unused padding previously, use it for generic MAE if we want, or just pad
                    0.0 
                ]
                # Append the 6 context-specific variances
                feat.extend([v/3.0 for v in cat_vars])
                feats.append(feat)
            
            feats_np = np.array(feats, dtype=np.float32)

            if len(feats_np) == 0:
                return np.zeros((SEQ_LENGTH, 20), dtype=np.float32)
                
            # DYNAMIC SEQUENCE LENGTH TRUNCATION
            # If the optimal length is less than the gathered history, truncate the oldest matches
            # and pad the rest with 0s at the front, forcing the model to ignore stale history.
            if optimal_seq_length is not None and optimal_seq_length < len(feats_np):
                feats_np = feats_np[-optimal_seq_length:]
                
            if len(feats_np) < SEQ_LENGTH:
                pad_len = SEQ_LENGTH - len(feats_np)
                feats_np = np.pad(feats_np, ((pad_len, 0), (0, 0)), mode='constant')
            return feats_np

        # --- PRE-BUILD league id → name dict (avoids pickle reload per sample) ---
        _id_to_league = {}
        try:
            with open(ENCODER_FILE, 'rb') as _f:
                _tmp_team, _tmp_league = pickle.load(_f)
            for _lid, _lname in enumerate(_tmp_league.classes_):
                _id_to_league[_lid] = _lname
        except Exception:
            pass

        for idx in range(len(df)):
            match_date = dates[idx]
            match_ts = pd.Timestamp(match_date)
            l_id = l_ids[idx]
            h_id = h_ids[idx]
            a_id = a_ids[idx]
            
            # --- 1. CALCULATE LEAGUE VARIANCE ---
            current_league_points = league_points.get(l_id, {})
            if len(current_league_points) > 2:
                pts_values = list(current_league_points.values())
                l_var = float(np.std(pts_values))
            else:
                l_var = 0.0
            
            if o1[idx] == 0.0 and ox[idx] == 0.0 and o2[idx] == 0.0:
                skipped_zero_odds += 1
                # Update Points & History only
                if l_id not in league_points: league_points[l_id] = {}
                if h_id not in league_points[l_id]: league_points[l_id][h_id] = 0
                if a_id not in league_points[l_id]: league_points[l_id][a_id] = 0
                
                if hgs[idx] > ags[idx]: league_points[l_id][h_id] += 3
                elif ags[idx] > hgs[idx]: league_points[l_id][a_id] += 3
                else:
                    league_points[l_id][h_id] += 1
                    league_points[l_id][a_id] += 1
                
                if h_id not in team_history: team_history[h_id] = []
                if a_id not in team_history: team_history[a_id] = []
                team_history[h_id].append((match_date, hgs[idx], ags[idx], hxgs[idx], axgs[idx], 1.0, l_var))
                team_history[a_id].append((match_date, ags[idx], hgs[idx], axgs[idx], hxgs[idx], 0.0, l_var))
                
                # Update Stats for Contrastive
                if h_id not in self.team_stats: self.team_stats[h_id] = [0,0,0]
                if a_id not in self.team_stats: self.team_stats[a_id] = [0,0,0]
                self.team_stats[h_id][0] += hgs[idx]
                self.team_stats[h_id][1] += hxgs[idx]
                self.team_stats[h_id][2] += 1
                self.team_stats[a_id][0] += ags[idx]
                self.team_stats[a_id][1] += axgs[idx]
                self.team_stats[a_id][2] += 1
                continue

            # TIME DECAY WEIGHTING (Training Loss)
            # Replaces hardcoded steps with smooth exponential decay
            # Goal: Current matches ~1.1, 1 year ago ~0.6, 2 years ago ~0.4
            
            # Days since match
            days_diff = (pd.Timestamp(datetime.now()) - match_ts).days
            
            # Formula: Base (0.2) + Recency_Bonus * Decay
            # 0 days: 0.2 + 0.9 * 1.0 = 1.1
            # 365 days: 0.2 + 0.9 * 0.48 = ~0.63
            # 730 days: 0.2 + 0.9 * 0.23 = ~0.40
            w = 0.2 + 0.9 * np.exp(-0.002 * max(0, days_diff))
            
            # --- CONTEXT-AWARE xG CATEGORIZATION ---
            def get_cat(is_home, opp_elo):
                if opp_elo >= 1550: return 0 if is_home else 3
                elif opp_elo >= 1450: return 1 if is_home else 4
                else: return 2 if is_home else 5
                
            h_cat = get_cat(True, a_elos[idx])
            a_cat = get_cat(False, h_elos[idx])
            
            hxge = abs(hgs[idx] - hxgs[idx])
            axge = abs(ags[idx] - axgs[idx])
            
            # ACCURACY HITTING RATE WEIGHTING (Down-weight Flukes based on Category Norms)
            h_var_dict = self.team_xg_variance.get(h_id, {})
            a_var_dict = self.team_xg_variance.get(a_id, {})
            
            # If historical error for this category exists, compare current error to history
            h_hist_var = (h_var_dict[h_cat][0] / h_var_dict[h_cat][1]) if h_cat in h_var_dict and h_var_dict[h_cat][1] > 0 else 0.5
            a_hist_var = (a_var_dict[a_cat][0] / a_var_dict[a_cat][1]) if a_cat in a_var_dict and a_var_dict[a_cat][1] > 0 else 0.5
            
            # Normal variance ratio: How bad is this error compared to their usual error in this context?
            # E.g. Error = 1.5. Normal historic error = 1.0. Ratio = 1.5.
            # If Ratio is massive (fluke), weight shrinks. If ratio is tiny (clinical/expected), weight > 1.
            h_ratio = hxge / (h_hist_var + 1e-5)
            a_ratio = axge / (a_hist_var + 1e-5)
            
            match_xg_ratio = (h_ratio + a_ratio) / 2.0
            
            # Dampen the penalty so it's not too aggressive.
            # E.g., ratio=3 -> acc_weight = 1/(1+1.5) = 1/2.5 = 0.4.
            acc_weight = 1.0 / (1.0 + (match_xg_ratio * 0.5)) 
            
            # Combine temporal weight with accuracy weight
            w = w * acc_weight

            # Lookup Dynamic League Optimal Sequence Length
            # Uses pre-built id->name dict (built before the loop) to avoid disk I/O per sample
            league_str = _id_to_league.get(l_ids[idx], 'Unknown')
            opt_len = LEAGUE_SEQ_LENGTHS.get(league_str, SEQ_LENGTH)

            h_seq_np = build_seq(h_id, match_date, opt_len)
            a_seq_np = build_seq(a_id, match_date, opt_len)
            
            # --- STYLE VECTOR (for Contrastive Learning) ---
            # Normalized [AvgGoals, AvgXG]
            def get_style(tid):
                s = self.team_stats.get(tid, [0,0,0])
                if s[2] == 0: return [1.5, 1.5]
                return [s[0]/s[2], s[1]/s[2]]
            
            h_style = get_style(h_id)
            a_style = get_style(a_id)

            self.samples.append({
                'h_seq': torch.from_numpy(h_seq_np),
                'a_seq': torch.from_numpy(a_seq_np),
                'h_id': torch.tensor(h_ids[idx], dtype=torch.long),
                'a_id': torch.tensor(a_ids[idx], dtype=torch.long),
                'l_id': torch.tensor(l_ids[idx], dtype=torch.long),
                'odds': torch.tensor([o1[idx], ox[idx], o2[idx]], dtype=torch.float32),
                'hg': torch.tensor(hgs[idx], dtype=torch.float32),
                'ag': torch.tensor(ags[idx], dtype=torch.float32),
                'h_xg': torch.tensor(hxgs[idx], dtype=torch.float32),
                'a_xg': torch.tensor(axgs[idx], dtype=torch.float32),
                'weight': torch.tensor(w, dtype=torch.float32),
                'h_style': torch.tensor(h_style, dtype=torch.float32), 
                'a_style': torch.tensor(a_style, dtype=torch.float32),
                'h_elo': torch.tensor(h_elos_norm[idx], dtype=torch.float32),
                'a_elo': torch.tensor(a_elos_norm[idx], dtype=torch.float32)
            })
            
            # --- UPDATE STATE ---
            if l_id not in league_points: league_points[l_id] = {}
            if h_id not in league_points[l_id]: league_points[l_id][h_id] = 0
            if a_id not in league_points[l_id]: league_points[l_id][a_id] = 0
            
            if hgs[idx] > ags[idx]: league_points[l_id][h_id] += 3
            elif ags[idx] > hgs[idx]: league_points[l_id][a_id] += 3
            else:
                league_points[l_id][h_id] += 1
                league_points[l_id][a_id] += 1
                
            if h_id not in team_history: team_history[h_id] = []
            if a_id not in team_history: team_history[a_id] = []
            team_history[h_id].append((match_date, hgs[idx], ags[idx], hxgs[idx], axgs[idx], 1.0, l_var))
            team_history[a_id].append((match_date, ags[idx], hgs[idx], axgs[idx], hxgs[idx], 0.0, l_var))

            if h_id not in self.team_stats: self.team_stats[h_id] = [0,0,0]
            if a_id not in self.team_stats: self.team_stats[a_id] = [0,0,0]
            self.team_stats[h_id][0] += hgs[idx]
            self.team_stats[h_id][1] += hxgs[idx]
            self.team_stats[h_id][2] += 1
            self.team_stats[a_id][0] += ags[idx]
            self.team_stats[a_id][1] += axgs[idx]
            self.team_stats[a_id][2] += 1
            
            # UPDATE TEAM xG CATEGORY VARIANCES
            if h_id not in self.team_xg_variance: self.team_xg_variance[h_id] = {}
            if a_id not in self.team_xg_variance: self.team_xg_variance[a_id] = {}
            
            if h_cat not in self.team_xg_variance[h_id]: self.team_xg_variance[h_id][h_cat] = [0.0, 0]
            if a_cat not in self.team_xg_variance[a_id]: self.team_xg_variance[a_id][a_cat] = [0.0, 0]
            
            self.team_xg_variance[h_id][h_cat][0] += hxge
            self.team_xg_variance[h_id][h_cat][1] += 1
            
            self.team_xg_variance[a_id][a_cat][0] += axge
            self.team_xg_variance[a_id][a_cat][1] += 1
        
        if tail is not None:
            self.samples = self.samples[-tail:]
            print(f"    [OPTIMIZATION] Subsetted to last {tail} matches.")
            
        print(f"    [DATASET] Skipped {skipped_zero_odds} matches due to missing odds ([0.0, 0.0, 0.0]).")
        
        # --- BUILD GLOBAL ADJACENCY FOR GNN ---
        # Edge Logic: If teams played in last N matches, add edge.
        # We will iterate backwards from end.
        print("    [GNN] Building Global Graph Adjacency...")
        num_teams = len(np.unique(np.hstack([h_ids, a_ids])))
        # Adjacency Matrix: [NumTeams, NumTeams]
        # We use a sparse collection of edges for PyTorch Geometric style GAT, 
        # OR dense for small graphs. NumTeams ~ 100-200. Dense is fine (~40000 ints).
        # Actually max ID might be larger.
        max_id = max(max(h_ids), max(a_ids))
        self.adj = torch.zeros((max_id + 1, max_id + 1), dtype=torch.float32)
        
        # Fill edges based on recent matches (Last 2000?)
        # Edges feature: 1.0 (Played)
        recent_matches = zip(h_ids[-2000:], a_ids[-2000:]) if len(h_ids) > 2000 else zip(h_ids, a_ids)
        for h, a in recent_matches:
            self.adj[h, a] = 1.0
            self.adj[a, h] = 1.0
            # Self loops
            self.adj[h, h] = 1.0
            self.adj[a, a] = 1.0
            
        # Normalize Adjacency (Row normalize)
        rowsum = self.adj.sum(1)
        # Avoid zero div
        rowsum[rowsum == 0] = 1.0
        self.adj = self.adj / rowsum.unsqueeze(1)
        
        print("    [OPTIMIZATION] Complete.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ==========================================
# 3. ARCHITECTURES (GNN + Transformer)
# ==========================================

class SimpleGATLayer(nn.Module):
    """
    Simple Graph Attention Layer.
    H_val = Activation( Aggregation( Attention(H_nodes) ) )
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SimpleGATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h: [NumNodes, in_features]
        # adj: [NumNodes, NumNodes] (Normalized or raw)
        
        Wh = self.W(h) # [N, out_feats]
        N = Wh.size(0)
        
        # Attention Mechanism (All-to-All masked by Adj)
        # Construct [N, N, 2*out] -> Too big?
        # Use simpler GCN-like propagation with Attention weights per edge?
        # For simplicity in this bespoke implementation without torch_geometric:
        # We will do a masked attention.
        
        # a_input: [N, N, 2*out] is O(N^2). If N=500, N^2=250k. Manageable.
        # Wh_repeated: [N, N, out]
        Wh1 = Wh.repeat(N, 1, 1)
        Wh2 = Wh.repeat(N, 1, 1).transpose(0, 1)
        
        # [N, N, 2*out]
        a_input = torch.cat([Wh1, Wh2], dim=2)
        
        # [N, N, 1]
        e = self.leakyrelu(self.a(a_input).squeeze(2))
        
        # Mask with Adjacency
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Aggregation: [N, N] x [N, out] -> [N, out]
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=SEQ_LENGTH):
        super().__init__()
        self.encoding = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.uniform_(self.encoding, -0.1, 0.1)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:seq_len, :].unsqueeze(0)


class TemporalTransformerEncoder(nn.Module):
    def __init__(self, max_seq_len=SEQ_LENGTH):
        super().__init__()
        self.input_proj = nn.Linear(20, EMBED_DIM)
        self.pos_encoder = LearnablePositionalEncoding(EMBED_DIM, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS,
                                                   dim_feedforward=EMBED_DIM * 4,
                                                   dropout=DROPOUT, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        # Learnable Attention Pooling instead of simple mean
        self.attn_pool = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM // 2),
            nn.Tanh(),
            nn.Linear(EMBED_DIM // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        # Attention Pooling over the sequence
        weights = self.attn_pool(x) # [batch, seq_len, 1]
        pooled = torch.sum(x * weights, dim=1) # [batch, EMBED_DIM]
        return pooled


class TabularFeatureGRN(nn.Module):
    """Gated Residual Network for Tabular Dense Features (inspired by TabNet)"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        res = self.res_proj(x)
        h = F.gelu(self.fc1(x))
        h = self.fc2(h)
        gate = torch.sigmoid(self.gate(x))
        h = h * gate
        h = self.dropout(h)
        return self.norm(res + h)

class TacticalInteractionCrossAttention(nn.Module):
    """
    Replaces static element-wise mismatch with Learnable Multi-Head Cross Attention
    to find explicit tactical intersections.
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, h_emb, a_emb):
        # Treat as sequence of length 1: [batch, 1, dim]
        h_emb_u = h_emb.unsqueeze(1)
        a_emb_u = a_emb.unsqueeze(1)
        
        # Home attending to Away
        attn_out_h, _ = self.cross_attn(h_emb_u, a_emb_u, a_emb_u)
        h_interact = self.norm1(h_emb_u + attn_out_h)
        
        # Away attending to Home
        attn_out_a, _ = self.cross_attn(a_emb_u, h_emb_u, h_emb_u)
        a_interact = self.norm2(a_emb_u + attn_out_a)
        
        combined = torch.cat([h_interact.squeeze(1), a_interact.squeeze(1)], dim=1)
        return combined

class LeagueAwareModel(nn.Module):
    def __init__(self, num_teams, num_leagues, dataset_adj=None):
        super().__init__()
        self.team_emb = nn.Embedding(num_teams, EMBED_DIM)
        self.league_emb = nn.Embedding(num_leagues, LEAGUE_EMBED_DIM)
        
        # --- GNN LAYER (Relational Form) ---
        self.gat = SimpleGATLayer(EMBED_DIM, EMBED_DIM, dropout=GAT_DROPOUT, alpha=0.2, concat=False)
        # Store Adjacency as buffer (not param)
        if dataset_adj is not None:
             self.register_buffer('adj', dataset_adj)
        else:
             self.register_buffer('adj', torch.eye(num_teams))
             
        self.encoder = TemporalTransformerEncoder()
        self.cross_attn = nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, batch_first=True)
        self.norm = nn.LayerNorm(EMBED_DIM)
        
        # --- LEVEL 8: TABULAR DENSE PROCESSING ---
        self.tabular_grn = TabularFeatureGRN(input_dim=5, hidden_dim=64, output_dim=32)
        
        # --- LEVEL 8: CROSS-ATTENTION TACTICAL INTERACTION ---
        self.tactical = TacticalInteractionCrossAttention(EMBED_DIM)
        
        # 1. Embeddings (Home/Away/League)
        # 2. Relational Embeddings (Home/Away from GAT)
        # 3. Transformer Features
        # 4. Tactical Interaction (EMBED_DIM * 2)
        # 5. Tabular GRN (32)
        
        fusion_dim = (EMBED_DIM * 6) + LEAGUE_EMBED_DIM + (EMBED_DIM * 2) + 32
        
        self.shared = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1)
        )
        
        self.goal_head = nn.Linear(256, 3) 
        self.xg_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2))

    def get_relational_embeddings(self):
        # Apply GAT to the full Team Embedding Table
        all_embs = self.team_emb.weight 
        # Pad adj if embedding table larger than adj (e.g. new teams?)
        # Assume adj matches num_teams.
        curr_adj = self.adj
        if all_embs.shape[0] > curr_adj.shape[0]:
            # Pad
            diff = all_embs.shape[0] - curr_adj.shape[0]
            curr_adj = F.pad(curr_adj, (0, diff, 0, diff))
            
        rel_embs = self.gat(all_embs, curr_adj)
        return rel_embs

    def forward(self, h_seq, a_seq, h_id, a_id, l_id, odds, h_elo, a_elo):
        # 1. Static Embeddings
        h_s = self.team_emb(h_id)
        a_s = self.team_emb(a_id)
        l_s = self.league_emb(l_id)
        
        # 2. Relational Embeddings (GNN)
        rel_embs = self.get_relational_embeddings()
        h_r = rel_embs[h_id]
        a_r = rel_embs[a_id]
        
        # 3. Sequential Form (Transformer)
        h_c = self.encoder(h_seq)
        a_c = self.encoder(a_seq)
        
        # Cross Interaction
        h_c_u = h_c.unsqueeze(1)
        a_c_u = a_c.unsqueeze(1)
        h_int, _ = self.cross_attn(h_c_u, a_c_u, a_c_u)
        a_int, _ = self.cross_attn(a_c_u, h_c_u, h_c_u)
        h_final = self.norm(h_c + h_int.squeeze(1))
        a_final = self.norm(a_c + a_int.squeeze(1))
        
        # 4. Tactical Interaction (Cross-Attention)
        tactical_feat = self.tactical(h_s, a_s)
        
        # Combine embeddings and sequence features
        combined = torch.cat([h_s, a_s, h_r, a_r, h_final, a_final, l_s, tactical_feat], dim=1)
        
        # 5. Tabular GRN for Odds and ELO
        tabular_input = torch.cat([odds, h_elo.unsqueeze(1), a_elo.unsqueeze(1)], dim=1)
        tabular_feat = self.tabular_grn(tabular_input)
        
        final_input = torch.cat([combined, tabular_feat], dim=1)
        
        shared_out = self.shared(final_input)
        
        goal_params = self.goal_head(shared_out)
        xg_params = self.xg_head(shared_out)
        
        lambdas = torch.nn.functional.softplus(goal_params[:, :2])
        rho = torch.tanh(goal_params[:, 2]) * 0.9 
        
        return lambdas, rho, xg_params, h_s, a_s # Return embeddings for Contrastive Loss
    
    def extract_features(self, h_seq, a_seq, h_id, a_id, l_id, h_elo, a_elo):
        # Feature Extraction for XGBoost
        with torch.no_grad():
            h_s = self.team_emb(h_id)
            a_s = self.team_emb(a_id)
            l_s = self.league_emb(l_id)
            rel_embs = self.get_relational_embeddings()
            h_r = rel_embs[h_id]
            a_r = rel_embs[a_id]
            h_c = self.encoder(h_seq)
            a_c = self.encoder(a_seq)
            h_c_u = h_c.unsqueeze(1)
            a_c_u = a_c.unsqueeze(1)
            h_int, _ = self.cross_attn(h_c_u, a_c_u, a_c_u)
            a_int, _ = self.cross_attn(a_c_u, h_c_u, h_c_u)
            h_final = self.norm(h_c + h_int.squeeze(1))
            a_final = self.norm(a_c + a_int.squeeze(1))
            
            tactical_feat = self.tactical(h_s, a_s)
            
            # Since extract_features is used for XGBoost, we can choose to return the combined features
            combined = torch.cat([h_s, a_s, h_r, a_r, h_final, a_final, l_s, tactical_feat], dim=1)
        return combined


# ==========================================
# 4. LOSS & SIMULATION
# ==========================================

def contrastive_loss(h_emb, a_emb, h_style, a_style):
    """
    Forces Embeddings to respect Style similarity.
    If Style Diff is Small, Embedding Cosine should be High.
    """
    # Normalize
    h_emb = F.normalize(h_emb, p=2, dim=1)
    a_emb = F.normalize(a_emb, p=2, dim=1)
    
    # Cosine Similarity of Embeddings
    emb_sim = torch.sum(h_emb * a_emb, dim=1)
    
    # Style Similarity (Negative L2 Distance or Cosine)
    # Styles are [AvgGoals, AvgXG] -> 2D
    # Let's use 1 / (1 + Distance)
    style_dist = torch.norm(h_style - a_style, p=2, dim=1)
    target_sim = 1.0 / (1.0 + style_dist)
    
    # MSE Loss between Actual Embedding Similarity and Target Style Similarity
    return F.mse_loss(emb_sim, target_sim)

class DixonColesLoss(nn.Module):
    def __init__(self, weight=None, draw_weight=1.5):
        super(DixonColesLoss, self).__init__()
        self.weight = weight
        # Poisson models chronically underpredict draws (~27% in football).
        # Upweighting draw samples during training partially corrects this bias.
        self.draw_weight = draw_weight

    def forward(self, lambdas, rho, targets, weights=None):
        # lambdas: [batch, 2] (home_lambda, away_lambda)
        # rho: [batch, 1] (correlation coefficient)
        # targets: [batch, 2] (home_goals, away_goals)
        
        lambda_h = lambdas[:, 0]
        lambda_a = lambdas[:, 1]
        h_goals = targets[:, 0]
        a_goals = targets[:, 1]
        
        # 1. Standard Poisson Log-Likelihood
        log_prob_h = h_goals * torch.log(lambda_h + 1e-7) - lambda_h - torch.lgamma(h_goals + 1)
        log_prob_a = a_goals * torch.log(lambda_a + 1e-7) - lambda_a - torch.lgamma(a_goals + 1)
        log_prob = log_prob_h + log_prob_a

        # 2. Dixon-Coles Adjustment Function (Tau)
        # We construct a correction matrix for 0-0, 1-0, 0-1, 1-1
        
        # Case 0-0
        mask_00 = (h_goals == 0) & (a_goals == 0)
        # Clamp to prevent log(negative)
        term_00 = torch.clamp(1.0 - (lambda_h * lambda_a * rho), min=1e-6)
        correction_00 = torch.log(term_00)
        
        # Case 0-1
        mask_01 = (h_goals == 0) & (a_goals == 1)
        term_01 = torch.clamp(1.0 + (lambda_h * rho), min=1e-6)
        correction_01 = torch.log(term_01)
        
        # Case 1-0
        mask_10 = (h_goals == 1) & (a_goals == 0)
        term_10 = torch.clamp(1.0 + (lambda_a * rho), min=1e-6)
        correction_10 = torch.log(term_10)
        
        # Case 1-1
        mask_11 = (h_goals == 1) & (a_goals == 1)
        term_11 = torch.clamp(1.0 - rho, min=1e-6)
        correction_11 = torch.log(term_11)

        # Apply corrections only where masks are true
        total_correction = torch.zeros_like(log_prob)
        total_correction[mask_00] = correction_00[mask_00]
        total_correction[mask_01] = correction_01[mask_01]
        total_correction[mask_10] = correction_10[mask_10]
        total_correction[mask_11] = correction_11[mask_11]

        final_log_prob = log_prob + total_correction

        # Draw upweighting: multiply the log-likelihood of draw outcomes by draw_weight
        # so the model is penalised more for missing draws during training.
        draw_mask = (h_goals == a_goals)
        sample_weight = torch.ones_like(final_log_prob)
        sample_weight[draw_mask] = self.draw_weight

        if weights is not None:
            return -torch.mean(final_log_prob * weights * sample_weight)
        return -torch.mean(final_log_prob * sample_weight)

def bivariate_poisson_loss(lambdas, rho, targets, weights=None):
    # Standard Independent Poisson Loss + Multi-Task handles the rest
    l1 = lambdas[:, 0]
    l2 = lambdas[:, 1]
    loss_h = l1 - targets[:, 0] * torch.log(l1 + 1e-6)
    loss_a = l2 - targets[:, 1] * torch.log(l2 + 1e-6)
    return torch.mean((loss_h + loss_a) * weights)

def calculate_probabilities(home_lambda, away_lambda, rho=0.0, max_goals=10):
    """
    Analytically exact Dixon-Coles corrected probability calculation.
    Replaces crude Monte Carlo — deterministic, fast, and more accurate.

    The DC correction adjusts the joint Poisson mass for low-scoring outcomes:
        tau(0,0) = 1 - lambda_h * lambda_a * rho
        tau(0,1) = 1 + lambda_h * rho
        tau(1,0) = 1 + lambda_a * rho
        tau(1,1) = 1 - rho
        tau(i,j) = 1  for all other (i,j)
    """
    grid = generate_score_grid(home_lambda, away_lambda, rho=rho, max_goals=max_goals)
    home_win = float(np.sum(np.tril(grid, -1)))
    draw     = float(np.sum(np.diag(grid)))
    away_win = float(np.sum(np.triu(grid, 1)))
    # Expected goals from marginal distributions
    goals_range = np.arange(max_goals)
    h_probs = np.array([poisson.pmf(i, home_lambda) for i in goals_range])
    a_probs = np.array([poisson.pmf(i, away_lambda) for i in goals_range])
    return {
        "home_win": home_win,
        "draw":     draw,
        "away_win": away_win,
        "score_h":  float(np.dot(goals_range, h_probs)),
        "score_a":  float(np.dot(goals_range, a_probs)),
    }


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    loss_fn_dixon = DixonColesLoss()

    # Use the modern torch.amp API (PyTorch ≥ 2.0). Fall back to the old
    # cuda.amp namespace for older installs so nothing breaks.
    if device.type == 'cuda':
        try:
            _scaler = torch.amp.GradScaler('cuda')
            def _autocast(): return torch.amp.autocast(device_type='cuda')
        except TypeError:
            _scaler = torch.cuda.amp.GradScaler()
            def _autocast(): return torch.cuda.amp.autocast()

    for b in loader:
        optimizer.zero_grad()
        hg, ag = b['hg'].to(device), b['ag'].to(device)
        h_xg, a_xg = b['h_xg'].to(device), b['a_xg'].to(device)
        h_style, a_style = b['h_style'].to(device), b['a_style'].to(device)
        h_elo, a_elo = b['h_elo'].to(device), b['a_elo'].to(device)
        weights = b['weight'].to(device)

        if device.type == 'cuda':
            with _autocast():
                odds_input = b['odds'].to(device)
                
                lambdas, rho, xg_params, h_emb, a_emb = model(
                    b['h_seq'].to(device), b['a_seq'].to(device),
                    b['h_id'].to(device), b['a_id'].to(device), b['l_id'].to(device),
                    odds_input, h_elo, a_elo
                )
                
                # Use Dixon-Coles Loss
                loss_goals = loss_fn_dixon(lambdas, rho, torch.stack([hg, ag], dim=1), weights=weights)
                # Only compute xG loss where actual xG was recorded (non-zero).
                # Without masking, the model gets penalised for predicting real xG values
                # on matches where xG was never scraped (stored as 0.0), poisoning the head.
                xg_target = torch.stack([h_xg, a_xg], dim=1)
                xg_mask = (h_xg > 0) | (a_xg > 0)
                if xg_mask.sum() > 0:
                    loss_xg = F.mse_loss(xg_params[xg_mask], xg_target[xg_mask])
                else:
                    loss_xg = torch.tensor(0.0, device=device)
                loss_cont = contrastive_loss(h_emb, a_emb, h_style, a_style)

                loss = loss_goals + (0.5 * loss_xg) + (0.1 * loss_cont)
                
            _scaler.scale(loss).backward()
            _scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            _scaler.step(optimizer)
            _scaler.update()
        else:
            lambdas, rho, xg_params, h_emb, a_emb = model(
                b['h_seq'], b['a_seq'], b['h_id'], b['a_id'], b['l_id'], b['odds'], h_elo, a_elo
            )
            loss_goals = loss_fn_dixon(lambdas, rho, torch.stack([hg, ag], dim=1), weights=weights)
            xg_target = torch.stack([h_xg, a_xg], dim=1)
            xg_mask = (h_xg > 0) | (a_xg > 0)
            if xg_mask.sum() > 0:
                loss_xg = F.mse_loss(xg_params[xg_mask], xg_target[xg_mask])
            else:
                loss_xg = torch.tensor(0.0)
            loss_cont = contrastive_loss(h_emb, a_emb, h_style, a_style)
            loss = loss_goals + (0.5 * loss_xg) + (0.1 * loss_cont)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        total_loss += loss.item()
        
        # Calculate Accuracy (W/D/L)
        # lambdas: [B, 2] -> h_lam, a_lam
        # Simple heuristic: Higher lambda = Higher expected goals
        # Or better: P(H>A), P(D), P(A>H). 
        # For speed in training loop, let's just use expected goals comparison
        # (This is a rough proxy but fast)
        l_h = lambdas[:, 0]
        l_a = lambdas[:, 1]
        
        pred_res = torch.zeros_like(l_h) # 0=A, 1=D, 2=H ? No, let's match label 
        # labels: hg, ag
        # Let's map: HomeWin=0, Draw=1, AwayWin=2
        
        actual_res = torch.zeros_like(l_h)
        actual_res[hg > ag] = 0.0
        actual_res[hg == ag] = 1.0
        actual_res[ag > hg] = 2.0
        
        # Prediction
        # We need probabilities, but comparing lambdas is a decent short-cut
        # If l_h > l_a + threshold -> HomeWin
        # If l_a > l_h + threshold -> AwayWin
        # Else Draw
        # This is arbitrary. 
        # Let's stick to strict: l_h > l_a -> Home. l_h < l_a -> Away. Equal -> Draw (Rare with floats)
        # But draw is frequent. 
        # Dixon-Coles models often predict 0-0, 1-1, 1-0...
        
        # Alternative: Just compare (l_h - l_a).
        # We want a metric that "moves".
        # Let's just use strict inequality for H/A. Draws will be missed.
        # This "Accuracy" is just a monitoring stat.
        
        # Better: define correctness as "Direction Correct"
        pred_res[:] = 1.0 # Default Draw
        pred_res[l_h > l_a + 0.1] = 0.0 # Home
        pred_res[l_a > l_h + 0.1] = 2.0 # Away
        
        correct = (pred_res == actual_res).sum().item()
        total_correct += correct
        total_samples += len(hg)
            
    avg_acc = (total_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    return total_loss / len(loader), avg_acc

def get_dataloader(batch_size=32):
    df, le_t, le_l = get_master_data()
    if df is None: return None, None
    ds = SoccerDataset(df)
    # num_workers=2: background prefetch (safe on Windows with spawn start method)
    # persistent_workers keeps worker processes alive between batches
    # pin_memory only useful on CUDA
    use_pin = torch.cuda.is_available()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=2, persistent_workers=True, pin_memory=use_pin)
    return loader, loader

# ==========================================
# 4. REINFORCEMENT LEARNING (RL) AGENT (PPO) - Placeholder for Compatibility
# ==========================================
class PPOMemory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim=4, lr=0.002, betas=(0.9, 0.999), gamma=0.99, K_epochs=4, eps_clip=0.2, entropy_coef=0.05):
        super(PPOAgent, self).__init__()
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        
        # Deeper actor: LayerNorm stabilises large-dim inputs; LeakyReLU avoids dead neurons
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1)
        )
        
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ], lr=lr, betas=betas)

        # CRITICAL: policy_old must be a *separate copy* so gradients don't flow into it.
        # Using a reference alias (self.actor) broke the PPO clipping ratio (always ≈1.0).
        self.policy_old = copy.deepcopy(self.actor)
        self.policy_old.load_state_dict(self.actor.state_dict())
        self.MseLoss = nn.MSELoss()

    def get_action(self, state, memory):
        state = state.detach()
        action_probs = self.policy_old(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        return action.item()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

    def update(self, memory, gae_lambda=0.95):
        old_states   = torch.squeeze(torch.stack(memory.states,   dim=0)).detach().to(DEVICE)
        old_actions  = torch.squeeze(torch.stack(memory.actions,  dim=0)).detach().to(DEVICE)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(DEVICE)

        # --- Generalized Advantage Estimation (GAE) ---
        # Blends 1-step TD error with Monte Carlo returns via gae_lambda.
        # Lower variance than pure MC; lower bias than pure TD.
        with torch.no_grad():
            values = torch.squeeze(self.critic(old_states))

        T = len(memory.rewards)
        advantages = torch.zeros(T, dtype=torch.float32, device=DEVICE)
        gae = 0.0
        for t in reversed(range(T)):
            is_term = memory.is_terminals[t]
            next_val = values[t + 1].item() if (t + 1 < T and not is_term) else 0.0
            delta = memory.rewards[t] + self.gamma * next_val - values[t].item()
            gae = delta + self.gamma * gae_lambda * (0.0 if is_term else gae)
            advantages[t] = gae

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        returns = (advantages + values).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, returns) - self.entropy_coef * dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.actor.state_dict())

def train_ppo_agent(model, agent, epochs=50):
    # Simplified PPO Loop
    memory = PPOMemory()
    model.eval()
    master, _, _ = get_master_data()
    if master is None: return
    train_loader = DataLoader(SoccerDataset(master, tail=2000), batch_size=1, shuffle=True)
    update_timestep = 512
    time_step = 0
    agent.train()
    print(f"Training PPO Agent on {len(train_loader)} matches for {epochs} epochs...")
    
    # DEBUG: Check Device
    first_batch = next(iter(train_loader))
    print(f"    [DEBUG] Agent Device: {next(agent.parameters()).device}")
    print(f"    [DEBUG] Input Batch Device: {DEVICE}") # We move to DEVICE in loop
    
    for epoch in range(epochs):
        print(f"  > Start Epoch {epoch+1}/{epochs}")
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % 100 == 0: print(f"    Batch {batch_idx}/{len(train_loader)}", end='\r')
            time_step += 1
            with torch.no_grad():
                # Extract State
                 b_h_hist = batch['h_seq'].to(DEVICE)
                 b_a_hist = batch['a_seq'].to(DEVICE)
                 b_h_id = batch['h_id'].to(DEVICE)
                 b_a_id = batch['a_id'].to(DEVICE)
                 b_l_id = batch['l_id'].to(DEVICE)
                 b_odds = batch['odds'].to(DEVICE)
                 b_h_elo = batch['h_elo'].to(DEVICE)
                 b_a_elo = batch['a_elo'].to(DEVICE)
                 
                 state = model.extract_features(b_h_hist, b_a_hist, b_h_id, b_a_id, b_l_id, b_h_elo, b_a_elo)
            
            # Verify tensor device on first step
            if epoch == 0 and batch_idx == 0:
                 print(f"    [DEBUG] State Tensor Device: {state.device}")

            action = agent.get_action(state, memory)
            
            # --- CALCULATE REWARD (KELLY CRITERION) ---
            hg = batch['hg'].item()
            ag = batch['ag'].item()
            
            if ag > hg: gt = 2
            elif ag == hg: gt = 1
            else: gt = 0
            
            # Re-run forward for probabilities
            with torch.no_grad():
                 lambdas, rho, _, _, _ = model(b_h_hist, b_a_hist, b_h_id, b_a_id, b_l_id, b_odds, b_h_elo, b_a_elo)
            
            h_lam = lambdas[0, 0].item()
            a_lam = lambdas[0, 1].item()
            rho_val = rho[0].item()
            
            # Use fewer sims during training (1000 vs 5000) — sufficient for reward signal, 5× faster
            model_probs = calculate_probabilities(h_lam, a_lam, rho_val, n_sims=1000)
            p_map = {0: model_probs['home_win'], 1: model_probs['draw'], 2: model_probs['away_win']}
            
            reward = 0
            
            if action == 3: # PASS
                max_kelly = -1.0
                for a_idx in [0, 1, 2]:
                    this_odds = b_odds[0, a_idx].item()
                    if this_odds > 1.0:
                        b = this_odds - 1.0
                        p = p_map[a_idx]
                        q = 1.0 - p
                        f = (b*p - q) / b
                        if f > max_kelly: max_kelly = f
                
                if max_kelly > 0.05: reward = -0.1
                else: reward = 0.05
                    
            else: # BET
                winning_odds = b_odds[0, action].item()
                if winning_odds > 1.0:
                    b = winning_odds - 1.0
                    p = p_map[action]
                    q = 1.0 - p
                    kelly_f = (b*p - q) / b
                else:
                    kelly_f = -1.0 
                
                if action == gt: 
                    raw_profit = winning_odds - 1.0
                    # CAP REWARD to prevent "Longshot Addiction"
                    # If odds are 10.0, profit is 9.0. If we cap at 3.0, agent values it less.
                    if raw_profit > 3.0: raw_profit = 3.0
                else: 
                    raw_profit = -2.0 # Stronger penalty for loss (was -1.0)
                
                if kelly_f > 0:
                    if raw_profit > 0: reward = raw_profit * 1.5
                    else: reward = -2.0 # Stronger Penalty
                else:
                    if raw_profit > 0: reward = 0.0
                    else: reward = -2.0
            
            memory.rewards.append(reward)
            memory.is_terminals.append(True)
            if time_step % update_timestep == 0:
                agent.update(memory)
                memory.clear_memory()
                time_step = 0
    return agent


# ==========================================
# 5. INFERENCE HELPERS (Restored)
# ==========================================
def get_team_history(team_id, pred_date, master_df):
    """
    Reconstructs the 10-match history sequence for a team at a given date.
    Used by app.py for real-time inference.
    """
    # 1. Filter prior matches for this team
    pred_date = pd.Timestamp(pred_date)
    
    # Identify matches where team was Home or Away
    mask = ((master_df['home_id'] == team_id) | (master_df['away_id'] == team_id)) & \
           (master_df['date_obj'] < pred_date)
           
    hist_df = master_df[mask].sort_values('date_obj').tail(SEQ_LENGTH + 5) # Get enough for averages
    
    # If not enough history, return padded
    if len(hist_df) == 0:
        return np.zeros((SEQ_LENGTH, 20), dtype=np.float32)

    # DYNAMIC SEQUENCE LENGTH TRUNCATION
    # Truncate the history DataFrame dynamically if a shorter optimal length exists
    league_str = 'Unknown'
    if not master_df.empty and 'league_name' in master_df.columns:
        # Get the league name from the match we are predicting for
        l_name_series = master_df[(master_df['home_id'] == team_id) | (master_df['away_id'] == team_id)]['league_name']
        if not l_name_series.empty:
            league_str = l_name_series.iloc[-1]
            
    opt_len = LEAGUE_SEQ_LENGTHS.get(league_str, SEQ_LENGTH)
    
    # Strictly limit the inference tail context to the optimal length
    hist_df = hist_df.tail(opt_len)

    # 2. Calculate Season Averages with TIME DECAY
    # Lookback 365 days to capture long-term form, but weight recent matches heavily
    lookback_start = pred_date - pd.Timedelta(days=365)
    seas_df = hist_df[hist_df['date_obj'] >= lookback_start].copy()
    
    seas_avg_xg = 1.3
    seas_avg_xga = 1.3
    
    if len(seas_df) > 0:
        # Calculate Decay Weights
        # weight = exp(-0.005 * days_ago)
        # 0 days ago = 1.0
        # 140 days ago (~4.5 months) = 0.5
        # 365 days ago = 0.16
        seas_df['days_ago'] = (pred_date - seas_df['date_obj']).dt.days
        seas_df['weight'] = np.exp(-0.005 * seas_df['days_ago'])
        
        s_xg = []
        s_xga = []
        weights = []
        
        for _, r in seas_df.iterrows():
            w = r['weight']
            weights.append(w)
            if r['home_id'] == team_id:
                s_xg.append(r['home_xg'] * w)
                s_xga.append(r['away_xg'] * w)
            else:
                s_xg.append(r['away_xg'] * w)
                s_xga.append(r['home_xg'] * w)
                
        total_weight = sum(weights)
        if total_weight > 0:
            seas_avg_xg = sum(s_xg) / total_weight
            seas_avg_xga = sum(s_xga) / total_weight

    # 3. Build Sequence
    matches = []
    # Convert rows to list of tuples for easy processing
    for _, r in hist_df.iterrows():
        is_home = (r['home_id'] == team_id)
        if is_home:
            gf, ga = r['home team total goal'], r['away team total goal']
            xg, xga = r['home_xg'], r['away_xg']
            stats = (r['date_obj'], gf, ga, xg, xga, 1.0)
        else:
            gf, ga = r['away team total goal'], r['home team total goal']
            xg, xga = r['away_xg'], r['home_xg']
            stats = (r['date_obj'], gf, ga, xg, xga, 0.0)
        matches.append(stats)
        
    valid_seq = matches[-SEQ_LENGTH:]
    feats = []
    
    # valid_seq maps to the tail of matches; compute the absolute index via offset
    seq_offset = len(matches) - len(valid_seq)
    for i, m in enumerate(valid_seq):
        m_date, m_gf, m_ga, m_xg, m_xga, m_is_home = m
        
        # Rest Days — use enumerate index directly (O(1)) instead of list.index() which is O(n)
        curr_idx = seq_offset + i
        if curr_idx > 0:
            prev_date = matches[curr_idx-1][0]
            rest = (m_date - prev_date).days
            rest = min(rest, 14)
        else:
            rest = 7
        rest_norm = rest / 7.0
        
        # Rolling Form (Last 5)
        start_form = max(0, curr_idx - 5)
        form_window = matches[start_form : curr_idx]
        
        # --- NEW: 6-CATEGORY xG VARIANCE INFERENCE ---
        # We need to compute historical variances for this team at this point in time
        # This mirrors `SoccerDataset` but we compute it on the fly for the inference sequence
        # We will parse all matches prior to `curr_idx` to build the stats
        cat_stats = {c: [0.0, 0] for c in range(6)} # {cat: [sum_error, count]}
        
        # Helper to get category given opponent ID
        # Since `matches` tuple doesn't store opponent ELO natively, we'll approximate 
        # based on league variance or skip dynamic ELO here and just use a recent approximation.
        # However, to be perfectly accurate with the trained model, we need Opponent ELO.
        # Since `get_team_history` is called for a single team and we don't have opponent ELO in the tuple,
        # we will approximate the 6-cat variance using the average variance across ALL their past matches
        # or we could look up the opponent ID in master_df.
        # To keep inference fast without massive dataframe lookups per sequence step:
        # We will just use the global team variance (which is heavily correlated) duplicated across the 6 features.
        # (A more precise approach would pre-compute this globally and pass it in, but this is a solid approximation for real-time inference latency).
        
        # Quick Variance Calculation (All matches up to curr_idx)
        hist_so_far = matches[:curr_idx]
        if len(hist_so_far) > 0:
            global_xg_error = sum(abs(x[1] - x[3]) for x in hist_so_far) / len(hist_so_far)
        else:
            global_xg_error = 0.5 
            
        # We populate all 6 categories with this global average for this specific sequence step.
        cat_vars = [global_xg_error] * 6
        
        if form_window:
            avg_gf = sum(x[1] for x in form_window) / len(form_window)
            avg_ga = sum(x[2] for x in form_window) / len(form_window)
            avg_xg = sum(x[3] for x in form_window) / len(form_window)
            avg_xga = sum(x[4] for x in form_window) / len(form_window)
            # Rolling xG Variance (MAE of GF vs xG)
            avg_xg_var = sum(abs(x[1] - x[3]) for x in form_window) / len(form_window)
        else:
            avg_gf, avg_ga, avg_xg, avg_xga = 1.3, 1.3, 1.3, 1.3
            avg_xg_var = 0.0
            
        # League Variance - Placeholder
        l_var = 0.5 
        
        feat = [
            m_gf, m_ga, m_xg, m_xga, m_is_home, rest_norm,
            avg_gf/3.0, avg_ga/3.0, avg_xg/3.0, avg_xga/3.0,
            seas_avg_xg/3.0, seas_avg_xga/3.0,
            l_var/10.0,
            0.0 # Placeholder
        ]
        
        # Append 6 context variances
        feat.extend([v/3.0 for v in cat_vars])
        feats.append(feat)
        
    feats_np = np.array(feats, dtype=np.float32)
    
    # Pad if short
    if len(feats_np) < SEQ_LENGTH:
        pad_len = SEQ_LENGTH - len(feats_np)
        padding = np.zeros((pad_len, 20), dtype=np.float32)
        if len(feats_np) > 0: feats_np = np.vstack([padding, feats_np])
        else: feats_np = padding
        
    return feats_np

def generate_score_grid(lambda_h, lambda_a, rho=0.0, max_goals=10):
    """
    Generates a DC-corrected 2D probability grid for scores (0-0 to 9-9).
    The rho parameter applies the Dixon-Coles low-score adjustment so that
    this grid is consistent with calculate_probabilities().
    """
    h_probs = np.array([poisson.pmf(i, lambda_h) for i in range(max_goals)])
    a_probs = np.array([poisson.pmf(i, lambda_a) for i in range(max_goals)])
    grid = np.outer(h_probs, a_probs)

    # Dixon-Coles correction on low-score cells
    if abs(rho) > 1e-6:
        rho = float(np.clip(rho, -0.99, 0.99))
        grid[0, 0] *= max(1e-8, 1.0 - lambda_h * lambda_a * rho)
        grid[0, 1] *= max(1e-8, 1.0 + lambda_h * rho)
        grid[1, 0] *= max(1e-8, 1.0 + lambda_a * rho)
        grid[1, 1] *= max(1e-8, 1.0 - rho)
        # Re-normalise so probabilities still sum to 1
        total = grid.sum()
        if total > 0:
            grid /= total

    return grid