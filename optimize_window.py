import pandas as pd
import numpy as np
import glob
import os
import re

# Load Master Data Logic (Reused)
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

def get_master_data():
    all_files = glob.glob('*_RESULTS.csv') + glob.glob('old_matches/*_RESULTS.csv')
    if not all_files: return None
    
    df_list = []
    
    # We need a rough year guess, default 2024
    current_season_start = 2024 
    
    for f in all_files:
        try:
            temp_df = pd.read_csv(f)
            temp_df.columns = temp_df.columns.str.strip().str.lower()
            
            # Simple Fix for Year
            file_year = current_season_start
            match = re.search(r'(\d{4})_\d{4}', f)
            if match: file_year = int(match.group(1))
            
            valid_dates = []
            for idx, row in temp_df.iterrows():
                dt = parse_football_date(row.get('date', ''), file_year)
                valid_dates.append(dt)
            
            temp_df['date_obj'] = valid_dates
            temp_df = temp_df.dropna(subset=['date_obj'])
            df_list.append(temp_df)
        except: pass
        
    if not df_list: return None
    return pd.concat(df_list, ignore_index=True).sort_values('date_obj')

def optimize():
    print("Loading Data...")
    df = get_master_data()
    if df is None:
        print("No data found.")
        return

    print(f"Dataset size: {len(df)} matches")
    
    # We want to test different 'N' (Window Sizes)
    # Logic: Calculate 'Form Score' for Home and Away using last N matches.
    # Predict: Team with higher Form Score wins.
    # Metric: Accuracy.
    
    candidates = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]
    results = {}
    
    teams = set(df['home team'].unique()) | set(df['away team'].unique())
    
    # To be efficient, we can pre-calculate points for all matches
    # But doing it chronologically is safer to simulate "real-time" knowledge
    
    # Let's run a faster approximation
    # For each N, we iterate through the dataframe
    
    for n in candidates:
        print(f"Testing Window Size: {n}...")
        correct = 0
        total = 0
        
        # Team History Tracking
        # {team: [list of results W/D/L scores]}
        # W=3, D=1, L=0
        team_form = {t: [] for t in teams}
        
        for _, row in df.iterrows():
            h = row['home team']
            a = row['away team']
            
            # 1. Predict based on CURRENT history (before this match)
            h_hist = team_form.get(h, [])[-n:]
            a_hist = team_form.get(a, [])[-n:]
            
            if len(h_hist) >= 3 and len(a_hist) >= 3: # Only predict if we have some data
                h_score = sum(h_hist) / len(h_hist) # Average points per game in window
                a_score = sum(a_hist) / len(a_hist)
                
                # Actual Result
                hg = row['home team total goal']
                ag = row['away team total goal']
                if hg > ag: actual = 'H'
                elif ag > hg: actual = 'A'
                else: actual = 'D'
                
                # Prediction
                pred = 'D'
                if h_score > a_score + 0.2: pred = 'H' # Threshold for home/away bias?
                elif a_score > h_score + 0.2: pred = 'A'
                
                if pred == actual:
                    correct += 1
                
                total += 1
            
            # 2. Update History
            hg = row['home team total goal']
            ag = row['away team total goal']
            
            if hg > ag: 
                team_form[h].append(3)
                team_form[a].append(0)
            elif ag > hg:
                team_form[h].append(0)
                team_form[a].append(3)
            else:
                team_form[h].append(1)
                team_form[a].append(1)
        
        acc = (correct / total * 100) if total > 0 else 0
        results[n] = acc
        print(f"  -> Accuracy: {acc:.2f}% (Samples: {total})")
        
    # Find Best
    best_n = max(results, key=results.get)
    print(f"\n🏆 OPTIMAL WINDOW SIZE: {best_n} Matches (Accuracy: {results[best_n]:.2f}%)")
    print("This number represents the 'Recency Bias' that works best for this dataset.")

if __name__ == "__main__":
    optimize()
