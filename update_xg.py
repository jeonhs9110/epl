import pandas as pd
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import math

# Import scraper logic
# We assume this script is in the same directory as scrape_flashscore.py
import scrape_flashscore

def update_xg_for_file(filename):
    print(f"\nProcessing {filename}...")
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return

    # Check if xG columns exist
    if 'home_xg' not in df.columns:
        df['home_xg'] = 0.0
    if 'away_xg' not in df.columns:
        df['away_xg'] = 0.0
        
    # Identify matches to update
    # We want to update matches where xG is 0.0 AND match_id exists
    # (Some older matches might genuinely be 0.0 but it's rare to be exactly 0.0 for both if data exists, usually > 0.01)
    
    # Filter for valid IDs
    if 'match_id' not in df.columns:
        print(f"Skipping {filename}: No match_id column.")
        return

    # Create a mask for rows to update
    # We update if home_xg == 0 and away_xg == 0. 
    # NOTE: This might re-scrape matches that truly have no xG (old matches). 
    # To avoid infinite loop of re-scraping old matches, we could check date? 
    # For now, let's just try to update all 0s.
    mask = (df['home_xg'] == 0.0) & (df['away_xg'] == 0.0) & (df['match_id'].str.startswith('g_1_'))
    
    to_update_df = df[mask]
    
    if to_update_df.empty:
        print(f"No matches need xG update in {filename}.")
        return

    print(f"Found {len(to_update_df)} matches to update in {filename}.")
    
    # Convert to list of dicts for the worker
    matches_to_scrape = to_update_df.to_dict('records')
    
    # Run Scraper (Re-using process_match_batch)
    # This will fetch Odds AND xG
    updated_matches = []
    
    # Use parallel workers
    NUM_WORKERS = 4
    chunk_size = math.ceil(len(matches_to_scrape) / NUM_WORKERS)
    if chunk_size < 1: chunk_size = 1
    
    chunks = [matches_to_scrape[i:i + chunk_size] for i in range(0, len(matches_to_scrape), chunk_size)]
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(scrape_flashscore.process_match_batch, chunk, i+1) for i, chunk in enumerate(chunks)]
        for f in futures:
            updated_matches.extend(f.result())
            
    # Update DataFrame
    # We create a lookup dict
    updates_map = {m['match_id']: m for m in updated_matches}
    
    updates_count = 0
    for idx, row in df.iterrows():
        mid = row.get('match_id')
        if mid in updates_map:
            new_data = updates_map[mid]
            # Update xG
            df.at[idx, 'home_xg'] = new_data.get('home_xg', 0.0)
            df.at[idx, 'away_xg'] = new_data.get('away_xg', 0.0)
            # Optionally update odds too since we have them
            df.at[idx, 'odds_1'] = new_data.get('odds_1', row['odds_1'])
            df.at[idx, 'odds_x'] = new_data.get('odds_x', row['odds_x'])
            df.at[idx, 'odds_2'] = new_data.get('odds_2', row['odds_2'])
            
            if new_data.get('home_xg', 0.0) > 0:
                updates_count += 1

    print(f"Updated xG for {updates_count} matches in {filename}.")
    
    # Save
    df.to_csv(filename, index=False)
    print(f"Saved {filename}.")

def main():
    print("="*60)
    print("xG BACKFILL UTILITY")
    print("="*60)
    
    files = glob.glob('*_RESULTS.csv')
    print(f"Found {len(files)} match files.")
    
    for f in files:
        update_xg_for_file(f)
        
    print("\nBatch Update Complete.")

if __name__ == "__main__":
    main()
