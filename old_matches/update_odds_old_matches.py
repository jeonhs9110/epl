
import os
import sys
import re
import time

# Add parent directory to path to allow importing scrape_flashscore
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scrape_flashscore import scrape_flashscore_final
from league_urls import LEAGUE_URLS

def get_league_url_key(filename_base):
    """
    Map filename parts (e.g. PREMIER_LEAGUE) to keys in LEAGUE_URLS via simple heuristics.
    """
    normalized_name = filename_base.upper()
    
    # Manual Mapping for known structures
    mapping = {
        "PREMIER_LEAGUE": "Premier League (England)",
        "CHAMPIONSHIP": "Championship (England)",
        "LALIGA": "La Liga (Spain)",
        "LALIGA2": "La Liga 2 (Spain)",
        "BUNDESLIGA": "Bundesliga (Germany)",
        "2_BUNDESLIGA": "2. Bundesliga (Germany)",
        "SERIE_A": "Serie A (Italy)",
        "SERIE_B": "Serie B (Italy)",
        "LIGUE_1": "Ligue 1 (France)",
        "LIGUE_2": "Ligue 2 (France)",
        "EREDIVISIE": "Eredivisie (Netherlands)",
        "CHAMPIONS_LEAGUE": "Champions League (Europe)",
        "EUROPA_LEAGUE": "Europa League (Europe)"
    }
    
    return mapping.get(normalized_name)

def update_old_matches():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files = [f for f in os.listdir(current_dir) if f.endswith("_RESULTS.csv") and "20" in f]
    
    print(f"Found {len(files)} season files to update.")
    
    for filename in files:
        print(f"\nProcessing {filename}...")
        
        # Regex to extract League Part and Season Part
        # Format: FOOTBALL_{LEAGUE_NAME}_{YEAR}_{YEAR}_RESULTS.csv
        # Example: FOOTBALL_PREMIER_LEAGUE_2024_2025_RESULTS.csv
        
        # We assume the pattern ends with _YYYY_YYYY_RESULTS.csv
        match = re.search(r"FOOTBALL_(.+)_(\d{4}_\d{4})_RESULTS\.csv", filename)
        if not match:
            print(f"Skipping {filename} - Pattern not matched.")
            continue
            
        league_part = match.group(1)
        season_part = match.group(2) # e.g. 2024_2025
        
        season_str = season_part.replace('_', '-') # 2024-2025
        
        print(f"   -> Detected League: {league_part}")
        print(f"   -> Detected Season: {season_str}")
        
        # 1. Get Base URL
        url_key = get_league_url_key(league_part)
        if not url_key or url_key not in LEAGUE_URLS:
            print(f"   -> ERROR: Could not map {league_part} to a known URL in league_urls.py")
            continue
            
        base_url = LEAGUE_URLS[url_key]
        # base_url example: https://www.flashscore.com/football/england/premier-league/results/
        
        # 2. Construct Archive URL
        # We need to insert the season string into the league slug
        # Strategy: Split by '/' and modify the 5th element (index 5) if standard structure
        # Standard: https: / / www.flashscore.com / football / country / league / results /
        #             0    1          2                3          4         5        6
        
        parts = base_url.split('/')
        if len(parts) >= 7 and parts[6] == 'results':
            league_slug = parts[5]
            # Construct new slug: premier-league-2024-2025
            new_slug = f"{league_slug}-{season_str}"
            parts[5] = new_slug
            archive_url = "/".join(parts)
            
            print(f"   -> Generated URL: {archive_url}")
            
            # 3. Run Scraper
            try:
                # We need to temporarily change CWD so scrape_flashscore saves CSV to 'old_matches'
                # Actually scrape_flashscore saves to CWD. ensure we are in old_matches
                original_cwd = os.getcwd()
                os.chdir(current_dir)
                
                scrape_flashscore_final(archive_url, force_full=True)
                
                os.chdir(original_cwd)
                print(f"   -> Successfully updated {filename}")
                
            except Exception as e:
                print(f"   -> Failed to scrape {filename}: {e}")
                os.chdir(original_cwd) # Restoration
        else:
            print(f"   -> ERROR: Base URL structure unexpected: {base_url}")

if __name__ == "__main__":
    update_old_matches()
