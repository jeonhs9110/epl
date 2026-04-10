import pandas as pd
import os
import sys

# Add parent directory to path to allow importing scrape_flashscore
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scrape_flashscore import process_match_batch
import math
from concurrent.futures import ThreadPoolExecutor

def fix_eredivisie_odds():
    # Define file path relative to this script (assuming script is in 'old_matches' or run from there)
    # The user said the file is: C:\Users\201-05\Desktop\SPORTS\EPL\old_matches\FOOTBALL_EREDIVISIE_2024_2025_RESULTS.csv
    
    # We will assume this script is placed in 'old_matches' or we construct absolute path
    # To be safe, let's use the explicit path if possible, or relative
    
    target_file = r"C:\Users\201-05\Desktop\SPORTS\EPL\old_matches\FOOTBALL_EREDIVISIE_2024_2025_RESULTS.csv"
    
    if not os.path.exists(target_file):
        print(f"File not found: {target_file}")
        return

    print(f"Loading {target_file}...")
    df = pd.read_csv(target_file)
    
    # Check for odds columns
    if 'odds_1' not in df.columns:
        df['odds_1'] = 0.0
        df['odds_x'] = 0.0
        df['odds_2'] = 0.0
        
    # Identify rows with missing odds
    # We assume '0.0' or 0.0 indicates missing
    # Pandas might read as float or object
    
    def is_missing(row):
        try:
            o1 = float(row['odds_1'])
            ox = float(row['odds_x'])
            o2 = float(row['odds_2'])
            return o1 == 0.0 and ox == 0.0 and o2 == 0.0
        except:
            return True

    missing_indices = []
    matches_to_scrape = []
    
    print("Identifying matches with missing odds...")
    for idx, row in df.iterrows():
        if is_missing(row):
            # We need match_id to scrape
            # The CSV might NOT have match_id if it was dropped.
            # However, looking at the previous file view, 'match_id' column was NOT present in the final CSV view 
            # (columns: sports,league,round,date,home team,away team,home team total goal,away team total goal,result)
            # WAIT. If match_id is gone, we CANNOT easily re-scrape unless we search by team name?
            # scrape_flashscore.py drops match_id before saving?
            # Let's check scrape_flashscore.py line 368: if 'match_id' in df.columns: df = df.drop...
            # The user's CSV view indeed shows NO match_id.
            
            # CRITICAL: Without match_id, we can't deep-link to the match page.
            # We would have to re-scrape the main list to get IDs again.
            pass
            
            # BUT! The user wants to fix this file. 
            # If we don't have IDs, we need to map them back.
            # Does the file have them? The view showed headers: sports,league,round,date,home team... result.
            # No match_id.
            
            # SOLUTION: We must re-scrape the "Eredivisie 2024/2025" archive page to match teams/dates to IDs.
            # Then scrape odds for those IDs.
            pass

    # Since we can't simple iterate the CSV, we need to:
    # 1. Scrape the Matches List from the Archive URL (Eredivisie 2024-2025)
    # 2. Match the scraped IDs to our CSV rows (by Date + Home + Away)
    # 3. For the matched rows that have missing odds, scrape the odds.
    # 4. Update the dataframe.
    
    # URL for Eredivisie 2024/2025
    # https://www.flashscore.com/football/netherlands/eredivisie-2024-2025/results/
    # Or generically constructed.
    
    archive_url = "https://www.flashscore.com/football/netherlands/eredivisie-2024-2025/results/"
    
    print(f"Match IDs missing in CSV. Re-scraping match list from: {archive_url}")
    
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from bs4 import BeautifulSoup
    import time
    import re
    from datetime import datetime
    from selenium.webdriver.common.action_chains import ActionChains
    import random

    # 1. Scrape List to get IDs
    options = webdriver.ChromeOptions()
    options.add_argument('--headless=new') 
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    scraped_matches_map = {} # Key: (Date, Home, Away) -> match_id
    
    try:
        driver.get(archive_url)
        time.sleep(3)
        
        # Handle consent
        try:
             accept_btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler")))
             accept_btn.click()
             time.sleep(1)
        except: pass
        
        # Load all matches
        while True:
            try:
                more_btn = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Show more matches')]")))
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", more_btn)
                time.sleep(1)
                ActionChains(driver).move_to_element(more_btn).click().perform()
                time.sleep(3)
            except:
                break
                
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        events = soup.find_all('div', class_=re.compile('event__'))
        
        for event in events:
            mid = event.get('id', '')
            if mid.startswith('g_1_'):
                try:
                    time_div = event.find('div', class_='event__time')
                    date_str = time_div.get_text(strip=True).replace('.', '/').split(' ')[0]
                    
                    # Fix Year (Assume 2024/2025 season logic)
                    # Simple heuristic: matches in 2024 or 2025
                    # Flashscore list doesn't show year usually.
                    # We need to reconstruct year.
                    # Aug-Dec -> 2024, Jan-May -> 2025
                    parts = date_str.split('/') # dd/mm
                    if len(parts) == 2:
                        d, m = int(parts[0]), int(parts[1])
                        y = 2024 if m >= 7 else 2025
                        date_str = f"{parts[0]}/{parts[1]}/{y}"
                    
                    home_team = event.select_one('.event__homeParticipant').get_text(strip=True)
                    away_team = event.select_one('.event__awayParticipant').get_text(strip=True)
                    
                    key = (date_str, home_team, away_team)
                    scraped_matches_map[key] = mid
                except:
                    pass
                    
    finally:
        driver.quit()
        
    print(f"Mapped {len(scraped_matches_map)} match IDs from source.")
    
    # 2. Match CSV rows to IDs
    rows_to_update = []
    
    for idx, row in df.iterrows():
        if is_missing(row):
            # Create key
            # CSV Date format: dd/mm/ (missing year?)
            # The CSV shown has '02/06/', '30/05/' ... Wait, it's missing the year in the VIEW?
            # Line 2: "02/06/"
            # Line 303: "25/08/"
            # We need to fix the year matching. 
            # Our scraper logic added years. The CSV might have dropped it or formatted weirdly.
            # Let's try partial match or just dd/mm
            
            c_date = str(row['date']).strip()
            # If CSV date is '02/06/', we need to handle that.
            
            c_home = row['home team']
            c_away = row['away team']
            
            # Find ID
            # We try to match loosely on date
            found_mid = None
            
            # Try exact match first
            # The CSV dates in previous step view were "02/06/" etc. 
            # The scraper logic was: date_str = date_str.replace('.', '/').split(' ')[0] ... then date_str += f"/{y}"
            # Maybe the CSV saved it without year?
            
            # Let's search in scraped_matches_map
            for (s_date, s_home, s_away), mid in scraped_matches_map.items():
                # Compare teams
                if s_home == c_home and s_away == c_away:
                     # Compare date (ignore year if CSV missing it)
                     if c_date in s_date or s_date.startswith(c_date):
                         found_mid = mid
                         break
            
            if found_mid:
                rows_to_update.append({
                    "index": idx,
                    "match_id": found_mid,
                    "home team": c_home,
                    "away team": c_away
                })
            else:
                print(f"Could not find ID for {c_home} vs {c_away} ({c_date})")

    if not rows_to_update:
        print("No matches found to update or map.")
        return

    print(f"Found {len(rows_to_update)} matches to repair. starting batch scrape...")
    
    # 3. Scrape Odds
    # Use ThreadPool logic from scraper
    # But stripped down
    
    # We can reuse process_match_batch if we construct the input correctly
    # Input expected: list of dicts with 'match_id', 'home team', 'away team'
    
    batch_input = [r for r in rows_to_update] # list of dicts
    
    # Run in batches of 6 (Power Mode style)
    NUM_WORKERS = 4
    chunk_size = math.ceil(len(batch_input) / NUM_WORKERS)
    chunks = [batch_input[i:i + chunk_size] for i in range(0, len(batch_input), chunk_size)]
    
    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_match_batch, chunk, i+1) for i, chunk in enumerate(chunks)]
        for f in futures:
            results.extend(f.result())
            
    # 4. Update DataFrame
    updated_count = 0
    for res in results:
        idx = res['index']
        o1 = res.get('odds_1', 0.0)
        ox = res.get('odds_x', 0.0)
        o2 = res.get('odds_2', 0.0)
        
        if o1 != 0.0:
            df.at[idx, 'odds_1'] = o1
            df.at[idx, 'odds_x'] = ox
            df.at[idx, 'odds_2'] = o2
            updated_count += 1
            
    print(f"Updated {updated_count} rows.")
    
    # 5. Save
    df.to_csv(target_file, index=False)
    print("File saved successfully.")

if __name__ == "__main__":
    fix_eredivisie_odds()
