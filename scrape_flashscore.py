
import pandas as pd
import re
import time
import math
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import random
from concurrent.futures import ThreadPoolExecutor

# Logic to extract odds from a specific match Page Source
def extract_odds_from_source(page_source):
    soup = BeautifulSoup(page_source, 'html.parser')
    odds_vals = [0.0, 0.0, 0.0]
    
    # Strategy 1: Find 1xBet specific row
    # The user screenshot shows <a title="1xBet"> inside the row
    # We look for the bookmaker identifier, then traverse up to the row container
    
    bookmaker_elem = soup.find(attrs={"title": "1xBet"})
    if not bookmaker_elem:
        # Try finding by img alt
        bookmaker_elem = soup.find('img', attrs={"alt": "1xBet"})
        
    target_row = None
    if bookmaker_elem:
        # Traverse parents until we find a container that has 3 odds values
        # or is a 'ui-table__row' or generic row wrapper
        curr = bookmaker_elem.parent
        for _ in range(10): # Depth limit
            if not curr: break
            # Check if this container has odds values
            # New class structure uses 'wcl-oddsValue'
            vals = curr.find_all(class_=re.compile("oddsValue"))
            if len(vals) >= 3:
                target_row = curr
                break
            curr = curr.parent
            
    # Strategy 2: Fallback to first row with enough odds if 1xBet not found
    if not target_row:
        # Find any element with class containing 'oddsValue'
        all_vals = soup.find_all(class_=re.compile("oddsValue"))
        if len(all_vals) >= 3:
            # Assume the first 3 are the primary market (1x2) for the first bookmaker
            # This is risky but better than 0.0
            html_content = "".join([str(v) for v in all_vals[:3]])
            # Mock a row object/list
            # We will just parse these directly here
            try:
                for i in range(3):
                    val_txt = all_vals[i].get_text(strip=True)
                    if val_txt and val_txt != '-':
                        odds_vals[i] = float(val_txt)
                return odds_vals
            except:
                pass

    if target_row:
        o_elems = target_row.find_all(class_=re.compile("oddsValue"))
        
        if len(o_elems) >= 3:
            for i in range(3):
                val_txt = o_elems[i].get_text(strip=True)
                if val_txt and val_txt != '-':
                    odds_vals[i] = float(val_txt)
                        
    return odds_vals

    # Logic to extract xG from match statistics Page Source
def extract_xg_from_source(page_source):
    soup = BeautifulSoup(page_source, 'html.parser')
    h_xg = 0.0
    a_xg = 0.0
    
    try:
        # Robust Method: Find "Expected Goals" text, then find 2 float numbers in the parent container.
        # This handles cases where class names differ (stat__row vs wcl-row vs widget-top-stats).
        
        # 1. Find the text node "Expected Goals" (case insensitive)
        target = soup.find(string=re.compile("Expected Goals", re.IGNORECASE))
        
        if target:
            # Traverse up the DOM to find the container row
            curr = target.parent
            found = False
            for _ in range(8): # Check up to 8 parents
                if not curr: break
                
                # Get all text in this container
                row_text = curr.get_text(" ", strip=True)
                
                # Look for patterns like "1.23 Expected Goals (xG) 0.88"
                # xG is almost always displayed with decimals (e.g. 1.23). 
                # Sometimes integers (1.00) are displayed as "1" or "1.00".
                # Regex to find numbers that LOOK like xG (float with dot, or just near the text)
                
                # We specifically look for floats with decimals like \d+\.\d+ to avoid capturing the string "Expected Goals" as a number if it had one.
                # But actually, finding ANY number sequence (float or int) in order might be safer, 
                # assuming the layout is [HomeVal] [Label] [AwayVal] or [Label] [HomeVal] [AwayVal]
                
                # Common layouts:
                # 1.23 Expected Goals (xG) 0.85
                # Expected Goals (xG)
                # 1.23 0.85
                
                # Let's extract all numbers from the text
                # We use a regex that captures floats (1.23) or ints (1)
                numbers = re.findall(r"(\d+\.\d+|\d+)", row_text)
                
                # Filter out numbers that might be part of the label (like '2' in '2nd Half'?) - unused here
                # Expected Goals line usually only has the two xG values.
                
                # However, we must be careful not to pick up unrelated numbers if we went too high up.
                # Current container should ideally be just the row.
                
                # If we found 2+ numbers in the row containing "Expected Goals", we use them.
                if len(numbers) >= 2:
                    # We assume the First number is Home, Last number is Away (or First/Second)
                    # Use the float values.
                    
                    # Heuristic: xG is usually < 5.0 (rarely higher). Possesion is 50. Shots 10+. 
                    # If we accidentally picked up possession, we'd see 50, 50.
                    # Expected Goals is unique in being low floats.
                    
                    v1 = float(numbers[0])
                    v2 = float(numbers[-1]) # Use last in case there are middle artifacts? usually just 2 numbers.
                    
                    # Refined selection: take the two numbers distinct from the label.
                    # Let's simple take first and second found numbers.
                    h_xg = float(numbers[0])
                    a_xg = float(numbers[1])
                    if len(numbers) > 2:
                        # If more than 2, implies we might have picked up something else? 
                        # Usually stats row is clean.
                        pass
                        
                    found = True
                    break
                    
                curr = curr.parent
            
            if not found:
                # Fallback to class search if regex text failed
                pass
                
    except Exception as e:
        pass
        
    return h_xg, a_xg

def handle_consent_popup(driver):
    try:
        # User requested explicit wait for popup to load
        time.sleep(3) 
        
        # Wait for the "I Accept" button
        accept_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
        )
        accept_btn.click()
        print("   -> Clicked 'I Accept' cookie consent.")
        time.sleep(1) # Wait for it to disappear
    except:
        # It might not appear, or already accepted
        pass

def process_match_batch(matches_batch, batch_id):
    """
    Worker function to process a batch of matches.
    matches_batch: list of dicts with 'id', 'home', 'away', etc.
    """
    if not matches_batch:
        return []

    print(f"   [Worker {batch_id}] Starting batch of {len(matches_batch)} matches...")
    
    # Setup Headless Driver for Worker
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless=new')  # DISABLED: Headless is blocked by Flashscore
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    # Anti-detection
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    processed_matches = []
    
    try:
        total = len(matches_batch)
        for idx, match in enumerate(matches_batch):
            match_id = match.get('match_id')
            if not match_id:
                processed_matches.append(match)
                continue
                
            # Construct URLs
            clean_id = match_id.replace('g_1_', '')
            main_url = f"https://www.flashscore.com/match/{clean_id}/#/match-summary"
            
            try:
                driver.get(main_url)
                # Handle consent on first load (or every time to be safe)
                if idx == 0:
                    handle_consent_popup(driver)
                
                time.sleep(2) # Wait for main load
                
                # --- 1. ODDS SCRAPING ---
                try:
                    odds_url = f"https://www.flashscore.com/match/{clean_id}/#/odds-comparison/1x2-odds/full-time"
                    driver.get(odds_url)
                    
                    # Wait for odds table
                    WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "ui-table__row"))
                    )
                    odds = extract_odds_from_source(driver.page_source)
                except:
                    # Retry once
                    time.sleep(1)
                    driver.get(odds_url)
                    time.sleep(2)
                    odds = extract_odds_from_source(driver.page_source)

                match['odds_1'] = odds[0]
                match['odds_x'] = odds[1]
                match['odds_2'] = odds[2]
                
                # --- 2. STATISTICS (xG) SCRAPING ---
                # URL structure: .../match-statistics/0 (0 usually means 'Match' or 'Summary')
                try:
                    stats_url = f"https://www.flashscore.com/match/{clean_id}/#/match-summary/match-statistics/0"
                    driver.get(stats_url)
                    
                    try:
                        # Scroll to trigger load
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
                        time.sleep(1)
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                        
                        # Wait for "Ball possession" or "Expected Goals" to appear in text
                        # Flashscore classes are obfuscated (e.g. wcl-row...), so we avoid class names.
                        WebDriverWait(driver, 5).until(
                            lambda d: "Ball possession" in d.page_source or "Expected Goals" in d.page_source or "Expected goals" in d.page_source
                        )
                        
                        time.sleep(1) # Extra buffer for render
                        h_xg, a_xg = extract_xg_from_source(driver.page_source)
                    except:
                        # Stats might not exist
                        h_xg, a_xg = 0.0, 0.0
                        
                except Exception as e_stats:
                    h_xg, a_xg = 0.0, 0.0
                
                match['home_xg'] = h_xg
                match['away_xg'] = a_xg
                
                print(f"   [Worker {batch_id}] {idx+1}/{total}: {match['home team']} vs {match['away team']} -> Odds:{odds} xG:[{h_xg}-{a_xg}]")
                
            except Exception as e:
                print(f"   [Worker {batch_id}] Error scraping {match['home team']}: {e}")
                match['odds_1'] = 0.0
                match['odds_x'] = 0.0
                match['odds_2'] = 0.0
                match['home_xg'] = 0.0
                match['away_xg'] = 0.0
                
            processed_matches.append(match)
            
    finally:
        driver.quit()
        print(f"   [Worker {batch_id}] Finished.")
        
    return processed_matches

def scrape_flashscore_final(url, force_full=False):
    # 1. Setup Main Browser to get List
    options = webdriver.ChromeOptions()
    options.add_experimental_option("detach", True)
    
    # Anti-detection for main window too
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    # Remove navigator.webdriver flag
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
            })
        """
    })

    print(f"Opening main page {url}...")
    
    # Retry logic for page load to handle connection resets
    load_success = False
    for attempt in range(3):
        try:
            driver.get(url)
            load_success = True
            break
        except Exception as e:
            print(f"   [Warning] Connection failed (Attempt {attempt+1}/3): {e}")
            time.sleep(5)
            
    if not load_success:
        print(f"   [Error] Failed to load {url} after 3 attempts. Skipping.")
        driver.quit()
        return

    # Handle consent on main page
    handle_consent_popup(driver)

    # 1.1 Determine Target CSV and Load Existing IDs
    parts = url.split('/')
    sport_name = parts[3]
    league_name = parts[5]
    filename = f"{sport_name.upper()}_{league_name.upper().replace('-', '_')}_RESULTS.csv"
    
    existing_ids = set()
    completed_ids = set() # IDs that have valid xG data
    try:
        existing_df = pd.read_csv(filename)
        if 'match_id' in existing_df.columns:
            existing_ids = set(existing_df['match_id'].dropna().astype(str).tolist())
            
            # Check for completed xG
            # We consider it complete if either home_xg or away_xg is > 0
            # Some matches might legitimately be 0-0 xG but rare. 
            # Or we can check if columns exist.
            if 'home_xg' in existing_df.columns:
                # Filter rows where xG > 0
                 completed_df = existing_df[(existing_df['home_xg'] > 0) | (existing_df['away_xg'] > 0)]
                 completed_ids = set(completed_df['match_id'].dropna().astype(str).tolist())
                 
        print(f"Loaded {len(existing_ids)} existing match IDs. {len(completed_ids)} have valid xG data.")
        print(f"Incremental mode active: Will re-scrape {len(existing_ids) - len(completed_ids)} matches with missing xG.")
    except FileNotFoundError:
        print(f"No existing file {filename}. Performing full scrape.")

    # ==============================================================================
    # AUTOMATED "SHOW MORE MATCHES"
    # ==============================================================================
    print("\n" + "=" * 60)
    print("AUTOMATION: Clicking 'Show more matches'...")
    print("=" * 60 + "\n")

    # Scroll to bottom first to ensure everything renders
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    retry_count = 0
    max_retries = 3

    while True:
        try:
            # INCREMENTAL CHECK
            if existing_ids and not force_full:
                 # Check if we have visible matches that are in our existing set
                 all_match_elements = driver.find_elements(By.CSS_SELECTOR, "[id^='g_1_']")
                 matches_found_count = 0
                 # Check the last 20 for overlap
                 for el in all_match_elements[-20:]:
                     mid = el.get_attribute('id')
                     # Stop if we hit a match that is in our COMPLETED set
                     if mid in completed_ids:
                         matches_found_count += 1
                 
                 if matches_found_count >= 1:
                     print(f"   -> Found {matches_found_count} existing matches. Stopping incremental load.")
                     break

            # Find 'Show more matches' button
            more_btn = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Show more matches')]")))
            
            # Scroll it into view
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", more_btn)
            time.sleep(1)
            
            # Click it
            ActionChains(driver).move_to_element(more_btn).click().perform()
            print("   -> Clicked 'Show more matches'. Waiting...")
            time.sleep(3) # Wait for load
            
            # Reset retries if click successful
            retry_count = 0
            
        except Exception as e:
            # If button not found, maybe we are at the end, or need to scroll down?
            print(f"   -> 'Show more' not found or not clickable. Retry {retry_count+1}/{max_retries}")
            
            # Scroll to bottom again to be sure
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            retry_count += 1
            if retry_count >= max_retries:
                print("   -> No more matches to load (Limit reached).")
                break
    
    # 2. Capture Basic Data
    print("Scraping match list from main page...")
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit() # Close main driver early
    
    # (Filename logic moved up)

    
    # Locate all event rows
    events = soup.find_all('div', class_=re.compile('event__'))
    
    basic_matches = []
    current_round = "Regular Season"
    
    print(f"Found {len(events)} event rows. Parsing basic info...")
    
    for event in events:
        classes = event.get('class', [])
        
        # Checking Round
        if 'event__round' in classes:
            text = event.get_text(strip=True)
            if "Round" in text:
                current_round = text
            continue
            
        # Checking Match
        match_id = event.get('id', '')
        if match_id.startswith('g_1_'):
            # INCREMENTAL CHECK: Skip ONLY if we already have this match AND it is complete
            if match_id in completed_ids and not force_full:
                continue

            try:
                # Extract basic info
                time_div = event.find('div', class_='event__time')
                date_str = time_div.get_text(strip=True) if time_div else "N/A"
                date_str = date_str.replace('.', '/').split(' ')[0]
                
                # Year logic
                if date_str.count('/') == 1:
                    today = datetime.now()
                    season_start_year = today.year if today.month >= 8 else today.year - 1
                    parts_d = date_str.split('/')
                    m = int(parts_d[1])
                    y = season_start_year if m >= 8 else season_start_year + 1
                    date_str += f"/{y}"
                
                home_team = event.select_one('.event__homeParticipant').get_text(strip=True)
                away_team = event.select_one('.event__awayParticipant').get_text(strip=True)
                
                # Clean Flashscore Artifacts
                home_team = home_team.replace("Advancing to next round", "").strip()
                away_team = away_team.replace("Advancing to next round", "").strip()
                home_team = re.sub(r'\s+[1-9]$', '', home_team).strip()
                away_team = re.sub(r'\s+[1-9]$', '', away_team).strip()
                home_team = re.sub(r'\s*\(\d+\)$', '', home_team).strip()
                away_team = re.sub(r'\s*\(\d+\)$', '', away_team).strip()
                
                home_score_elem = event.select_one('.event__score--home')
                away_score_elem = event.select_one('.event__score--away')
                
                if not home_score_elem or not away_score_elem:
                    continue
                    
                hg = int(home_score_elem.get_text(strip=True))
                ag = int(away_score_elem.get_text(strip=True))
                
                # Result
                if hg > ag: res = 0
                elif hg == ag: res = 1
                else: res = 2
                
                basic_matches.append({
                    "sports": sport_name,
                    "league": league_name.replace('-', ' ').title(),
                    "round": current_round,
                    "date": date_str,
                    "home team": home_team,
                    "away team": away_team,
                    "home team total goal": hg,
                    "away team total goal": ag,
                    "result": res,
                    "match_id": match_id
                })
                
            except Exception as e:
                pass

    total_matches = len(basic_matches)
    print(f"Identified {total_matches} valid matches with scores. Starting parallel scrape...")
    
    # 3. Parallel Execution
    final_data = []

    if total_matches > 0:
        # Determine workers (Increased to 6 as requested)
        NUM_WORKERS = 6
        chunk_size = math.ceil(total_matches / NUM_WORKERS)
        # Ensure chunk_size is at least 1 just in case, though ceil(>0) should be >=1
        if chunk_size < 1: chunk_size = 1
        
        chunks = [basic_matches[i:i + chunk_size] for i in range(0, total_matches, chunk_size)]
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Submit all batches
            futures = [executor.submit(process_match_batch, chunk, i+1) for i, chunk in enumerate(chunks)]
            
            # Collect results
            for f in futures:
                final_data.extend(f.result())
    else:
        print("   -> No new matches to scrape.")  
            
    # 4. Save to CSV
    if final_data:
        df = pd.DataFrame(final_data)
        # Drop the match_id usage column before saving if desired, or keep it
        if 'match_id' in df.columns:
            # We convert to string just in case, to match existing_ids format
            df['match_id'] = df['match_id'].astype(str)
            # df = df.drop(columns=['match_id']) # Keep match_id for future incremental scrapes
            
        filename = f"{sport_name.upper()}_{league_name.upper().replace('-', '_')}_RESULTS.csv"
        
        try:
            existing_df = pd.read_csv(filename)
            print(f"Combining with existing {len(existing_df)} rows...")
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['date', 'home team', 'away team'], keep='last')
            df = combined_df
        except FileNotFoundError:
            pass
            
        try:
            df['date_obj'] = pd.to_datetime(df['date'], format='%d/%m/%Y', dayfirst=True)
            df = df.sort_values(by='date_obj')
            df = df.drop(columns=['date_obj'])
        except:
            pass
            
        df.to_csv(filename, index=False)
        print(f"SUCCESS! Saved {len(df)} total matches to {filename}")
    else:
        print("No data collected.")

def run_scraper_interface():
    # Import the list of leagues
    try:
        from league_urls import LEAGUE_URLS
    except ImportError:
        print("Error: league_urls.py not found. Using default URL.")
        LEAGUE_URLS = {"La Liga 2 (Default)": "https://www.flashscore.com/football/spain/laliga2/results/"}
        
    print("\n" + "="*50)
    print("FLASHCORE SCRAPING INTERFACE (PARALLEL)")
    print("="*50)
    print("\nAvailable Leagues:")
    leagues_list = list(LEAGUE_URLS.keys())
    for idx, name in enumerate(leagues_list):
        print(f"{idx + 1}. {name}")
    print("A. SCRAPE ALL")
    
    choice = input("\nEnter number to scrape ONE league, 'A' for ALL, or 'S' to SKIP: ").strip().upper()
    
    if choice == 'S':
        print("Skipping scraping...")
        return

    if choice == 'A':
        for name, url in LEAGUE_URLS.items():
            print(f"\n>>> STARTING BATCH SCRAPE: {name}")
            try:
                scrape_flashscore_final(url)
            except Exception as e:
                print(f"!!! CRITICAL ERROR SCRAPING {name}: {e}")
                print("Continuing to next league...")
                time.sleep(3)

    elif choice == 'F':
        print("\n=== FORCE FULL SCRAPE MODE ===")
        print("This will ignore existing data and re-scrape EVERYTHING. This takes a long time.")
        confirm = input("Are you sure? [y/N]: ").strip().lower()
        if confirm == 'y':
            for name, url in LEAGUE_URLS.items():
                print(f"\n>>> FORCE SCRAPING: {name}")
                try:
                    scrape_flashscore_final(url, force_full=True)
                except Exception as e:
                    print(f"!!! CRITICAL ERROR SCRAPING {name}: {e}")
        else:
             print("Aborted.")

    elif choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(leagues_list):
            selected_name = leagues_list[idx]
            print(f"\n>>> SCRAPING: {selected_name}")
            scrape_flashscore_final(LEAGUE_URLS[selected_name])
        else:
            print("Invalid selection.")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    run_scraper_interface()