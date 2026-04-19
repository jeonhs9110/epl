
import pandas as pd
import re
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from selenium import webdriver
import browser_utils
import os
import time
import random
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

# Import league URLs
try:
    from league_urls import LEAGUE_URLS
except ImportError:
    print("Error: league_urls.py not found.")
    LEAGUE_URLS = {}

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

def extract_odds_from_source(page_source):
    # Logic matched to scrape_flashscore.py
    soup = BeautifulSoup(page_source, 'html.parser')
    odds_vals = [0.0, 0.0, 0.0]
    
    # Strategy 1: Find 1xBet specific row
    bookmaker_elem = soup.find(attrs={"title": "1xBet"})
    if not bookmaker_elem:
        bookmaker_elem = soup.find('img', attrs={"alt": "1xBet"})
        
    target_row = None
    if bookmaker_elem:
        curr = bookmaker_elem.parent
        for _ in range(10): 
            if not curr: break
            vals = curr.find_all(class_=re.compile("oddsValue"))
            if len(vals) >= 3:
                target_row = curr
                break
            curr = curr.parent
            
    # Strategy 2: Fallback
    if not target_row:
        all_vals = soup.find_all(class_=re.compile("oddsValue"))
        if len(all_vals) >= 3:
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

def process_upcoming_batch(matches_batch, batch_id):
    """
    Worker function to process a batch of upcoming matches.
    """
    if not matches_batch:
        return []

    print(f"   [Worker {batch_id}] Starting batch of {len(matches_batch)} matches...")
    
    # Setup Headless One-time for batch
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    # Using full browser to avoid detection issues, consistent with other scrapers
    # options.add_argument('--headless=new')

    browser_utils.apply_cloud_options(options)
    browser_utils.strip_incompatible_options(options)
    driver = webdriver.Chrome(service=browser_utils.build_service(), options=options)
    
    processed_matches = []
    
    try:
        total = len(matches_batch)
        for idx, match in enumerate(matches_batch):
            match_id = match.get('match_id')
            if not match_id:
                match['odds_vals'] = [0.0, 0.0, 0.0]
                processed_matches.append(match)
                continue
                
            clean_id = match_id.replace('g_1_', '')
            main_url = f"https://www.flashscore.com/match/{clean_id}/#/match-summary"
            
            odds_vals = [0.0, 0.0, 0.0]
            try:
                driver.get(main_url)
                if idx == 0:
                    handle_consent_popup(driver)
                time.sleep(1.5)
                
                odds_url = f"https://www.flashscore.com/match/{clean_id}/#/odds-comparison/1x2-odds/full-time"
                driver.get(odds_url)
                
                try:
                    WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "ui-table__row"))
                    )
                except:
                    time.sleep(1)
                    driver.get(odds_url)
                    time.sleep(2)
                    
                odds_vals = extract_odds_from_source(driver.page_source)
                print(f"   [Worker {batch_id}] {match['Home']} vs {match['Away']} -> {odds_vals}")
                
            except Exception as e:
                print(f"   [Worker {batch_id}] Error: {e}")
                
            match['odds_vals'] = odds_vals
            processed_matches.append(match)
            
    finally:
        driver.quit()
        
    return processed_matches

def scrape_fixtures(days=30):
    if not LEAGUE_URLS:
        print("No leagues to scrape.")
        return False

    print("\n" + "="*50)
    print(f"  SCRAPING UPCOMING FIXTURES (Next {days} Days)")
    print("="*50)

    today = datetime.now()
    end_date = today + timedelta(days=days)
    print(f"Filtering for matches between {today.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")

    driver = None
    
    def init_driver():
        options = webdriver.ChromeOptions()
        options.add_experimental_option("detach", True)

        # Anti-detection
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

        browser_utils.apply_cloud_options(options)
        browser_utils.strip_incompatible_options(options)
        d = webdriver.Chrome(service=browser_utils.build_service(), options=options)
        
        # Remove navigator.webdriver flag
        d.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
                })
            """
        })
        return d

    try:
        driver = init_driver()
        all_upcoming = [] 

        for league_name, url in LEAGUE_URLS.items():
            print(f"\nTarget: {league_name} -> {url.replace('/results/', '/fixtures/')}")
            fixtures_url = url.replace('/results/', '/fixtures/')
            
            try:
                # [CRITICAL] Check if driver is alive
                try:
                    _ = driver.window_handles
                except:
                    print("   [WARNING] Driver died. Restarting...")
                    driver = init_driver()

                driver.get(fixtures_url)
                
                # Consent
                handle_consent_popup(driver)
                
                # --- ROBUST AUTO-EXPAND ---
                print("  Checking for 'Show more matches'...")
                max_retries = 3
                consecutive_errors = 0
                
                while True:
                    try:
                        current_matches = len(driver.find_elements(By.CSS_SELECTOR, "[id^='g_1_']"))
                        
                        # 1. Find button by text
                        more_btn = WebDriverWait(driver, 5).until(
                            EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Show more matches')]"))
                        )
                        
                        # 2. Scroll
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", more_btn)
                        time.sleep(random.uniform(1.0, 2.0))
                        
                        # 3. Click (Stealth)
                        actions = ActionChains(driver)
                        actions.move_to_element(more_btn).pause(random.uniform(0.2, 0.5)).click().perform()
                        
                        print("   -> Clicked 'Show more matches', waiting for load...")
                        
                        # 4. Wait for count increase
                        try:
                            WebDriverWait(driver, 10).until(
                                lambda d: len(d.find_elements(By.CSS_SELECTOR, "[id^='g_1_']")) > current_matches
                            )
                            consecutive_errors = 0
                            time.sleep(random.uniform(2.0, 3.5))
                        except:
                            if consecutive_errors >= max_retries:
                                break
                            consecutive_errors += 1
                            time.sleep(2)
                            continue
                            
                    except Exception:
                        # Break loop if button not found (end of list)
                        break

                # Parse content
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                events = soup.find_all('div', class_=re.compile('event__'))
                print(f"  Found {len(events)} event rows. Parsing basic info...")
                
                # 1. Parse all valid matches first
                parsed_matches = []
                
                for event in events:
                    if not event.get('id', '').startswith('g_1_'):
                        continue
                        
                    try:
                        # Date Extraction
                        time_div = event.find('div', class_='event__time')
                        date_text = time_div.get_text(strip=True) if time_div else ""
                        
                        if not date_text: continue
                        
                        date_part = date_text.split(' ')[0].replace('.', '/')
                        try:
                            day, month = map(int, date_part.split('/')[:2])
                        except:
                            continue
                            
                        match_year = today.year
                        if today.month == 12 and month == 1:
                            match_year += 1
                        
                        match_date = datetime(match_year, month, day)
                        
                        if today.replace(hour=0, minute=0, second=0, microsecond=0) <= match_date <= end_date:
                            home_team = event.select_one('.event__homeParticipant').get_text(strip=True)
                            away_team = event.select_one('.event__awayParticipant').get_text(strip=True)
                            
                            # Clean Flashscore Artifacts ('TeamName2', 'Lazio 3') safely
                            home_team = home_team.replace("Advancing to next round", "").strip()
                            away_team = away_team.replace("Advancing to next round", "").strip()
                            home_team = re.sub(r'\s+[1-9]$', '', home_team).strip()
                            away_team = re.sub(r'\s+[1-9]$', '', away_team).strip()
                            home_team = re.sub(r'\s*\(\d+\)$', '', home_team).strip()
                            away_team = re.sub(r'\s*\(\d+\)$', '', away_team).strip()
                            parts = date_text.strip().split(' ')
                            time_val = parts[1] if len(parts) > 1 else "00:00"
                            
                            parsed_matches.append({
                                'League': league_name,
                                'Date': match_date.strftime('%Y-%m-%d'),
                                'Time': time_val,
                                'Home': home_team,
                                'Away': away_team,
                                'match_id': event.get('id'),
                                'odds_vals': [0.0, 0.0, 0.0] # Placeholder
                            })
                    except:
                        continue
                
                print(f"  -> Identified {len(parsed_matches)} matches in date range.")
                
                # 2. Process in Batches with Parallel Workers
                # We assume matches are roughly chronological. 
                # We will process in chunks of (NUM_WORKERS * matches_per_worker)
                
                from concurrent.futures import ThreadPoolExecutor
                import math
                
                NUM_WORKERS = 3
                matches_per_worker = 5 # Each worker processes 5 matches per browser session
                batch_total_size = NUM_WORKERS * matches_per_worker
                
                consecutive_zero_odds_count = 0
                STOP_THRESHOLD = 5 # If we see 5 consecutive matches with [0,0,0], we stop. (Actually, if we process in batches, checking consecutive across batches is key)
                
                # Determine chunks
                total_m = len(parsed_matches)
                main_chunks = [parsed_matches[i:i + batch_total_size] for i in range(0, total_m, batch_total_size)]
                
                should_stop = False
                
                for chunk_idx, main_chunk in enumerate(main_chunks):
                    if should_stop: 
                        print("   -> Skipping remaining matches due to consecutive missing odds.")
                        break
                        
                    # Split main_chunk into worker_chunks
                    w_chunk_size = math.ceil(len(main_chunk) / NUM_WORKERS)
                    worker_chunks = [main_chunk[i:i + w_chunk_size] for i in range(0, len(main_chunk), w_chunk_size)]
                    
                    results = []
                    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                        futures = [executor.submit(process_upcoming_batch, wc, i+1) for i, wc in enumerate(worker_chunks)]
                        for f in futures:
                            results.extend(f.result())
                    
                    # Check results for stop condition
                    # Sort results by date/time if possible to ensure we check end of list correctly?
                    # The input list 'parsed_matches' comes from DOM, usually sorted by time.
                    # We check the results in order of the list.
                    
                    for m in results:
                        odds = m.get('odds_vals', [0.0, 0.0, 0.0])
                        
                        if odds == [0.0, 0.0, 0.0]:
                            consecutive_zero_odds_count += 1
                        else:
                            consecutive_zero_odds_count = 0
                            
                        all_upcoming.append({
                            'League': m['League'],
                            'Date': m['Date'],
                            'Time': m['Time'],
                            'Home': m['Home'],
                            'Away': m['Away'],
                            'Odds_1': odds[0],
                            'Odds_X': odds[1],
                            'Odds_2': odds[2]
                        })
                        
                        if consecutive_zero_odds_count >= STOP_THRESHOLD:
                            should_stop = True
                            break
                
                print(f"  -> Extracted {len(all_upcoming)} valid upcoming matches.")
                if should_stop:
                    print("  -> Stopped early for this league.")
            
            except Exception as e_league:
                print(f"!!! Error scraping {league_name}: {e_league}")
                continue # Skip to next league

    except Exception as e:
        print(f"Error during scraping: {e}")
    finally:
        if driver: driver.quit()

    # Save Results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, "UPCOMING_MATCHES.csv")

    if all_upcoming:
        df = pd.DataFrame(all_upcoming)
        df = df.sort_values(by=['Date', 'Time'])
        df.to_csv(filename, index=False)
        print(f"\nSAVED {len(df)} UPCOMING MATCHES TO '{filename}'")
        return True
    else:
        print("\nNo upcoming matches found.")
        df = pd.DataFrame(columns=['League', 'Date', 'Home', 'Away', 'Time', 'Odds_1', 'Odds_X', 'Odds_2'])
        df.to_csv(filename, index=False)
        return False

if __name__ == "__main__":
    scrape_fixtures()
