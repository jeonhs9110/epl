import os
import sys
import subprocess

# Import modules (assuming they are in the same directory)
try:
    import scrape_flashscore
    import check_data
    import scrape_upcoming
    import train_hybrid # Checkpoint: New Import
    from league_urls import LEAGUE_URLS # Import for Power Mode
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def run_power_mode():
    print("\n" + "!"*60)
    print("POWER MODE ACTIVATED")
    print("!"*60 + "\n")

    # 1. Update Old Matches Odds
    print("[POWER MODE] Running update_odds_old_matches...")
    try:
        # Assuming the script is in 'old_matches' folder
        script_path = os.path.join("old_matches", "update_odds_old_matches.py")
        if os.path.exists(script_path):
             subprocess.run([sys.executable, script_path], check=True)
             print(">> Matches updated successfully.")
        else:
             print(f">> WARNING: {script_path} not found. Skipping.")
    except Exception as e:
        print(f">> Error running update_odds_old_matches: {e}")

    # 2. Scrape All Current Season
    print("\n[POWER MODE] Scraping ALL Current Season Leagues...")
    for name, url in LEAGUE_URLS.items():
        print(f"\n>>> POWER SCRAPE: {name}")
        scrape_flashscore.scrape_flashscore_final(url)

    # 3. Upcoming Matches
    print("\n[POWER MODE] Scraping Upcoming Fixtures...")
    if not scrape_upcoming.scrape_fixtures(days=30):
        print(">> [WARNING] Upcoming fixture scrape returned no matches.")
    
    # 4. Force Validation
    print("\n[POWER MODE] Validating Data...")
    check_data.run_checklist()

    # 4.5 Force Full Deep Learning Training
    print("\n[POWER MODE] Training Deep Learning Model (Architecture Sync)...")
    try:
         # Run train_dl.py
         script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_dl.py")
         subprocess.run([sys.executable, script_path], check=True)
         # Mark DL as already trained so train_hybrid's auto-retrain guard skips it
         os.environ["FOBO_SKIP_DL_TRAIN"] = "true"
         print(">> Deep Learning Model Trained Successfully.")
    except Exception as e:
         print(f">> Critical Error in DL Training: {e}")
         sys.exit(1)

    # 5. Force Full Training (Config)
    print("\n[POWER MODE] Configuring Full Training...")
    return True # Indicates Power Mode was run, proceed to training with overrides

def main():
    print("\n" + "="*60)
    print("FOBO AI UNIFIED PIPELINE")
    print("="*60)

    # 0. POWER MODE CHECK
    is_power_mode = False
    pm = input("\nEnable POWER MODE? (Updates everything + Full Training) [y/N]: ").strip().lower()
    if pm == 'y':
        is_power_mode = run_power_mode()
    
    if not is_power_mode:
        # 1. SCRAPING STAGE
        print("\n[STEP 1] DATA ACQUISITION")
        scrape_flashscore.run_scraper_interface()
        
        # 1.5 UPCOMING FIXTURES
        print("\n[STEP 1.5] UPCOMING FIXTURES")
        do_scrape = input("Scrape upcoming matches for next 30 days? [y/N]: ")
        if do_scrape.lower() == 'y':
            success = scrape_upcoming.scrape_fixtures(days=30)
            if not success:
                print("\n[CRITICAL ERROR] Scraper failed or found no matches.")
                print("Aborting pipeline to prevent bad data ingestion.")
                sys.exit(1)
            print(">> Scrape Successful.")
        else:
            print(">> Skipping upcoming fixtures scrape.")

        # 2. VALIDATION STAGE
        print("\n[STEP 2] DATA INTEGRITY CHECK")
        check_data.run_checklist()

    # 3. CONFIGURATION STAGE
    print("\n[STEP 3] MODEL CONFIGURATION")
    
    if is_power_mode:
         # Force Retrain settings
         retrain = 'y'
         os.environ['FOBO_SKIP_TRAINING'] = 'false'
         os.environ["FOBO_TEST_MODE"] = "false"
         print(">> Power Mode: Retraining ENABLED (Full Mode).")
    else:
        retrain = input("Do you want to retrain the models (study new/existing data)? [y/N]: ").strip().lower()
        
        if retrain == 'y':
            print(">> Retraining ENABLED. This may take some time.")
            os.environ['FOBO_SKIP_TRAINING'] = 'false'
            
            # TEST MODE PROMPT
            if input("TEST MODE? (Runs only 1 Epoch for speed) [y/N]: ").strip().lower() == 'y':
                os.environ["FOBO_TEST_MODE"] = "true"
                print("TEST MODE ENABLED: Models will train for 1 epoch only.")
            else:
                os.environ["FOBO_TEST_MODE"] = "false"
            
            # SEQUENCE LENGTH OPTIMIZATION PROMPT
            if input("OPTIMIZE SEQUENCE LENGTHS? (Calculate optimal lookup windows per league)? [y/N]: ").strip().lower() == 'y':
                print("\n>> Running Sequence Length Optimization (This may take a few minutes)...")
                try:
                    opt_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimize_seq_length.py")
                    subprocess.run([sys.executable, opt_script_path], check=True)
                    print(">> Sequence Lengths Optimized and Saved.")
                except Exception as e:
                    print(f">> Optimization failed: {e}. Falling back to default lengths.")
            
            # SKIP DL PROMPT
            if input("SKIP DL Training (Use existing model and jump to RL)? [y/N]: ").strip().lower() == 'y':
                os.environ["FOBO_SKIP_DL_TRAIN"] = "true"
                print("SKIPPING DL Training: Generating PPO Agent only.")
            else:
                os.environ["FOBO_SKIP_DL_TRAIN"] = "false"

            # SKIP PPO PROMPT
            if input("SKIP PPO Training (Use existing ppo_agent.pth)? [y/N]: ").strip().lower() == 'y':
                os.environ["FOBO_SKIP_PPO_TRAIN"] = "true"
                print("SKIPPING PPO Training: Using saved agent.")
            else:
                os.environ["FOBO_SKIP_PPO_TRAIN"] = "false"
            
            # FORCE DL TRAINING BEFORE HYBRID (Fixes Model Mismatch)
            _skip_dl = os.environ.get("FOBO_SKIP_DL_TRAIN", "false").lower() == "true"
            if _skip_dl:
                print("> Skipping DL Training (using existing model). Running PPO Agent only...")
            else:
                print("> Retraining Deep Learning Model (Architecture Sync)...")
            try:
                 script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_dl.py")
                 subprocess.run([sys.executable, script_path], check=True)
                 # Mark DL as already trained so train_hybrid's auto-retrain guard skips it
                 os.environ["FOBO_SKIP_DL_TRAIN"] = "true"
                 if _skip_dl:
                     print("> PPO Agent complete. FOBO_SKIP_DL_TRAIN remains true.")
                 else:
                     print("> DL Training complete. FOBO_SKIP_DL_TRAIN set to prevent duplicate run.")
            except Exception as e:
                 print(f"> Deep Learning Training Failed: {e}")
                 # If DL fails, Hybrid will fail too - abort
                 sys.exit(1)
                
        else:
            print(">> Using PRE-EXISTING models (No training).")
            os.environ['FOBO_SKIP_TRAINING'] = 'true'

    # 3.5 HYBRID TRAINING (XGBoost)
    print("\n[STEP 3.5] HYBRID ENSEMBLE TRAINING")
    if retrain == 'y':
        skip_xgb = input("SKIP XGBoost training (use existing saved model, calibrator only)? [y/N]: ").strip().lower() == 'y'
        if skip_xgb:
            os.environ['FOBO_SKIP_XGB_TRAIN'] = 'true'
            print(">> Skipping XGBoost training. Running calibrator only.")
        if train_hybrid.train_hybrid():
            print(">> Hybrid Model Trained Successfully.")
        else:
            print(">> [WARNING] Hybrid Training Failed. App will fall back to Deep Learning only.")
        os.environ.pop('FOBO_SKIP_XGB_TRAIN', None)
    else:
        print(">> Skipping Hybrid Training (Configured to use pre-existing).")

    # 4. EXECUTION STAGE
    print("\n[STEP 4] LAUNCHING APPLICATION")
    print("Starting Flask Server... (Press Ctrl+C to stop)")
    print("="*60 + "\n")
    
    # Run app.py as a subprocess so it picks up the environment variable
    # and keeps the pipeline script clean from Flask's infinite loop blocking
    try:
        # Get absolute path to app.py based on this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(script_dir, "app.py")
        
        # CRITICAL: Prevent app.py from training AGAIN since we just did it above
        os.environ['FOBO_SKIP_TRAINING'] = 'true'
        
        subprocess.run([sys.executable, app_path], check=True, cwd=script_dir)
    except KeyboardInterrupt:
        print("\nPipeline stopped by user.")
    except Exception as e:
        print(f"Error running app.py: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline stopped by user.")
