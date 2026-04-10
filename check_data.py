import pandas as pd
import glob
import os
import re
from datetime import datetime


# ==========================================
# HELPER FUNCTIONS (Matched to your Model)
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


# ==========================================
# MAIN CHECKLIST SCRIPT
# ==========================================
def run_checklist():
    print("\n" + "=" * 50)
    print("FOBO AI PRE-FLIGHT CHECKLIST (Advanced)")
    print("=" * 50)

    # --- 1. FILE AUDIT ---
    print("\n[CHECK 1] Scanning for CSV Files...")
    csv_files = glob.glob('*_RESULTS.csv')

    if not csv_files:
        print("CRITICAL ERROR: No '*_RESULTS.csv' files found!")
        return

    csv_files.sort()
    for f in csv_files:
        print(f"  Found: {f}")

    # --- LOADING DATA ---
    print("\n[LOADING] Parsing data to track duplicates...")

    df_list = []
    today = datetime.now()
    current_season_start = today.year if today.month >= 8 else today.year - 1

    for f in csv_files:
        try:
            temp_df = pd.read_csv(f)
            # Normalize columns
            temp_df.columns = temp_df.columns.str.strip().str.lower()

            # Check required columns
            required = ['date', 'home team', 'away team', 'home team total goal', 'away team total goal']
            missing = [c for c in required if c not in temp_df.columns]
            if missing:
                print(f"  WARNING: Skipping {f} (Missing columns: {missing})")
                continue

            # --- NEW: Track the Source File ---
            temp_df['source_file'] = f

            # Determine Year
            file_year = current_season_start
            match = re.search(r'(\d{4})_\d{4}', f)
            if match: file_year = int(match.group(1))

            # Parse Dates
            temp_df['date_obj'] = temp_df['date'].apply(lambda x: parse_football_date(x, file_year))

            # Drop invalid dates
            valid_rows = temp_df.dropna(subset=['date_obj']).copy()
            df_list.append(valid_rows)

        except Exception as e:
            print(f"  Error reading {f}: {e}")

    if not df_list:
        print("CRITICAL ERROR: No valid data loaded.")
        return

    master_df = pd.concat(df_list, ignore_index=True)
    master_df = master_df.sort_values('date_obj').reset_index(drop=True)

    # --- 2. MATCH COUNT ---
    print("\n[CHECK 2] Dataset Size")
    print(f"  Total Matches to Train: {len(master_df):,}")

    # --- 3. DUPLICATE CHECK (UPDATED) ---
    print("\n[CHECK 3] Duplicates Analysis")
    # Identify duplicates based on Date + Teams
    key_cols = ['date_obj', 'home team', 'away team']
    dup_mask = master_df.duplicated(subset=key_cols, keep=False)

    if dup_mask.any():
        dup_count = dup_mask.sum()
        print(f"  FOUND {dup_count} DUPLICATE ENTRIES!")
        print("  Investigating sources...\n")

        # Filter to just the duplicates
        dup_rows = master_df[dup_mask].copy()

        # Group by the match details to see which files they come from
        dup_groups = dup_rows.groupby(key_cols)['source_file'].unique().reset_index()

        # Print the first 10 conflicts
        print("  CONFLICT REPORT (First 10 Duplicates):")
        for idx, row in dup_groups.head(10).iterrows():
            d_str = row['date_obj'].strftime('%Y-%m-%d')
            match_lbl = f"{row['home team']} vs {row['away team']}"
            files_found = list(row['source_file'])

            print(f"    Match: {d_str} | {match_lbl}")
            print(f"       Found in: {files_found}")

        print("\n  ACTION: Delete one of the conflicting files or remove overlapping rows.")
    else:
        print("  No duplicates found. Clean dataset!")

    # --- 4. DATA HEALTH ---
    print("\n[CHECK 4] Data Quality")
    nan_goals = master_df[master_df['home team total goal'].isna() | master_df['away team total goal'].isna()]
    if len(nan_goals) > 0:
        print(f"  WARNING: {len(nan_goals)} matches have missing scores (NaN).")
    else:
        print("  All matches have valid scores.")

    print("\n" + "=" * 50)
    print("CHECKLIST COMPLETE")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_checklist()