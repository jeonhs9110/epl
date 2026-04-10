import pandas as pd
import glob
import re
from datetime import datetime


def extract_season_years(filename):
    """
    Tries to find years in the filename (e.g., '2022_2023' -> 2022, 2023).
    Defaults to 2024-2025 if not found.
    """
    # Look for patterns like "2022_2023" or "2022-2023"
    match = re.search(r'(\d{4})[_-](\d{4})', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        # Default assumption if no year in filename: Current Season
        return 2024, 2025


def clean_and_fix_date(date_str, start_year, end_year):
    """
    Fixes dates like '27/11/' or '14/08' and assigns the correct historical year.
    """
    if not isinstance(date_str, str):
        return None

    # 1. Clean dirty formats (e.g., "27/11/" -> "27/11")
    date_str = date_str.strip().strip('/').strip('.')

    # 2. Split into Day and Month
    # We expect formats like "27/11" or "27/11/2025"
    parts = date_str.split('/')

    if len(parts) >= 2:
        day = int(parts[0])
        month = int(parts[1])

        # 3. Intelligent Year Assignment
        # In football, the season splits across years.
        # Aug (8) to Dec (12) -> Start Year (e.g., 2022)
        # Jan (1) to July (7) -> End Year (e.g., 2023)
        if month >= 8:
            correct_year = start_year
        else:
            correct_year = end_year

        return datetime(correct_year, month, day)

    return None


def merge_csvs():
    # Find all result CSV files
    all_files = glob.glob("*_RESULTS.csv") + glob.glob("old_matches/*_RESULTS.csv")
    print(f"Found {len(all_files)} files to merge: {all_files}")

    if not all_files:
        print("No CSV files found! Make sure they are in this folder.")
        return

    df_list = []

    for filename in all_files:
        print(f"Processing {filename}...")
        df = pd.read_csv(filename)

        # Detect season years from the filename
        start_year, end_year = extract_season_years(filename)
        print(f"   -> Detected Season: {start_year}/{end_year}")

        # Apply the date fix to every row
        # We use a lambda function to apply our smart logic row by row
        df['date_obj'] = df['date'].apply(lambda x: clean_and_fix_date(x, start_year, end_year))

        # Remove rows where date failed to parse
        df = df.dropna(subset=['date_obj'])

        df_list.append(df)

    # Combine all dataframes
    master_df = pd.concat(df_list, ignore_index=True)

    # Update the string 'date' column to match the fixed objects
    master_df['date'] = master_df['date_obj'].dt.strftime('%d/%m/%Y')

    # Sort chronologically
    master_df = master_df.sort_values(by='date_obj')

    # Drop the temporary object column
    master_df = master_df.drop(columns=['date_obj'])

    # Save
    output_filename = "MASTER_MATCH_DATA.csv"
    master_df.to_csv(output_filename, index=False)
    print(f"\nSUCCESS! Merged {len(master_df)} clean matches into '{output_filename}'.")
    print(master_df.head())


if __name__ == "__main__":
    merge_csvs()