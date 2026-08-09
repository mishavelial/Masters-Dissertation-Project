"""
In this module, we clean the data, standardise, filter, compute moneyness and finally construct a cleaner version of the file for easier use.

Essentially, we are setting up the scene for the point estimator and the HMC to read and model the data.

We search for a CBOE CSV containing the requested date, filter the file down to SPX call options, select a small set of representative maturities, and write a cleaned market-data file for downstream modelling.

Outputs
-------
A CSV saved to data/raw/spx_<date>.csv with the following fields:
- Date
- Expiry
- OptionType
- Strike
- Maturity_T
- Spot
- Moneyness_K
- ImpliedVol
- Bid
- Ask
- LastPrice
- Volume
- OpenInterest

Notes
-----
- The script expects CBOE-style column names in the input file. This data is downloaded manually from CBOE.
- The output is tailored for the later preprocessing and surface-fitting stages.
- Quotes with missing or extremely small implied volatility are discarded.
"""

import pandas as pd
import numpy as np
import argparse
import os
import glob
import sys


def ingest_data(date_str, raw_dir="data/raw"):
    """ingest_data function takes in the dates that we want to analyse and returns a cleaned csv file.

    Parameters
    ----------
    date_str : str
        Run date in YYYY-MM-DD format.
    raw_dir : str, optional
        Directory used to search for the raw CBOE file and store the output.

    Returns
    -------
    None
        This function may seem a bit excessive, but across the data used in this research, the most common crash in the code is that it cant find the columns or the file. It is simply easier to troubleshoot it using some levels than perform guess-work of what went wrong.

        Additionally, some of the code is overexplained as the intended audience of this pipeline are statisticians and not financial professionals.
    """
    print(f"Inserting data for {date_str}...")

    #we dynamically search for a CSV whose filename contains the requested date as sometimes data names are random, but they still include the data.
    search_pattern = os.path.join(raw_dir, f"*{date_str}*.csv")
    matching_files = glob.glob(search_pattern)
    input_files = [f for f in matching_files if os.path.basename(f) != f"spx_{date_str}.csv"]

    if not input_files:
        root_matching = glob.glob(f"*{date_str}*.csv")
        input_files = [f for f in root_matching if os.path.basename(f) != f"spx_{date_str}.csv"]

    if not input_files:
        print(f"Error: No CBOE CSV in '{date_str}'.")
        sys.exit(1)

    cboe_path = input_files[0]
    print(f"Found CBOE file in {cboe_path}")

    try:
        #read the raw file and standardise the header names
        df_cboe = pd.read_csv(cboe_path)
        df_cboe.columns = [str(c).lower().strip() for c in df_cboe.columns]

        #require the underlying symbol field so that we can isolate SPX rows.
        if 'underlying_symbol' not in df_cboe.columns:
            print(f"Error: 'underlying_symbol' column missing. Found headers: {list(df_cboe.columns)}")
            sys.exit(1)
        df = df_cboe[df_cboe['underlying_symbol'].str.contains('SPX', na=False)].copy()
        df = df[df['option_type'].str.upper() == 'C'].copy()

        #fixed columns:
        quote_date_col = 'quote_datetime'
        expiration_col = 'expiration'
        spot_col = 'active_underlying_price'
        bid_col = 'bid'
        ask_col = 'ask'
        iv_col = 'implied_volatility'
        vol_col = 'trade_volume'
        oi_col = 'open_interest'

        # Convert dates and compute time-to-maturity in years.
        df['quote_date'] = pd.to_datetime(df[quote_date_col])
        df['expiration'] = pd.to_datetime(df[expiration_col])
        df['days_to_maturity'] = (df['expiration'] - df['quote_date']).dt.days
        df['Maturity_T'] = np.maximum(df['days_to_maturity'] / 365.0, 0.001)

        # Select representative maturities closest to 14, 30, 90, and 180 days. The reason for the choice of these maturities is explained in Section 6.1 of the main text.
        unique_dtes = sorted(df['days_to_maturity'].unique())

        if unique_dtes:
            target_dtes = [
                min(unique_dtes, key=lambda x: abs(x - 14)),
                min(unique_dtes, key=lambda x: abs(x - 30)),
                min(unique_dtes, key=lambda x: abs(x - 90)),
                min(unique_dtes, key=lambda x: abs(x - 180))
            ]
            target_dtes = sorted(list(set(target_dtes)))
            print(f"Selected historical Days-to-Maturity (DTE): {target_dtes}")
            df = df[df['days_to_maturity'].isin(target_dtes)].copy()
        else:
            print("Error: No unique dte's found.")
            sys.exit(1)

        # use the first remaining quote to define the spot level for the slice.
        underlying_price = df[spot_col].iloc[0]

        # reformat each option quote into the model input structure.
        all_quotes = []
        for _, row in df.iterrows():
            iv = row[iv_col]

            #we skip the quotes with small IV's as they might not be representative. They essentially create sharp spikes in the surface, which is something we want to avoid.
            if pd.isna(iv) or iv < 0.01:
                continue

            all_quotes.append({
                'Date': date_str,
                'Expiry': row['expiration'].strftime("%Y-%m-%d"),
                'OptionType': 'C',
                'Strike': row['strike'],
                'Maturity_T': row['Maturity_T'],
                'Spot': underlying_price,
                'Moneyness_K': np.log(row['strike'] / underlying_price),
                'ImpliedVol': iv,
                'Bid': row[bid_col],
                'Ask': row[ask_col],
                'LastPrice': row['close'],
                'Volume': row[vol_col],
                'OpenInterest': row[oi_col]
            })

        df_market = pd.DataFrame(all_quotes)
        if len(df_market) == 0:
            print("Error: 0 option quotes remained after filtering")
            sys.exit(1)

        os.makedirs(raw_dir, exist_ok=True)
        out_path = os.path.join(raw_dir, f"spx_{date_str}.csv")
        df_market.to_csv(out_path, index=False)

        print(f"Success! Extracted {len(df_market)} true historical option quotes.")
        print(f"Historical SPX Spot Price {underlying_price:.2f}")
        print(f"Saved to '{out_path}'")

    except Exception as e:
        print(f"Python Crash inside 01_ingest_spx: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    ingest_data(args.date)