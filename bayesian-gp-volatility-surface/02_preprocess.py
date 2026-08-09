"""
Simple preprocessing script for S&P 500 data. We split the data into a design matrix X and target y, using the 80/20 rule.

Outputs
-------
design_matrix_date.npz
    This is the main file, we compress the numpy array into a .npz file, and the .npz file is loaded by the main script in the next module for prior fitting.

"""
import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split


def preprocess_data(date_str, raw_dir="data/raw", proc_dir="data/processed"):
    """
    Main function that preprocesses the data for the S&P 500 data, creates a path for the SPX quotes that we need, meaning that it filters out the small iv and the super large, by volume (>0), and non-zero bid-ask.

    Output
    ------
    A CSV file stored in /data/processed containing the design matrix and target variables.
    out_path
    X_raw
    iv_raw
    w_raw
    train_idx
    test_idx
    spot

    Notes
    -----
    None

    """
    print(f"Preprocessing data for {date_str}...")
    in_path = os.path.join(raw_dir, f"spx_{date_str}.csv")
    df = pd.read_csv(in_path)

    df = df[(df['ImpliedVol'] > 0.01) & (df['ImpliedVol'] < 2.5)]
    df = df[df['Volume'] > 0]
    df = df[(df['Bid'] > 0) & (df['Ask'] > 0)]

    # transform to Total Variance
    df['TotalVariance'] = (df['ImpliedVol'] ** 2) * df['Maturity_T']

    #create design matrix x and target y
    X_raw = df[['Moneyness_K', 'Maturity_T']].values
    iv_raw = df['ImpliedVol'].values
    w_raw = df['TotalVariance'].values

    # train and test split
    indices = np.arange(len(X_raw))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    #make a new directory with the processed data, exist_ok=True ensures that the directory is created if it doesn't exist'
    os.makedirs(proc_dir, exist_ok=True)
    out_path = os.path.join(proc_dir, f"design_matrix_{date_str}.npz")
    np.savez(
        out_path,
        X_raw=X_raw,
        iv_raw=iv_raw,
        w_raw=w_raw,
        train_idx=train_idx,
        test_idx=test_idx,
        spot=df['Spot'].iloc[0]
    )
    print(f"Saved processed data to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True)
    args = parser.parse_args()
    preprocess_data(args.date)
