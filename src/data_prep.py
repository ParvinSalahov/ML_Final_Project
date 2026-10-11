"""
data_prep.py — loading, cleaning, splitting, and the price-tier label.

Fill in the TODOs. Keep ALL data wrangling here so the rest of the code just
calls these functions. Fix your random seed everywhere (see SEED).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
# --- Config -----------------------------------------------------------------
SEED = 42
DATA_PATH = "data/house_sale.csv"   # change if your filename differs

LEAKAGE_COLUMNS: list[str] = [
    "unit_price",
    "total_price",
    "owner_name",
    "shop_name",
    "id_x", 
    "rel_url", 
    "estate_rel_url_x",
    "datetime_scrape_x",
    "day_x",
    "city_when",
    "hour_x",
    "vip",
    "featured",
    "mortgage",
    "repair",
    "bill_of_sale",
    "products_label",
    "estate_details_id_y",
    "estate_id",
    "estate_rel_url_y",
    "day_y",
    "hour_y",
    "datetime_scrape_y",
    "currency_x",
    "currency_y",
    "updated",
    "estate_details_id_x",
    "img_url",
    "id_y",
    "price"
]


def load_raw(path: str = DATA_PATH):
    """Read the raw CSV into a DataFrame (keep UTF-8). Return it unmodified."""
    data = pd.read_csv(DATA_PATH)
    return data 


def clean(df):
    """
    Clean the raw frame:
      - fix dtypes (prices/areas to numeric; strip currency/text),
      - handle missing values,
      - drop duplicates,
      - handle outliers (document your rule),
      - drop LEAKAGE_COLUMNS.
    Return the cleaned DataFrame.
    """
    df['area'] = df['Sahə'].str.replace(r'[^0-9.]', '', regex=True)
    df['area'] = pd.to_numeric(df['Sahə'])
    df['land_area'] = df['Torpaq sahəsi'].str.replace(r'[^0-9.]', '', regex=True)
    df['land_area'] = pd.to_numeric(df['Torpaq sahəsi'])
    df[['floor', 'total_floor']] = df['Mərtəbə'].str.split('/', expand=True)
    df['floor'] = pd.to_numeric(df['floor'].str.strip(), errors='coerce').astype('Int64')
    df['total_floor'] = pd.to_numeric(df['total_floor'].str.strip(), errors='coerce').astype('Int64')
    raise NotImplementedError("TODO: clean the data")


def make_features(df):
    """
    Turn the cleaned frame into a numeric design matrix.
      - encode categoricals (one-hot / ordinal — your choice, justify it),
      - assemble X (features) and y (target `price`).
    Return (X, y, feature_names).
    """
    raise NotImplementedError("TODO: build X, y")


def make_tier_label(y_price, threshold=None):
    """
    Derived classification target: price TIER (premium vs standard).
    Use the MEDIAN of the TRAINING price as the split threshold.
      - premium = 1 (price >  threshold), standard = 0.
    Return (y_tier, threshold).  The threshold must come from TRAIN only
    (no peeking at val/test) to avoid leakage.
    """
    raise NotImplementedError("TODO: derive the price-tier label")


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Deterministic split into train / validation / test.
    Return (X_tr, y_tr, X_val, y_val, X_te, y_te).
    """
    raise NotImplementedError("TODO: split the data")


def standardize(X_tr, *others):
    """
    Standardize features using TRAIN statistics only (mean/std from X_tr),
    then apply the same transform to the other splits.
    Return the scaled arrays in the same order as given.
    (Needed for the SVM; trees don't require it.)
    """
    raise NotImplementedError("TODO: standardize using train stats")
