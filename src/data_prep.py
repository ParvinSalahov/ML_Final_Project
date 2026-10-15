"""
data_prep.py — loading, cleaning, splitting, and the price-tier label.

Fill in the TODOs. Keep ALL data wrangling here so the rest of the code just
calls these functions. Fix your random seed everywhere (see SEED).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tools as tt 
# --- Config -----------------------------------------------------------------
SEED = 42
DATA_PATH = "data/house_sale.csv"   # change if your filename differs

UNUSED_COLUMNS: list[str] = [
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
    "attributes",
    "extra_info",
    "estate_rel_url",
    "Binanın növü",
    "description",
    "owner_title",
    "shop_title"
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
    df['area'] = pd.to_numeric(df['area'],errors='coerce')
    df['land_area'] = df['Torpaq sahəsi'].str.replace(r'[^0-9.]', '', regex=True)
    df['land_area'] = pd.to_numeric(df['land_area'],errors='coerce')
    df[['floor', 'total_floor']] = df['Mərtəbə'].str.split('/', expand=True)
    df['floor'] = pd.to_numeric(df['floor'].str.strip(), errors='coerce').astype('Int64')
    df['total_floor'] = pd.to_numeric(df['total_floor'].str.strip(), errors='coerce').astype('Int64')
    df = df.drop(columns=['Sahə'])
    df = df.drop(columns=['Torpaq sahəsi'])
    df = df.drop(columns=['Mərtəbə'])

    non_building_categories = ['Torpaq', 'Qaraj', 'Həyət evi/Bağ evi','Ofis','Obyekt']
    mask = df['Kateqoriya'].isin(non_building_categories)
    df.loc[mask, 'floor'] = df.loc[mask, 'floor'].fillna(0)
    df.loc[mask, 'total_floor'] = df.loc[mask, 'total_floor'].fillna(0)


    non_room_categories = ['Torpaq', 'Qaraj', 'Obyekt']
    mask_rooms = df['Kateqoriya'].isin(non_room_categories)
    df.loc[mask_rooms, 'Otaq sayı'] = df.loc[mask_rooms, 'Otaq sayı'].fillna(0)

    mask = (df['Kateqoriya'] == 'Həyət evi/Bağ evi') & (df['Otaq sayı'].isna())
    df.loc[mask, 'Otaq sayı'] = df[mask].apply(tt.get_rooms_from_row, axis=1)

    mask = df['Otaq sayı'].isna() & df['area'].notna()
    df.loc[mask, 'Otaq sayı'] = df[mask].apply(
    lambda row: tt.impute_rooms_byarea(row, df), 
    axis=1
    )

    mask = (df['Kateqoriya'] == 'Torpaq')
    df.loc[mask & df['land_area'].isna(), 'land_area'] = df.loc[mask, 'area']

    mask = (df['Kateqoriya'] == 'Torpaq')
    df.loc[mask, 'area'] = 0

    other_categories = ~df['Kateqoriya'].isin(['Həyət evi/Bağ evi', 'Torpaq'])
    df['land_area'] = pd.to_numeric(df['land_area'], errors='coerce')
    df.loc[other_categories, 'land_area'] = 0

    mask = df['repair'].isna()
    df.loc[mask, 'repair'] = df[mask].apply(tt.get_repair_from_row, axis=1)

    df = df.drop(columns=['repair'])
    df.rename(columns={'Təmir': 'repair'}, inplace=True)
    df['repair'] = df['repair'].fillna(0)

    df = df.drop(columns=['mortgage'])
    df.rename(columns={'İpoteka': 'mortgage'}, inplace=True)
    df['mortgage'] = df['mortgage'].fillna(0)

    df = df.drop(columns=['bill_of_sale'])
    df.rename(columns={'Çıxarış': 'bill_of_sale'}, inplace=True)

    df.rename(columns={'Otaq sayı': 'rooms'}, inplace=True)
    df.rename(columns={'Kateqoriya': 'category'}, inplace=True)
    for cols in UNUSED_COLUMNS:
        df = df.drop(columns=[cols])

    df = tt.handle_duplicates_and_average_price(df)
    outlier_cols = ['price', 'area', 'land_area']
    df = tt.remove_outliers_iqr(df, columns=[col for col in outlier_cols if col in df.columns], multiplier=1.5)

    return df


def make_features(df):
    """
    Turn the cleaned frame into a numeric design matrix.
      - encode categoricals (one-hot / ordinal — your choice, justify it),
      - assemble X (features) and y (target `price`).
    Return (X, y, feature_names).
    """
    binary_columns = ['repair', 'bill_of_sale', 'mortgage']
    for col in binary_columns:
        if col in df.columns:
            df[col] = df[col].apply(tt.convert_to_binary)

    df = pd.get_dummies(df, columns=['city', 'category'], drop_first=False, dtype=int)

    for col in ['location', 'address']:
        if col in df.columns:
            freq_map = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq_map).fillna(0)

    category_rename_map = {
    'category_Həyət evi/Bağ evi': 'category_country_house',
    'category_Torpaq': 'category_land',
    'category_Köhnə tikili': 'category_old_building',
    'category_Yeni tikili': 'category_new_building',
    'category_Obyekt': 'category_commercial', 
    'category_Ofis': 'category_office',
    'category_Qaraj': 'category_garage'
    }

    df = df.rename(columns=category_rename_map)

    city_rename_map = {
        'city_bakı': 'city_baku',
        'city_xırdalan': 'city_khirdalan',
        'city_xaçmaz': 'city_khachmaz',
        'city_astara': 'city_astara',
        'city_sumqayıt': 'city_sumgayit'
    }

    df = df.rename(columns=city_rename_map)

    y = np.log1p(df['price'])
    X = df.drop(columns=['price'])
    feature_names = X.columns.tolist()
    
    return X, y, feature_names


def make_tier_label(y_price, threshold=None):
    """
    Derived classification target: price TIER (premium vs standard).
    Use the MEDIAN of the TRAINING price as the split threshold.
      - premium = 1 (price >  threshold), standard = 0.
    Return (y_tier, threshold).  The threshold must come from TRAIN only
    (no peeking at val/test) to avoid leakage.
    """
    if threshold is None:
        threshold = np.median(y_price)
    
    y_tier = (y_price > threshold).astype(int)
    
    return y_tier, threshold


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Deterministic split into train / validation / test.
    Return (X_tr, y_tr, X_val, y_val, X_te, y_te).
    """

    np.random.seed(seed)
    
    price_bins = pd.qcut(y, q=4, labels=False, duplicates='drop')
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for bin_val in np.unique(price_bins):
        bin_idx = np.where(price_bins == bin_val)[0]
        np.random.shuffle(bin_idx)
        
        n = len(bin_idx)
        n_test = int(n * test_size)
        n_val = int(n * val_size)
        
        test_indices.extend(bin_idx[:n_test])
        val_indices.extend(bin_idx[n_test:n_test + n_val])
        train_indices.extend(bin_idx[n_test + n_val:])
        
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    X_tr, y_tr = X.iloc[train_indices], y.iloc[train_indices]
    X_val, y_val = X.iloc[val_indices], y.iloc[val_indices]
    X_te, y_te = X.iloc[test_indices], y.iloc[test_indices]
    
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def standardize(X_tr, *others):
    """
    Standardize features using TRAIN statistics only (mean/std from X_tr),
    then apply the same transform to the other splits.
    Return the scaled arrays in the same order as given.
    (Needed for the SVM; trees don't require it.)
    """

    means = X_tr.mean()
    stds = X_tr.std()
    
    stds = stds.replace(0, 1)
    
    X_tr_scaled = (X_tr - means) / stds
    
    scaled_others = []
    for df_split in others:
        df_scaled = (df_split - means) / stds
        scaled_others.append(df_scaled)
        
    return (X_tr_scaled, *scaled_others)
