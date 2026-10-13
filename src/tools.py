import re
import pandas as pd

def extract_rooms_from_description(text: str) -> int | None:
    """
    Extracts the room count from real estate property descriptions using regex patterns.
    Handles variations like 'otaq', 'otag', suffixes (-lı, -dir, -dı, etc.), and attached numbers (e.g., '4otag').
    """
    if pd.isna(text):
        return None
    
    match = re.search(r'(\d+)\s*-?\s*(?:otaq|otag)(?:lı|li|lu|lü|dır|dir|dur|dür|dı|di|du|dü)?', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    match_alt = re.search(r'(?:otaq|otag)\s*(?:sayı)?[:\-]?\s*(\d+)', text, re.IGNORECASE)
    if match_alt:
        return int(match_alt.group(1))
        
    return None

def extract_repair_from_description(text: str) -> str | None:
    """
    Extracts the repair/renovation status from real estate property descriptions.
    Handles variations like 'təmirli', 'təmirlidir', 'remontlu', suffixes, and safely ignores negative forms.
    """
    if pd.isna(text):
        return None
    
    text_lower = text.lower()
    
    if re.search(r'(?:təmir|temir|remont)\s*siz', text_lower) or 'təmirli deyil' in text_lower or 'temirli deyil' in text_lower:
        return 'yoxdur'
        
    if re.search(r'(?:təmir|temir|remont)\s*(?:li|lı|lu|lü|dir|dır|dur|dür|di|dı|du|dü)?', text_lower):
        return 'var'
        
    return None

def get_rooms_from_row(row):
    # 1. Önce attributes alanını senin fonksiyonla deneyelim
    if pd.notna(row.get('attributes')):
        rooms = extract_rooms_from_description(str(row['attributes']))
        if rooms is not None:
            return rooms
            
    # 2. attributes'ta yoksa description ile deneyelim
    if pd.notna(row.get('description')):
        rooms = extract_rooms_from_description(str(row['description']))
        if rooms is not None:
            return rooms
            
    return row['Otaq sayı']
def get_repair_from_row(row):

    if pd.notna(row.get('description')):
        repair_val = extract_repair_from_description(str(row['description']))
        if repair_val is not None:
            return repair_val
            
    return row.get('repair')

def impute_rooms_byarea(row, df_full, room_col='Otaq sayı', area_col='area', cat_col='Kateqoriya'):
    if pd.notna(row[room_col]):
        return row[room_col]
    
    cat = row[cat_col]
    target_area = row[area_col]
    
    if pd.isna(target_area):
        return row[room_col]
        
    train_pool = df_full[
        (df_full[cat_col] == cat) & 
        (df_full[room_col].notna()) & 
        (df_full[area_col].notna())
    ]
    
    if len(train_pool) == 0:
        return row[room_col]
        
    diffs = np.abs(train_pool[area_col] - target_area)
    nearest_indices = diffs.nsmallest(3).index
    
    imputed_rooms = int(round(train_pool.loc[nearest_indices, room_col].median()))
    return imputed_rooms

def handle_duplicates_and_average_price(df, price_col='price'):
    print(f"Row count before handling duplicates: {len(df)}")
    
    if price_col not in df.columns:
        for alt in ['Qiymət', 'qiymət', 'Price']:
            if alt in df.columns:
                price_col = alt
                break
                
    subset_cols = [col for col in df.columns if col != price_col]
    
    df[price_col] = df.groupby(subset_cols, dropna=False)[price_col].transform('mean')
    
    df = df.drop_duplicates(subset=subset_cols, keep='first').reset_index(drop=True)
    
    print(f"Row count after handling duplicates: {len(df)}")
    return df

def remove_outliers_iqr(df, columns, multiplier=1.5):
    print(f"Row count before outlier removal: {len(df)}")
    
    initial_len = len(df)
    mask = pd.Series(True, index=df.index)
    
    for col in columns:
        if col not in df.columns:
            continue
            )
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - (multiplier * iqr)
        upper_bound = q3 + (multiplier * iqr)
        
        col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
        mask = mask & col_mask

        
    df_filtered = df[mask].reset_index(drop=True)
    return df_filtered