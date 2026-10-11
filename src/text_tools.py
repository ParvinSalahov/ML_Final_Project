import re
import pandas as pd

def extract_rooms_from_description(text: str) -> int | None:
    """
    Extracts the room count from real estate property descriptions using regex patterns.
    Handles variations like 'otaq', 'otag', suffixes (-lı, -dir, -dı, etc.), and attached numbers (e.g., '4otag').
    """
    if pd.isna(text):
        return None
    
    # Pattern 1: Matches formats with number followed by otaq/otag and optional suffixes 
    # e.g., "3 otaq", "4-otagli", "5otaqdır", "3 otaqdı", "4otag"
    match = re.search(r'(\d+)\s*-?\s*(?:otaq|otag)(?:lı|li|lu|lü|dır|dir|dur|dür|dı|di|du|dü)?', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Pattern 2: Matches alternative formats like "otaq sayı: 3", "otag: 4"
    match_alt = re.search(r'(?:otaq|otag)\s*(?:sayı)?[:\-]?\s*(\d+)', text, re.IGNORECASE)
    if match_alt:
        return int(match_alt.group(1))
        
    return None

import re
import pandas as pd

def extract_repair_from_description(text: str) -> str | None:
    """
    Extracts the repair/renovation status from real estate property descriptions.
    Handles variations like 'təmirli', 'təmirlidir', 'remontlu', suffixes, and safely ignores negative forms.
    """
    if pd.isna(text):
        return None
    
    text_lower = text.lower()
    
    # 1. Check for explicit negative conditions (təmirsiz, təmirli deyil, remontsuz, vb.)
    if re.search(r'(?:təmir|temir|remont)\s*siz', text_lower) or 'təmirli deyil' in text_lower or 'temirli deyil' in text_lower:
        return 'yoxdur'
        
    # 2. Check for positive repair conditions
    if re.search(r'(?:təmir|temir|remont)\s*(?:li|lı|lu|lü|dir|dır|dur|dür|di|dı|du|dü)?', text_lower):
        return 'var'
        
    return None