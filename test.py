import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df_clubs_overview = pd.read_pickle("./df_clubs_overview.pkl")

df_clubs = df_clubs_overview.loc[~df_clubs_overview.apply(lambda row: row.astype(str).str.contains('-').any(), axis=1)]


# Convert to numeric

def convert_market_value(value):
    try:
        # Entferne Kommas und ersetze sie durch Punkte
        value = value.replace(",", ".")
        
        # Millionen
        if "Mio. €" in value:
            value = value.replace("Mio. €", "")
            return float(value) * 1e6  # Multipliziere mit 1 Million

        # Tausend
        elif "Tsd. €" in value:
            value = value.replace("Tsd. €", "")
            return float(value) * 1e3  # Multipliziere mit 1 Tausend

        else:
            return None  # Falls die Konvertierung fehlschlägt, gebe None zurück

    except Exception as e:
        print(f"Konvertierungsfehler: {e}, Wert: {value}")
        return None  # Falls die Konvertierung fehlschlägt, gebe None zurück

# Anwenden der Konvertierungsfunktion auf die betreffenden Spalten
df_clubs['AVG_MARKET_VALUE'] = df_clubs['AVG_MARKET_VALUE'].apply(convert_market_value)
df_clubs['TOTAL_MARKET_VALUE'] = df_clubs['TOTAL_MARKET_VALUE'].apply(convert_market_value)


print(df_clubs['AVG_MARKET_VALUE'] .isna().sum())

