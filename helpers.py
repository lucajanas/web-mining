import pandas as pd
import numpy as np

Mapping = {
    '1.FC Kaiserslautern': "Lautern.csv", 
    '1.FC Nürnberg':'Nuernberg.csv', 
    '1.FSV Mainz 05':'Mainz.csv', 
    'Arminia Bielefeld':'Bielefeld.csv', 
    'Bayer 04 Leverkusen':'Leverkusen.csv',
    'Bayern München':'Bayern.csv',
    'Borussia Dortmund':'Dortmund.csv', 
    'Borussia Mönchengladbach':"Gladbach.csv", 
    'FC Schalke 04':'Schalke.csv', 
    'Hamburger SV':'Hamburg.csv', 
    'Hannover 96':'Hannover.csv', 
    'Hansa Rostock':'Rostock.csv', 
    'Hertha BSC':'Hertha.csv', 
    'SC Freiburg':'Freiburg.csv', 
    'VfB Stuttgart':'Stuttgart.csv', 
    'VfL Bochum': 'Bochum.csv', 
    'VfL Wolfsburg':'Wolfsburg.csv', 
    'Werder Bremen':'Werder.csv', 
    '1.FC Köln':'Koeln.csv', 
    'Eintracht Frankfurt':'Frankfurt.csv', 
    'MSV Duisburg':'Duisburg.csv', 
    'Alemannia Aachen':'Aachen.csv', 
    'Energie Cottbus':'Cottbus.csv', 
    'Karlsruher SC':'Karlsruhe.csv', 
    'TSG Hoffenheim':'Hoffenheim.csv', 
    'FC St. Pauli':'Pauli.csv', 
    'FC Augsburg':'Augsburg.csv', 
    'Fortuna Düsseldorf':'Duesseldorf.csv', 
    'Greuther Fürth':'Fuerth.csv', 
    'Eintracht Braunschweing':'Braunschweig.csv', 
    'SC Paderborn':'Paderborn.csv', 
    'Darmstadt 98':'Darmstadt.csv', 
    'FC Ingolstadt':'Ingolstadt.csv', 
    'RasenBallsport Leipzig':'RB Leipzig.csv', 
    'Union Berlin':'Union Berlin.csv'
}

Not_available = ['FC St. Pauli', 'RasenBallsport Leipzig','Union Berlin']

def elo_value(path, match_date, home_team, away_team):
    
    home_team_path = f'{path}/{Mapping[home_team]}'
    away_team_path = f'{path}/{Mapping[away_team]}'
    match_date = pd.to_datetime(match_date)
    
    df_home = pd.read_csv(home_team_path)
    df_away = pd.read_csv(away_team_path)
    
    df_home['From'] = pd.to_datetime(df_home['From'])
    df_home['To'] = pd.to_datetime(df_home['To'])
    df_away['From'] = pd.to_datetime(df_away['From'])
    df_away['To'] = pd.to_datetime(df_away['To'])
    
    home = df_home[(df_home['From'] <= match_date) & (df_home['To'] >= match_date)]
    away = df_away[(df_away['From'] <= match_date) & (df_away['To'] >= match_date)]
    
    closest_home_elo = 'N/A'
    closest_away_elo = 'N/A'
    
    if not home.empty:
        closest_home_elo = home.iloc[0]['Elo']
    else:
        closest_home = df_home[df_home['From'] <= match_date].sort_values('To', ascending=False)
        if not closest_home.empty:
            closest_home_elo = closest_home.iloc[0]['Elo']
    
    if not away.empty:
        closest_away_elo = away.iloc[0]['Elo']
    else:
        closest_away = df_away[df_away['From'] <= match_date].sort_values('To', ascending=False)
        if not closest_away.empty:
            closest_away_elo = closest_away.iloc[0]['Elo']

    return {
        'home': closest_home_elo,
        'away': closest_away_elo,
    }
    
    
def elo_rating(path, match_date, home_team, away_team):
    '''
    Der Elo-Wert ist aus der Perspektive der Heimmanschaft
    0 : sichere Niederlage
    1 : sicherer Sieg
    '''
    elo = elo_value(path,match_date, home_team, away_team)
    if elo['home'] != 'N/A' and elo ['away'] != 'N/A':
        dr = 100 + elo['home'] - elo['away']
        w_e = 1 / (pow(10, -dr / 400) + 1)
        return w_e
    else:
        return None