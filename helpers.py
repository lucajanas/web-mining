import pandas as pd
import numpy as np
import os
import json
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.metrics import classification_report, accuracy_score,  balanced_accuracy_score

Mapping = {
    '1.FC Kaiserslautern': "Lautern.csv", 
    '1.FC Nürnberg':'Nuernberg.csv', 
    '1.FSV Mainz 05':'Mainz.csv', 
    'Arminia Bielefeld':'Bielefeld.csv', 
    'Bayer 04 Leverkusen':'Leverkusen.csv',
    'FC Bayern München':'Bayern.csv',
    'Borussia Dortmund':'Dortmund.csv', 
    'Borussia Mönchengladbach':"Gladbach.csv", 
    'FC Schalke 04':'Schalke.csv', 
    'Hamburger SV':'Hamburg.csv', 
    'Hannover 96':'Hannover.csv', 
    'FC Hansa Rostock':'Rostock.csv', 
    'Hertha BSC':'Hertha.csv', 
    'SC Freiburg':'Freiburg.csv', 
    'VfB Stuttgart':'Stuttgart.csv', 
    'VfL Bochum': 'Bochum.csv', 
    'VfL Wolfsburg':'Wolfsburg.csv', 
    'SV Werder Bremen':'Werder.csv', 
    '1.FC Köln':'Koeln.csv', 
    'Eintracht Frankfurt':'Frankfurt.csv', 
    'MSV Duisburg':'Duisburg.csv', 
    'Alemannia Aachen':'Aachen.csv', 
    'FC Energie Cottbus':'Cottbus.csv', 
    'Karlsruher SC':'Karlsruhe.csv', 
    'TSG 1899 Hoffenheim':'Hoffenheim.csv', 
    'FC St. Pauli':'Pauli.csv', 
    'FC Augsburg':'Augsburg.csv', 
    'Fortuna Düsseldorf':'Duesseldorf.csv', 
    'SpVgg Greuther Fürth':'Fuerth.csv', 
    'Eintracht Braunschweig':'Braunschweig.csv', 
    'SC Paderborn 07':'Paderborn.csv', 
    'SV Darmstadt 98':'Darmstadt.csv', 
    'FC Ingolstadt 04':'Ingolstadt.csv', 
    'RasenBallsport Leipzig':'RB Leipzig.csv', 
    '1.FC Union Berlin':'Union Berlin.csv'
}

Not_available = ['FC St. Pauli', 'RasenBallsport Leipzig','1.FC Union Berlin']

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
    if elo['home'] == 'N/A':
        print(f'home N/A: {home_team}')
        return None
    elif elo ['away'] == 'N/A':
        print(f'home N/A: {away_team}')
        return None 
    else:
        dr = 100 + elo['home'] - elo['away']
        w_e = 1 / (pow(10, -dr / 400) + 1)
        return w_e


def get_team_metrics(X_test, y_test, y_pred, le_teams):
    
    round_precision = 2
    # print(type(X_test))
    # print(type(y_test))
    # print(type(y_pred))
    # print(X_test.head())
    X_test_decoded = X_test.copy()
    X_test_decoded['HOME_TEAM'] = le_teams.inverse_transform(pd.to_numeric(X_test['HOME_TEAM'], errors='coerce'))
    X_test_decoded['AWAY_TEAM'] = le_teams.inverse_transform(pd.to_numeric(X_test['AWAY_TEAM'], errors='coerce'))
    X_test_decoded['RESULT'] = y_test
    X_test_decoded['PREDICTED_RESULT'] = y_pred

    # Liste der Teams
    unique_teams = pd.concat([X_test_decoded['HOME_TEAM'], X_test_decoded['AWAY_TEAM']]).unique()

    # Initialisiere ein leeres Dictionary zur Speicherung der Metriken
    y_pred = pd.to_numeric(y_pred, errors='coerce').to_numpy()
    y_test = pd.to_numeric(y_test, errors='coerce').to_numpy()
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results = {'Total_Accuracy': round(accuracy_score(y_test, y_pred),round_precision)}
    results['Total_Balanced_Accuracy'] =  round(balanced_accuracy_score(y_test, y_pred),round_precision)
    results['Total_Precision_Class_0'] = round(report['0']['precision'],round_precision)
    results['Total_Precision_Class_1'] = round(report['1']['precision'],round_precision)
    results['Total_Precision_Class_2'] = round(report['2']['precision'],round_precision)
    results['Total_Recall_Class_0'] = round(report['0']['recall'],round_precision)
    results['Total_Recall_Class_1'] = round(report['1']['recall'],round_precision)
    results['Total_Recall_Class_2'] = round(report['2']['recall'],round_precision)
    results['Total_F1-Score_Class_0'] = round(report['0']['f1-score'],round_precision)
    results['Total_F1-Score_Class_1'] = round(report['1']['f1-score'],round_precision)
    results['Total_F1-Score_Class_2'] = round(report['2']['f1-score'],round_precision)
    
    # Berechne Metriken pro Team
    for team in unique_teams:
        team_data = X_test_decoded[(X_test_decoded['HOME_TEAM'] == team)]
        result_distribution = team_data['RESULT'].value_counts().sort_index()
        y_true_team = pd.to_numeric(team_data['RESULT'], errors='coerce').to_numpy()
        y_pred_team =  pd.to_numeric(team_data['PREDICTED_RESULT'], errors='coerce').to_numpy()
        num_games = len(team_data)
        
        # Berechne die Home-Metriken
        if num_games > 0:
            report = classification_report(y_true_team, y_pred_team, output_dict=True)
            results[team] = {
                'home_games': num_games
            }
            
            results[team]['home_games_win'] = int(result_distribution.get(2, 0))
            results[team]['home_games_draw'] = int(result_distribution.get(1, 0))
            results[team]['home_games_loss'] = int(result_distribution.get(0, 0))
            results[team]['home_accuracy'] = round(accuracy_score(y_true_team, y_pred_team),round_precision)
            results[team]['home_balanced_accuracy'] = round(balanced_accuracy_score(y_true_team, y_pred_team),round_precision)
            results[team]['home_precision_class_0'] = round(report['0']['precision'],round_precision) # Für das jeweilige Team im Heim: Prognose auf Verlust 
            if '1' in report:
                results[team]['home_precision_class_1'] = round(report['1']['precision'],round_precision) # Für das jeweilige Team im Heim: Prognose auf Unentschieden 
            results[team]['home_precision_class_2'] = round(report['2']['precision'],round_precision) # Für das jeweilige Team im Heim: Prognose auf Sieg
            results[team]['home_recall_class_0'] = round(report['0']['recall'],round_precision)
            if '1' in report:
                results[team]['home_recall_class_1'] = round(report['1']['recall'],round_precision)
            results[team]['home_recall_class_2'] = round(report['2']['recall'],round_precision)
            results[team]['home_f1-score_class_0'] = round(report['0']['f1-score'],round_precision)
            if '1' in report:
                results[team]['home_f1-score_class_1'] = round(report['1']['f1-score'],round_precision)
            results[team]['home_f1-score_class_2'] = round(report['2']['f1-score'],round_precision)
        
        else:
            raise Exception("Number of games is 0")
            
        team_data = X_test_decoded[(X_test_decoded['AWAY_TEAM'] == team)]
        result_distribution = team_data['RESULT'].value_counts().sort_index()
        y_true_team = pd.to_numeric(team_data['RESULT'], errors='coerce').to_numpy()
        y_pred_team =  pd.to_numeric(team_data['PREDICTED_RESULT'], errors='coerce').to_numpy()
        num_games = len(team_data)
        
        
        # Berechne die Away-Metriken
        if num_games > 0:
            report = classification_report(y_true_team, y_pred_team, output_dict=True)
            results[team]['away_games'] = num_games
            results[team]['away_games_win'] = int(result_distribution.get(0, 0))  # !Umkehrung der Klasse für Sieg und Niederlage wenn auswärts gespielt wird
            results[team]['away_games_draw'] = int(result_distribution.get(1, 0))
            results[team]['away_games_loss'] = int(result_distribution.get(2, 0)) # !Umkehrung der Klasse für Sieg und Niederlage wenn auswärts gespielt wird
            results[team]['away_accuracy'] = round(accuracy_score(y_true_team, y_pred_team),round_precision)
            results[team]['away_balanced_accuracy'] = round(balanced_accuracy_score(y_true_team, y_pred_team),round_precision)
            if '0' in report:
                results[team]['away_precision_class_0'] = round(report['0']['precision'],round_precision) # Für das jeweilige Team auswärts: Prognose auf Sieg (Auswärtssieg)
            results[team]['away_precision_class_1'] = round(report['1']['precision'],round_precision) # Für das jeweilige Team auswärts: Prognose auf Unentschieden
            results[team]['away_precision_class_2'] = round(report['2']['precision'],round_precision) # Für das jeweilige Team auswärts: Prognose auf Niederlage
            if '0' in report:
                results[team]['away_recall_class_0'] = round(report['0']['recall'],round_precision)
            results[team]['away_recall_class_1'] = round(report['1']['recall'],round_precision)
            results[team]['away_recall_class_2'] = round(report['2']['recall'],round_precision)
            if '0' in report:
                results[team]['away_f1-score_class_0'] = round(report['0']['f1-score'],round_precision)
            results[team]['away_f1-score_class_1'] = round(report['1']['f1-score'],round_precision)
            results[team]['away_f1-score_class_2'] = round(report['2']['f1-score'],round_precision)
            
            results[team] ['games'] = results[team]['home_games'] + results[team] ['away_games']
            results[team] ['games_win'] = results[team]['home_games_win'] + results[team] ['away_games_win']
            results[team] ['games_draw'] = results[team]['home_games_draw'] + results[team] ['away_games_draw']
            results[team] ['games_loss'] = results[team]['home_games_loss'] + results[team] ['away_games_loss']
        else:
            raise Exception("Number of games is 0")
            

    return results


def get_elo_forecast_results(X_test, y_test, label_mapping_TEAM, PATH_ELO_CLUBS, HOME_TEAM_WIN, AWAY_TEAM_WIN, DRAW, le_teams):
    
    total_count = 0
    total_correct = 0
    # Ein DataFrame für die Ausgabe-Ergebnisse
    results_df = pd.DataFrame(columns=['HOME_TEAM', 'AWAY_TEAM', 'RESULT', 'PREDICTED_RESULT'])
    
    for index, row in X_test.iterrows():
        home_team = label_mapping_TEAM[row['HOME_TEAM']]
        away_team = label_mapping_TEAM[row['AWAY_TEAM']]

        if home_team in Not_available or away_team in Not_available:
            continue
        
        elo = elo_rating(PATH_ELO_CLUBS, row['DATE'], home_team, away_team)
        if elo is None:
            raise Exception("No Elo available")
        
        if elo >= 0.51:
            prediction = HOME_TEAM_WIN
        elif elo <= 0.49:
            prediction = AWAY_TEAM_WIN
        else:
            prediction = DRAW
            
        label = y_test.iloc[index]
        total_count += 1
        
        # Gesamtgenauigkeit
        if prediction == label:
            total_correct += 1
            
        new_row = pd.DataFrame({'HOME_TEAM': [row['HOME_TEAM']],
                                'AWAY_TEAM': [row['AWAY_TEAM']],
                                'RESULT': [label],
                                'PREDICTED_RESULT': [prediction]})
        results_df = pd.concat([results_df, new_row]).reset_index(drop=True)
        
    # Berechne die Gesamtgenauigkeit
    print(f"Total Accuracy: {total_correct / total_count}")
    
    # Ausgabe der Genauigkeit und des Klassifikationsberichts für jedes Team
    results = get_team_metrics(results_df[['HOME_TEAM', 'AWAY_TEAM']], results_df['RESULT'], results_df['PREDICTED_RESULT'], le_teams)
    return results


def get_model_forecast_results(opt, X_test, y_test, le_teams, is_optimizer=True):
    
    if is_optimizer:
        model = opt.best_estimator_
    else:
        model = opt
        
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    # print(type(X_test)) # pd dataframe
    # print(type(y_test)) # pd series
    # print(type(y_pred)) # pd numpy
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Recall and F-score are ill-defined and being*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Precision and F-score are ill-defined.*") 
    warnings.filterwarnings("ignore", message="`np.int` is a deprecated alias for the builtin `int`")
    result = get_team_metrics(X_test, y_test, pd.Series(y_pred), le_teams) 
    return result  


def plot_sorted_vertical_bar(df, column_to_sort, base_color='Blues', ascending=False):
    # Sortieren des DataFrames nach der gewünschten Spalte
    df_sorted = df.sort_values(by=column_to_sort, ascending=ascending)
    
    # Erstellen des Balkendiagramms
    plt.figure(figsize=(12, 8))
    
    x_pos = np.arange(len(df_sorted))
    
    # Farbpalette basierend auf dem übergebenen Grundfarbton
    cmap = cm.get_cmap(base_color)
    colors = [cmap(1 - i * 0.5 / len(df_sorted)) for i in range(len(df_sorted))]
    
    bars = plt.bar(x_pos, df_sorted[column_to_sort], color=colors)
    
    # Beschriftungen und Titel hinzufügen
    plt.ylabel(column_to_sort)
    plt.xlabel('Model')
    
    # Titel anpassen
    title = f'Model Performance Based on {column_to_sort}'
    column_lower = column_to_sort.lower()
    if "class_0" in column_lower:
        title += " (away win)"
    elif "class_1" in column_lower:
        title += " (draw)"
    elif "class_2" in column_lower:
        title += " (home win)"
    
    plt.title(title)
    plt.xticks(x_pos, df_sorted['Model'], rotation=45)
    
    # Legende
    #plt.legend(bars, df_sorted['Model'])
    
    # Textbeschriftungen für die Balken hinzufügen
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), va='bottom')  # va: vertical alignment
    
    plt.tight_layout()
    title = title.replace('(','_')
    title = title.replace(')','_')
    plt.savefig(f'./plots/{title}.jpeg')
    plt.show()
    
    
def get_json_results_in_df(directory_path):
    
    list_total = []
    list_clubs = []
    
    # Gehe durch alle Dateien im Verzeichnis
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            # Modellname aus dem Dateinamen extrahieren
            model_name = filename.split('.')[0]
            model_name = model_name.replace("_results", "")
            
            # JSON-Datei lesen
            with open(os.path.join(directory_path, filename), 'r') as f:
                data = json.load(f)
            
            # Extrahieren der "Total_" Einträge
            total_data = {k: v for k, v in data.items() if k.startswith('Total_')}
            total_data['Model'] = model_name  # Spalte für den Modellnamen
            list_total.append(total_data)
            
            # Extrahieren der Club-spezifischen Einträge
            club_data = {k: v for k, v in data.items() if not k.startswith('Total_')}
            for club, metrics in club_data.items():
                metrics['Club'] = club  # Spalte für den Clubnamen
                metrics['Model'] = model_name  # Spalte für den Modellnamen
                list_clubs.append(metrics)

    # Erstellen der DataFrames
    df_total = pd.DataFrame(list_total)
    df_clubs = pd.DataFrame(list_clubs)
    
    return df_total, df_clubs


def plot_best_model_per_club(df, column_to_sort, top_x=10, ascending=False, base_color='Blues', game_columns=None):
    # Gruppieren und Sortieren des DataFrames, und Auswahl der besten x Clubs
    df = df.sort_values(by=[column_to_sort], ascending=ascending)
    df_best_models = df.groupby('Club').first().reset_index()
    df_best_models = df_best_models.sort_values(by=[column_to_sort], ascending=ascending).head(top_x)

    plt.figure(figsize=(15, 10))

    clubs = df_best_models['Club']
    values = df_best_models[column_to_sort]
    models = df_best_models['Model']

    color_values = np.linspace(0.2, 1, top_x)[::-1]  # umgekehrte Farbordnung
    colors = [plt.get_cmap(base_color)(i) for i in color_values]

    bars = plt.bar(clubs, values, color=colors)
    
    for i, bar in enumerate(bars):
        model_name = models.iloc[i]
        club_name = clubs.iloc[i]
        plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height(), f"{model_name}", ha='center', va='bottom')

        # Anzahl der Spiele als Text im Balken
        if game_columns:
            game_count = df_best_models.loc[df_best_models['Club'] == club_name, game_columns].values[0]
            plt.text(bar.get_x() + bar.get_width()/2.0, bar.get_height()/2, f"{game_count} games", ha='center', va='center', color='red', fontsize=14)

    plt.xlabel('Club')
    plt.ylabel(column_to_sort)

    title = f'Top {top_x} Clubs Based on {column_to_sort}'
    if "class_0" in column_to_sort.lower():
        title += " (away win)"
    elif "class_1" in column_to_sort.lower():
        title += " (draw)"
    elif "class_2" in column_to_sort.lower():
        title += " (home win)"
        
    plt.title(title)
    title = title.replace('(','_')
    title = title.replace(')','_')
    plt.savefig(f'./plots/{title}.jpeg')
    plt.show()
        


def get_model_forecast_predictions(opt, X_test, is_optimizer=True):
    
    if is_optimizer:
        model = opt.best_estimator_
    else:
        model = opt
        
    y_pred = model.predict(X_test) 
    return y_pred

