

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import re
import unicodedata
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

BASE_ELO = 1500
K = 32


excel_file = 'Input your file path for xlsx here.'


xls = pd.read_excel(excel_file, sheet_name=None)


def clean_name(name):
    if pd.isna(name):
        return None
    name = str(name)
    name = unicodedata.normalize('NFKD', name)
    name = re.sub(r'\s+', ' ', name)
    name = name.strip()
    return name


all_players_set = set()
for sheet_name, df in xls.items():
    df.columns = df.columns.str.strip().str.lower()
    if 'winner' in df.columns and 'loser' in df.columns:
        winners = df['winner'].dropna().apply(clean_name)
        losers = df['loser'].dropna().apply(clean_name)
        all_players_set.update(winners)
        all_players_set.update(losers)
all_players_list = list(all_players_set)

elo_ratings = {player: BASE_ELO for player in all_players_list}

def expected_score(p1, p2):
    return 1 / (1 + 10 ** ((p2 - p1) / 400))

def update_elo(winner, loser, K=32):
    w_elo = elo_ratings[winner]
    l_elo = elo_ratings[loser]

    expected_win = expected_score(w_elo, l_elo)
    expected_loss = expected_score(l_elo, w_elo)

    new_w_elo = w_elo + K * (1 - expected_win)
    new_l_elo = l_elo + K * (0 - expected_loss)

    elo_ratings[winner] = new_w_elo
    elo_ratings[loser] = new_l_elo

    return new_w_elo, new_l_elo

elo_winners = []
elo_losers = []
match_records = []

for sheet_name, df in xls.items():
    df.columns = df.columns.str.strip().str.lower()
    if 'winner' in df.columns and 'loser' in df.columns and 'date' in df.columns:
        df = df.dropna(subset=['winner', 'loser', 'date'])
        df['winner'] = df['winner'].apply(clean_name)
        df['loser'] = df['loser'].apply(clean_name)
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df = df.sort_values('date').reset_index(drop=True)

        for i, row in df.iterrows():
            w = row['winner']
            l = row['loser']
            if w in elo_ratings and l in elo_ratings:
                w_elo, l_elo = update_elo(w, l)
                elo_winners.append(w_elo)
                elo_losers.append(l_elo)
                match_records.append([w, l, row['date'], w_elo, l_elo])
all_matches = pd.DataFrame(match_records, columns=['Winner', 'Loser', 'Date', 'Winner_ELO', 'Loser_ELO'])

with open('elo_ratings.pkl', 'wb') as f:
    pickle.dump(elo_ratings, f)

all_matches['Fav_ELO'] = all_matches[['Winner_ELO', 'Loser_ELO']].max(axis=1)
all_matches['ELO_Diff'] = abs(all_matches['Winner_ELO'] - all_matches['Loser_ELO'])
all_matches['Fav_Won'] = all_matches['Winner_ELO'] > all_matches['Loser_ELO']

X = all_matches[['ELO_Diff']]
y = all_matches['Fav_Won'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

def predict_match(p1, p2):
    p1 = clean_name(p1)
    p2 = clean_name(p2)

    p1_elo = elo_ratings.get(p1, BASE_ELO)
    p2_elo = elo_ratings.get(p2, BASE_ELO)

    diff = abs(p1_elo - p2_elo)
    input_df = pd.DataFrame({'ELO_Diff': [diff]})

    prob = float(model.predict_proba(input_df)[0][1])

    print(f"{p1} Elo: {p1_elo}, {p2} Elo: {p2_elo}")
    print(f"Favorite win probability: {round(prob * 100, 2)}%")
    return prob

def calc_ev(model_prob, bookmaker_odds):
    payout = bookmaker_odds - 1
    ev = (model_prob * payout) - (1 - model_prob)
    return ev


player_1 = "Ofner S."
player_2 = "Shelton B."  
bookmaker_odds = 3.4

model_prob = predict_match(player_1, player_2)
ev = calc_ev(model_prob, bookmaker_odds)
print(f"Expected Value (EV): {ev}")
if ev > 0:
    print("This is a +EV (good) bet!")
else:
    print("This bet is not recommended.")
