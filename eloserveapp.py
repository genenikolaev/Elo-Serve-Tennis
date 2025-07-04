import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import re
from sklearn.ensemble import RandomForestClassifier


BASE_ELO = 1500
K = 32

@st.cache_resource
def process_data(uploaded_file):
    xls = pd.read_excel(uploaded_file, sheet_name=None)
    elo_ratings = {}
    match_records = []

    for sheet_name, df in xls.items():
        df.columns = df.columns.str.strip().str.lower()
        if all(col in df.columns for col in ['winner', 'loser', 'date']):
            df = df.dropna(subset=['winner', 'loser', 'date'])
            df['winner'] = df['winner'].apply(clean_name)
            df['loser'] = df['loser'].apply(clean_name)
            df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
            df = df.sort_values('date').reset_index(drop=True)

            for _, row in df.iterrows():
                w = row['winner']
                l = row['loser']
                if w and l:
                    updated_w_elo, updated_l_elo = update_elo(w, l, elo_ratings)

                    match_records.append([
                        w, l, row['date'],
                        updated_w_elo, updated_l_elo
                    ])

    matches_df = pd.DataFrame(match_records, columns=['Winner', 'Loser', 'Date', 'Winner_ELO', 'Loser_ELO'])
    matches_df['Fav_ELO'] = matches_df[['Winner_ELO', 'Loser_ELO']].max(axis=1)
    matches_df['ELO_Diff'] = abs(matches_df['Winner_ELO'] - matches_df['Loser_ELO'])
    matches_df['Fav_Won'] = matches_df['Winner_ELO'] > matches_df['Loser_ELO']

    return elo_ratings, matches_df


st.title("🎾 Elo Serve Tennis Predictor (Live Build from Excel)")

uploaded_file = st.file_uploader("Upload your Excel file (.xlsx)", type=["xlsx"])
if uploaded_file is not None:
    elo_ratings, match_df = process_data(uploaded_file)
    model = train_model(match_df)

    player1 = st.text_input("Player 1 Name (e.g. 'Shelton B.')")
    player2 = st.text_input("Player 2 Name (e.g. 'Ofner S.')")

    odds_p1 = st.number_input("Bookmaker Odds for Player 1", min_value=1.01)
    odds_p2 = st.number_input("Bookmaker Odds for Player 2", min_value=1.01)

    if st.button("Predict"):
        try:
            prob, elo1, elo2, fav = predict_match(player1, player2, elo_ratings, model)
            underdog = player2 if fav == player1 else player1
            st.subheader("📊 Win Probabilities")
            st.markdown(f"**{player1} Elo:** {elo1:.0f}")
            st.markdown(f"**{player2} Elo:** {elo2:.0f}")
            st.markdown(f"**Predicted Favorite:** {fav}")
            st.markdown(f"**{fav} win probability:** {prob * 100:.2f}%")
            st.markdown(f"**{underdog} win probability:** {(1 - prob) * 100:.2f}%")

            st.subheader("💸 Expected Value (EV)")
            ev1 = calc_ev(prob if fav == player1 else 1 - prob, odds_p1)
            ev2 = calc_ev(prob if fav == player2 else 1 - prob, odds_p2)

            if ev1 > 0:
                st.success(f"{player1} is a +EV bet! (EV = {ev1:.3f})")
            if ev2 > 0:
                st.success(f"{player2} is a +EV bet! (EV = {ev2:.3f})")
            if ev1 <= 0 and ev2 <= 0:
                st.warning("No value detected.")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Please upload your Excel file to begin.")
