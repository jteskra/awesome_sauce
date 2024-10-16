import pandas as pd
import joblib
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('nba.keras')
# st.write(model.summary())
st.title('NBA Moneyline Model')

st.sidebar.write('**Please fill out the criteria to get a result**')
st.info('**Machine learning models must use numbers to make predictions. So we must convert teams into numbers and keep them  consistant.**')

team1_decimal_odds = st.sidebar.slider('team 1 decimal odds', .1, 12.1, 6.2)
team2_decimal_odds = st.sidebar.slider('team 2 decimal odds', .1, 12.1, 6.2)
decimal_delta = st.sidebar.slider('odds delta (absolute value of the difference between the odds)', .1, 12.1, 6.2)
delta_points_per_game = st.sidebar.slider('points delta (favorite-underdog)', -4.0, 4.0, 0.0, step=0.1)
delta_assists_per_game = st.sidebar.slider('assists delta (favorite-underdog)', -4.0, 4.0, 0.0, step=0.1)
delta_free_throw_percentage = st.sidebar.slider('free throw percentage delta (favorite-underdog)', -4.0, 4.0, 0.0, step=0.1)
team1_encoded = st.sidebar.slider('team 1 encoded', 0, 30, 1)
favorite_encoded = st.sidebar.slider('favorite encoded', 0, 30, 1)
visitor_encoded = st.sidebar.slider('visitor team encoded', 0, 30, 1)

with st.expander('**Why am I entering these random numbers?**'):
    st.write('We found that these exact parameters yeilded the highest accuracy on unseen data AKA: val_accuracy. It may seem odd and counterproductive to enter some of these values, but all of these columns have positive coeffiences for correlation.')
    st.write('Taking the difference between two teams, rather than using each teams general statistic yeilded 5 percent higher val_accuracy and the log_loss dropped around .3')
    st.write('Achieving ~70 percent accuracy is remarkably difficult when trying to predict something with extremely high randomness, we apologize for the inconvience. Some things we just cannot get around :(')

# with st.expander('**Data for Deltas**'):
#     st.markdown("[Click for NBA Odds](https://www.espn.com/nba/odds)")
#     st.write('**Using 2024 Stats for the beginning part of the 2025 NBA season could yeild better accuracy; just make sure the starters are the same from 2024 to 2025**')
#     st.markdown("[Click for 2024 NBA Stats](https://www.basketball-reference.com/leagues/NBA_2024.html)")
#     st.markdown("[Click for 2025 NBA Stats](https://www.basketball-reference.com/leagues/NBA_2025.html)")


# Use st.expander to group the content
with st.expander('**Data for Deltas**'):
    st.markdown("[Click for NBA Odds](https://www.espn.com/nba/odds)")
    st.write('**Using 2024 Stats for the beginning part of the 2025 NBA season could yeild better accuracy; just make sure the starters are the same from 2024 to 2025. Just change the 2025 in the URL to 2024!**')

    
    # Dictionary mapping team names to URLs (you'll need to replace the URLs with the actual ones)
    teams_urls = {
        'Atlanta Hawks': 'https://www.basketball-reference.com/teams/ATL/2024.html',
        "Boston Celtics": 'https://www.basketball-reference.com/teams/BOS/2024.html', 
        "Brooklyn Nets": 'https://www.basketball-reference.com/teams/BRK/2024.html', 
        "Charlotte Hornets": 'https://www.basketball-reference.com/teams/CHA/2024.html', 
        "Chicago Bulls": 'https://www.basketball-reference.com/teams/CHI/2024.html',
        'Cleveland Cavaliers': 'https://www.basketball-reference.com/teams/CLE/2024.html',
        'Dallas Mavericks': 'https://www.basketball-reference.com/teams/DAL/2024.html',
        'Denver Nuggets': 'https://www.basketball-reference.com/teams/DEN/2024.html',
        'Detroit Pistons': 'https://www.basketball-reference.com/teams/DET/2024.html',
        'Golden State Warriors': 'https://www.basketball-reference.com/teams/GSW/2024.html',
        'Houston Rockets': 'https://www.basketball-reference.com/teams/HOU/2024.html',
        'Indiana Pacers': 'https://www.basketball-reference.com/teams/IND/2024.html',
        'Los Angeles Clippers': 'https://www.basketball-reference.com/teams/LAC/2024.html',
        'Los Angeles Lakers': 'https://www.basketball-reference.com/teams/LAL/2024.html',
        'Memphis Grizzlies': 'https://www.basketball-reference.com/teams/MEM/2024.html',
        'Miami Heat': 'https://www.basketball-reference.com/teams/MIA/2024.html',
        'Milwaukee Bucks': 'https://www.basketball-reference.com/teams/MIL/2024.html',
        'Minnesota Timberwolves': 'https://www.basketball-reference.com/teams/MIN/2024.html',
        'New Orleans Pelicans': 'https://www.basketball-reference.com/teams/NOP/2024.html',
        'New York Knicks': 'https://www.basketball-reference.com/teams/NYK/2024.html',
        'OKC Thunder': 'https://www.basketball-reference.com/teams/OKC/2024.html',
        'Orlando Magic': 'https://www.basketball-reference.com/teams/ORL/2024.html',
        'Philadelphia 76ers': 'https://www.basketball-reference.com/teams/PHI/2024.html',
        'Phoenix Suns': 'https://www.basketball-reference.com/teams/PHO/2024.html',
        'Portland Trail Blazers': 'https://www.basketball-reference.com/teams/POR/2024.html',
        'Sacramento Kings': 'https://www.basketball-reference.com/teams/SAC/2024.html',
        'San Antonio Spurs': 'https://www.basketball-reference.com/teams/SAS/2024.html',
        'Toronto Raptors': 'https://www.basketball-reference.com/teams/TOR/2024.html',
        'Utah Jazz': 'https://www.basketball-reference.com/teams/UTA/2024.html',
        'Washington Wizards': 'https://www.basketball-reference.com/teams/WAS/2024.html',
    }
    
    # Loop through team names and their corresponding URLs
    for team, url in teams_urls.items():
        col1, col2 = st.columns([3, 3])  # Two equal width columns
        col1.write(team)  # Display team name on the left
        col2.markdown(f"[2025 Team Data]({url})", unsafe_allow_html=True)  # Create clickable link on the right


with st.expander('**Teams Encoded**'):
        # Loop through team names and numbers to create a two-column layout
    teams_encoded = {
        'Atlanta Hawks': 0,
        "Boston Celtics": 1, 
        "Brooklyn Nets": 2, 
        "Charlotte Hornets": 3, 
        "Chicago Bulls": 4,
        'Cleveland Cavaliers': 5,
        'Dallas Mavericks': 6,
        'Denver Nuggets': 7,
        'Detroit Pistons': 8,
        'Golden State Warriors': 9,
        'Houston Rockets': 10,
        'Indiana Pacers': 11,
        'Los Angeles Clippers': 12,
        'Los Angeles Lakers': 13,
        'Memphis Grizzlies': 14,
        'Miami Heat': 15,
        'Milwaukee Bucks': 16,
        'Minnesota Timberwolves': 17,
        'New Orleans Pelicans': 18,
        'New_York Knicks':19,
        'OKC Thunder': 20,
        'Orlando Magic':21,
        'Philadelphia 76ers':22,
        'Phoenix Suns':23,
        'Portland Trail Blazers': 24,
        'Sacramento Kings': 25,
        'San Antonio Spurs': 26,
        'Toronto Raptors': 28,
        'Utah Jazz': 29,
        'Washington Wizards': 30,


        # Add more teams as needed
    }
    
    for team, code in teams_encoded.items():
        col1, col2 = st.columns([3, 1])  # Create two columns, left wider than right
        col1.write(team)  # Display team name on the left
        col2.write(code)  # Display encoded number on the right
    # st.write('Boston = 1, Brooklyn = 2, Charlotte = 3, Chicago = 4,')


# Create an expander section
with st.expander('**American Odds to Decimal Odds Converter**'):
    
    # Function to convert American odds to Decimal odds
    def american_to_decimal(american_odds):
        if american_odds > 0:
            # For positive American odds
            decimal_odds = (american_odds / 100) + 1
        else:
            # For negative American odds
            decimal_odds = (100 / abs(american_odds)) + 1
        return decimal_odds

    # Create an input box for the user to enter American odds
    american_odds = st.number_input("Enter American odds:", value=100)  # Default value set to 100

    # Button to perform conversion
    if st.button("Convert to Decimal Odds"):
        # Perform the conversion
        decimal_odds = american_to_decimal(american_odds)
        
        # Display the result
        st.write(f"American odds: {american_odds} -> Decimal odds: {decimal_odds}")



with st.expander('**Calculator for Deltas**'):
    # User inputs for the calculator
    num1 = st.number_input("Enter number for the **Favorite**", value=0)
    num2 = st.number_input("Enter number for the **Underdog**", value=0)

    # Dropdown for selecting the operation
    operation = st.selectbox("Choose an operation", ["Add", "Subtract", "Multiply", "Divide"])

    # Perform calculation based on the selected operation
    if operation == "Add":
        result = num1 + num2
    elif operation == "Subtract":
        result = num1 - num2
    elif operation == "Multiply":
        result = num1 * num2
    elif operation == "Divide":
        if num2 != 0:
            result = num1 / num2
        else:
            result = "Cannot divide by zero"

    # Display the result
    st.write("Result:", result)




#columns
#'team1_odds', 'odds_delta', 'delta_points_per_game', 'delta_assists_per_game', 
#'delta_free_throw_percentage', 'team1_encoded', 'favorite_encoded', 'visitor_encoded'

#'team1_decimal_odds', 'team2_decimal_odds', 'decimal_delta', 'delta_points_per_game',
# 'delta_assists_per_game','delta_free_throw_percentage', 'team1_encoded', 'favorite_encoded', 'visitor_encoded'

submit = st.sidebar.button('**Submit**')  # ,

if submit:
    # print(petal_width, sepal_length)
    data_dict = {'team1_decimal_odds':team1_decimal_odds, 'team2_decimal_odds':team2_decimal_odds, 'decimal_delta':decimal_delta, 'delta_points_per_game': delta_points_per_game, 
    'delta_assists_per_game':delta_assists_per_game,'delta_free_throw_percentage':delta_free_throw_percentage, 
    'team1_encoded':team1_encoded, 'favorite_encoded': favorite_encoded,'visitor_encoded': visitor_encoded}


    # Convert to DataFrame
    sample_df = pd.DataFrame(data_dict, index=[0])

    # Convert DataFrame to NumPy array to feed into the model
    input_array = sample_df.to_numpy()

    #Display 
    st.write(sample_df)

    print(sample_df)

    #perform the predictions
    prediction =model.predict(input_array)

    #Display the prediction
    # st.write(prediction)

    # Assuming prediction is a probability between 0 and 1, convert it to percentage
    prediction_percent = prediction[0][0] * 100  # Multiply by 100 to convert to percentage

    # Display the custom message with the prediction
    st.write(f"The favorite has a {prediction_percent:.2f}% chance of winning the game")
    print(prediction)
