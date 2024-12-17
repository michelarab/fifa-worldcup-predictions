import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LinearRegression

def fifa_rank_impute(data, target, fifa_rank_column):
    # Identify indices with missing target values
    missing_indices = data[data[target].isnull()].index
    
    if len(missing_indices) > 0:
        # Drop rows with missing FIFA rank or target values
        complete_data = data.dropna(subset=[fifa_rank_column, target])
        
        if complete_data.empty:
            return
        
        # Fit the regression model
        model = LinearRegression()
        model.fit(complete_data[[fifa_rank_column]], complete_data[target])
        
        # Impute each missing target value individually
        for idx in missing_indices:
            fifa_rank_value = data.loc[idx, fifa_rank_column]
            # Check if FIFA rank is available for this row
            if pd.notnull(fifa_rank_value):
                # Predict and impute the missing target value
                predicted_value = model.predict([[fifa_rank_value]])[0]
                data.at[idx, target] = predicted_value


def make_running_average(df):
    """ 
    Modifies a dataframe in place, expects specific columns, home_team, away_team, average_home_score, average_away_score
    """

    home_name_list = df['home_team'].unique()
    away_name_list = df['away_team'].unique()

    df['average_home_score'] = 0.0
    df['average_away_score'] = 0.0

    home_score_dict = {k:v for k,v in zip(home_name_list, np.zeros(len(home_name_list)))}
    away_score_dict = {k:v for k,v in zip(away_name_list, np.zeros(len(away_name_list)))}
    home_counter_list = {k: 0 for k in home_name_list}
    away_counter_list = {k: 0 for k in away_name_list}

    for index, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        result = row['home_team_result']

        home_score_dict[home_team] += row['home_team_score']
        home_counter_list[home_team] += 1.0

        away_score_dict[away_team] += row['away_team_score']
        away_counter_list[away_team] += 1.0

        df.at[index, 'average_home_score'] = home_score_dict[home_team] / home_counter_list[home_team]
        df.at[index, 'average_away_score'] = away_score_dict[away_team] / away_counter_list[away_team]
    
    return df

def calculate_default_elo(ranking):
    """Calculate default Elo based on team's ranking."""
    return 1600 - (ranking - 1) * 4

def calculate_expected(elo_team, elo_opponent):
    """Calculate the expected result using the Elo formula."""
    return 1 / (1 + 10 ** ((elo_opponent - elo_team) / 600))