import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data_PG = pd.read_csv("player_PG.csv")
data_SG = pd.read_csv("player_SG.csv")
data_SF = pd.read_csv("player_SF.csv")
data_PF = pd.read_csv("player_PF.csv")
data_C = pd.read_csv("player_C.csv")

df = [data_PG, data_SG, data_SF, data_PF, data_C]  # Your list of dataframes

# Dictionary for putting predicted points for next season
all_predicted_points_next_season = {}

for data in df:
    data = data.drop(['Pos', 'Age', 'G', 'GS', 'MP', 'Tm', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'PF'], axis=1)

    # Create a dictionary to store player models to be put in dictionary later
    player_models = {}

    # setting x and y for a prediction on points per game
    for player in data['Player'].unique():
        player_data = data[data['Player'] == player] #Get rid of any string datatypes that are unnecessary
        data = data.sort_values(by=['Player', 'YR'])

        # Check if player has played more than 1 year
        if len(player_data) <= 1:
            print(f"Skipping {player} as they've played only for one year.")
            continue

        X = player_data.drop(columns=['Player', 'PTS'])
        y = player_data['PTS']
        
        # Train test split
        train_size = int(0.8 * len(player_data))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        model = LinearRegression()
        model.fit(X_train, y_train)

        player_models[player] = model
    for player, model in player_models.items():
            # Extract the most recent season's data for the player
            latest_data = data[data['Player'] == player].iloc[-1]
            
            # Drop non-feature columns (Player and PTS) and ensuring it's a DataFrame
            X_latest = pd.DataFrame([latest_data.drop(labels=['Player', 'PTS'])], columns=X_train.columns)

            # Predict the points for next season 
            predicted_points = model.predict(X_latest)[0]

            # Store the prediction in a dictionary, combining all positions
            all_predicted_points_next_season[player] = round(predicted_points, 2)

# After finishing all data frames, sort the combined predictions and print
all_predicted_points_next_season = dict(sorted(all_predicted_points_next_season.items(), key=lambda item: item[1], reverse=True))

print(all_predicted_points_next_season)

top_50_predicted = dict(sorted(all_predicted_points_next_season.items(), key=lambda item: item[1], reverse=True)[:50])

players = list(top_50_predicted.keys())
points = list(top_50_predicted.values())

plt.figure(figsize=(10, 15)) 

plt.barh(players, points, color='red')
plt.xlabel('Predicted Points for Next Season')
plt.ylabel('Players')
plt.title('Top 50 Predicted Player Points for Next Season')
plt.gca().invert_yaxis()

for index, value in enumerate(points):
    plt.text(value, index, str(value))  

plt.show()


