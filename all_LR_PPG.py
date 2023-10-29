import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load your data
data_PG = pd.read_csv("player_PG.csv")
data_SG = pd.read_csv("player_SG.csv")
data_SF = pd.read_csv("player_SF.csv")
data_PF = pd.read_csv("player_PF.csv")
data_C = pd.read_csv("player_C.csv")

df = [data_PG, data_SG, data_SF, data_PF, data_C]  # Your list of dataframes

all_data = pd.concat(df, axis=0)
all_data = all_data.drop(['Pos', 'Age', 'G', 'GS', 'MP', 'Tm', 'ORB', 'DRB', 'PF'], axis=1)

# Separate the data into training and test sets
X_train = all_data[all_data['YR'] < 2023].drop(columns=['Player', 'PTS'])
y_train = all_data[all_data['YR'] < 2023]['PTS']

X_test = all_data[all_data['YR'] == 2023].drop(columns=['Player', 'PTS'])
y_test = all_data[all_data['YR'] == 2023]['PTS']

model = LinearRegression()
model.fit(X_train, y_train)

# You can print out the MSE for 2023 data here
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE for 2023 data: {mse}")

# Predict the points for next season using the trained models
all_predicted_points_next_season = {}
for player in all_data['Player'].unique():
    latest_data = all_data[all_data['Player'] == player].iloc[-1]
    X_latest = pd.DataFrame([latest_data.drop(labels=['Player', 'PTS'])], columns=X_train.columns)
    
    predicted_points = model.predict(X_latest)[0]
    all_predicted_points_next_season[player] = round(predicted_points, 2)

# After finishing all data frames, sort the predictions and print
all_predicted_points_next_season = dict(sorted(all_predicted_points_next_season.items(), key=lambda item: item[1], reverse=True))
print(all_predicted_points_next_season)

top_50_predicted = dict(sorted(all_predicted_points_next_season.items(), key=lambda item: item[1], reverse=True)[:50])

players = list(top_50_predicted.keys())
points = list(top_50_predicted.values())

plt.figure(figsize=(10, 15))
plt.barh(players, points, color='skyblue')
plt.xlabel('Predicted Points for Next Season')
plt.ylabel('Players')
plt.title('Top 50 Predicted Player Points for Next Season')
plt.gca().invert_yaxis()

for index, value in enumerate(points):
    plt.text(value, index, str(value))  

plt.show()
