import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



center_data = pd.read_csv("player_C.csv")
pforward_data = pd.read_csv("player_PF.csv")
sforward_data = pd.read_csv("player_SF.csv")
sguard_data = pd.read_csv("player_SG.csv")
pguard_data = pd.read_csv("player_PG.csv")


cdf = center_data.drop(['Age', 'G', 'GS', 'MP', 'Tm', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'PF'], axis=1)
print(cdf)
feature_weights = []


"""
Column Weights: STARTING WITH CENTERS
Player
Position
Age x
Team x
Games x?
Games Started x?
Minutes Played x?
FG **
FG avg **
FG% **
3P x
3P avg x
3P% *
2P x
2P avg x
2P% ***
eFG% ****
FT x
FTA x
FT% **
ORB x
DRB x
TRB ****
AST **
STL *
BLK ****
TOV *
PF x
PPG ****
"""
