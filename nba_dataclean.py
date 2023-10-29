import pandas as pd

#5 years of data
nba_2016 = pd.read_csv('2015_2016NBA.csv', encoding='latin-1')
nba_2017 = pd.read_csv('2016_2017NBA.csv', encoding='latin-1')
nba_2018 = pd.read_csv('2017_2018NBA.csv', encoding='latin-1')
nba_2019 = pd.read_csv('2018_2019NBA.csv', encoding='latin-1')
nba_2020 = pd.read_csv('2019_2020NBA.csv', encoding='latin-1')
nba_2021 = pd.read_csv('2020_2021NBA.csv', encoding='latin-1')
nba_2022 = pd.read_csv('2021_2022NBA.csv', encoding='latin-1')
nba_2023 = pd.read_csv('2022_2023NBA.csv', encoding='latin-1')


data_list = [nba_2019, nba_2020, nba_2021, nba_2022, nba_2023]
stuff = []
#removes duplicates, fills in NaN, Removes unnecessary columns
for i in data_list: 
    players_with_multiple_teams = i[i.duplicated(subset=['Player'], keep=False)]
    filtered_data = i.drop_duplicates(subset=['Player'], keep='first')
    stuff.append(filtered_data.drop(['Rk'], axis=1))
data = pd.concat(stuff)
data = data.fillna(0).sort_values(by='Player')
#Keeps only current players
current_players = data[data['YR'] == 2023]['Player']
still_active_players = data[data['Player'].isin(current_players)]
data = data[data['Player'].isin(still_active_players['Player'])]
#Year fix in data
data['YR'] = data['YR'].astype(int)
data = data[data['PTS'] != 0]
data = data.sort_values(by=['Player', 'YR'])



#Makes a priority for 
def determine_primary_position(row):
    pos = row['Pos']
    if '-' in pos:
        primary_pos = pos.split('-')[0]
    else:
        primary_pos = pos
    row['Pos'] = primary_pos
    return row

data = data.apply(determine_primary_position, axis=1)

position_groups = data.groupby('Pos')
def determine_majority_position(player_data):
    return player_data['Pos'].value_counts().idxmax()

for player in data['Player'].unique():
    player_data = data[data['Player'] == player]
    majority_pos = determine_majority_position(player_data)
    
    data.loc[data['Player'] == player, 'Pos'] = majority_pos

print(data.to_csv('current_players.csv'))

position_groups = data.groupby('Pos')
for position, group in position_groups:
    print(f"Group for Most Popular Position {position}:")
    print(group)
    print("\n")

    filename = f"player_{position}.csv"
    group.to_csv(filename, index=False)
    print(f"CSV file saved for Most Popular Position {position}: {filename}")
