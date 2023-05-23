import pickle
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

st.set_page_config(
    page_title= 'Football Improvements Radar Plots',
    page_icon= '✅',
    layout= 'wide',
    )

#open the pickle file
generator_loaded = pickle.load(open('C:/Users/crmch/OneDrive/Desktop/College/Year 4/Semester 2/Project 2/GAN/generator.pkl', 'rb'))

#open the csv file for plotting
#data = pd.read_csv('C:/Users/crmch/OneDrive/Desktop/College/Year 4/Semester 2/Project 2/Datasets/portugalFinal.csv', sep = ',')
data_read = pd.read_csv('C:/Users/crmch/OneDrive/Desktop/College/Year 4/Semester 2/Project 2/Datasets/FootballLabel.csv', sep = ',')

data = data_read[data_read['label'] != 2]

data = data.drop('label', axis = 1)

#set the generator and the scaler to variables
generator = generator_loaded['generator']
scaler = generator_loaded['scaler']

team_list = data['Squad'].to_list()

#sidebar for variable selection
selected_team = st.sidebar.selectbox('Select Team', team_list)

#assign a team's data to the variable
team = data[data['Squad'] == selected_team]
team_name = team['Squad'].values[0]

#dashboard title
st.title(f'Original Data v Improvements - {team_name}')

#drop the categorical values
team = team.drop(['Squad', 'League'], axis = 1)
data_read = data_read.drop(['Squad', 'League', 'label'], axis = 1)

#scale the validation data using the scaler from the training data
val_check = scaler.transform(team)
syn_val_df = pd.DataFrame(val_check)

tf_val = tf.convert_to_tensor(syn_val_df)
#generate synthetic data using the generator model
synthetic_data_scaled = generator.predict(tf_val)

#inverse the scaling to get the unscaled synthetic data and convert to dataframe to show the results
inver = scaler.inverse_transform(synthetic_data_scaled)
val_df = pd.DataFrame(inver, columns=team.columns)

df = pd.concat([team, val_df], ignore_index = True)
df = df.rename(index={0: 'Original Data', 1: 'Improved Data'})

df.loc['Difference'] = df.loc['Improved Data'] - df.loc['Original Data']

#define the column names for the radar plot
categories = team.columns.tolist()
misc = categories[0:7]
gk = categories[7:16]
shots = categories[16:21]
passing = categories[21:40]
scga = categories[40:46]
defence = categories[46:59]
touches = categories[59:76]

category_list = [misc, gk, shots, passing, scga, defence, touches]

#sidebar for variable selection
category_names = ['Miscallaneous', 'Goalkeeping', 'Shooting', 'Passing', 'Shot/Goal Creating Actions', 'Defence', 'Touches']
selected_category_index = st.sidebar.selectbox('Select Category', range(len(category_list)), format_func=lambda x: category_names[x])
selected_category = category_list[selected_category_index]

#extract the values for the original data and the synthetic data
row1_values = team.loc[:, selected_category].values
row2_values = val_df.loc[:, selected_category].values
maximum = data_read.loc[:, selected_category].describe()[7:8]

#concatenate the rows vertically
values = np.concatenate((row1_values, row2_values, maximum), axis=0)

row1_values_norm = (values[0] / values[2]) * 100
row2_values_norm = (values[1] / values[2]) * 100

df1 = pd.DataFrame(row1_values_norm).T
df2 = pd.DataFrame(row2_values_norm).T

values_norm = np.concatenate((df1, df2), axis=0)
#assign the number of categories to a variable
num_categories = len(selected_category)

#compute the angle for each category and send to a list
angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()

#make the plot circular by appending the first angle at the end
angles += angles[:1]
angles = angles[:-1]

#repeat the first value at the end to complete the radar plot
values_norm = np.concatenate((values_norm, [values_norm[0]]), axis=0)

#create a figure and a polar subplot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

#plot the values for the original data
ax.plot(angles, values_norm[0], label='Original')

#plot the values for the synthetic data
ax.plot(angles, values_norm[1], label='Improvements')

#fill the area between the two lines with colors to show the differences
ax.fill(angles, values_norm[0], alpha=0.5)
ax.fill(angles, values_norm[1], alpha=0.5)
ax.legend(loc='upper right')

#set the tick labels and angles for the categories
ax.set_xticks(angles)
ax.set_xticklabels(selected_category, rotation=45, ha='right')
ax.set_yticklabels([])

st.pyplot(fig)

# show the dataframe 
st.markdown('### Detailed Data View')
st.dataframe(df)