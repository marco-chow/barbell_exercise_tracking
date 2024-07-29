import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error
import os

# path = os.getcwd()
# print(path)
# os.chdir("../features")

#Load Data
df = pd.read_pickle("../../data/interim/01_data_resampled.pkl")
df = df[df['label']!='rest']
df['category'] = df['category'].str.replace('_MetaWear_2019', '', regex=False)

acc_r = df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2
gyr_r = df['gyr_x']**2 + df['gyr_y']**2 + df['gyr_z']**2
df['acc_r'] = np.sqrt(acc_r)
df['gyr_r'] = np.sqrt(gyr_r)

#Split data by exercise

bench_df = df[df['label']=='bench']
squats_df = df[df['label']=='squat']
row_df = df[df['label']=='row']
ohp_df = df[df['label']=='ohp']
dead_df = df[df['label']=='dead']

#lowpass filter

fs = 1000/200
LowPass = LowPassFilter()

bench_set = bench_df[bench_df['set']==bench_df['set'].unique()[0]]
squats_set = squats_df[squats_df['set']==squats_df['set'].unique()[0]]
row_set = row_df[row_df['set']==row_df['set'].unique()[0]]
row_set['category'] = row_set['category'].str.replace('_MetaWear_2019', '', regex=False)
ohp_set = ohp_df[ohp_df['set']==ohp_df['set'].unique()[0]]
dead_set = dead_df[dead_df['set']==dead_df['set'].unique()[0]]

col = "acc_r"

bench_processed = LowPass.low_pass_filter(bench_set, col, sampling_frequency=fs, cutoff_frequency=0.4, order = 5)
bench_processed[col+'_lowpass'].plot()

indexes = argrelextrema(bench_processed[col+'_lowpass'].values, np.greater)
peaks = bench_processed.iloc[indexes]

def countreps(dataset, col="acc_r", cutoff=0.4, order=10): 
    data_processed = LowPass.low_pass_filter(dataset, col, sampling_frequency=fs, cutoff_frequency=cutoff, order = order)
    indexes = argrelextrema(data_processed[col+'_lowpass'].values, np.greater)
    peaks = data_processed[col+'_lowpass'].iloc[indexes]
    #Plot
    exercise = dataset.iloc[0]["label"].title()
    weight = dataset.iloc[0]["category"].title()
    #Hiding plots for now
    # plt.figure(figsize=(20,10))
    # plt.plot(data_processed[col+'_lowpass'])
    # plt.plot(peaks, "o", c='r')
    # plt.title(f"{weight} {exercise} Rep Counts: {len(peaks)}")
    # plt.ylabel(f"{col}")
    return len(peaks)

countreps(bench_set, col="acc_r", cutoff=0.4, order=10)
countreps(squats_set, col="acc_r", cutoff=0.4, order=10)
countreps(row_set, col="acc_r", cutoff=0.4, order=10)
countreps(ohp_set, col="acc_r", cutoff=0.4, order=10)
countreps(dead_set, col="acc_r", cutoff=0.4, order=10)


#Create a column for reps with expected number of reps

df['rep'] = df['category'].apply(lambda x:5 if x == "heavy" else 10)
df['category']  = df['category'].apply(lambda x: "heavy" if (x == "heavy1") or (x == "heavy2") else x)
reps_df = df.groupby(['label', 'set','category'])['rep'].max().reset_index()
reps_df['predicted_reps'] = 0

for s in df['set'].unique():
    subset = df.loc[df['set']==s]
    #default values for rep counting
    cutoff = 0.4
    column = 'acc_r'
    
    if subset['label'].iloc[0] == "squat":
        cutoff = 0.35
        
    if subset['label'].iloc[0] == "row":
        cutoff = 0.65
        column = "gyr_x"
    
    if subset['label'].iloc[0] == "dead":
        cutoff = 0.35
        
    reps =  countreps(subset, col=column, cutoff=0.4, order=10)
    reps_df.loc[reps_df['set']==s,'predicted_reps'] = reps
    
    #Error
    
    error = mean_absolute_error(reps_df['rep'], reps_df['predicted_reps']).round(2)
    
    
    #Visualize performance of rep counting for each exercise/weight category
    reps_df.groupby(['label', 'category'])['rep', 'predicted_reps'].mean().plot.bar()
    