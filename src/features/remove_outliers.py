import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor

df = pd.read_pickle("../../data/interim/01_data_resampled.pkl")

outlier_columns = list(df.columns[:6])
#Boxplots
df[outlier_columns[:3] + ["label"]].boxplot(by="label", layout=(1,3)) #Acc
df[outlier_columns[3:6] + ["label"]].boxplot(by="label", layout=(1,3)) #Gyr

#Histograms
df[outlier_columns[:3] + ["label"]].hist(by="label", figsize=(20,20), layout=(3,3)) #Acc
df[outlier_columns[3:6] + ["label"]].hist(by="label", figsize=(20,20),layout=(3,3)) #Gyr

#Outlier Detection with IQR
def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

        Args:
            dataset (pd.DataFrame): The dataset
            col (string): The column you want apply outlier detection to

        Returns:
            pd.DataFrame: The original dataframe with an extra boolean column 
            indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset

 #Visualize the outliers    
for col in outlier_columns:
        plt.figure()
        df_plot = mark_outliers_iqr(df, col)
        df_plot = df_plot.reset_index()
        sns.scatterplot(data=df_plot, x=df_plot.index, y=col, hue=col+"_outlier")  
        
#Outlier detection function with Chauvenet's criterion        
def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset
 
 #Visualize the outliers      
for col in outlier_columns:
        plt.figure()
        df_plot = mark_outliers_chauvenet(df, col)
        df_plot = df_plot.reset_index()
        sns.scatterplot(data=df_plot, x=df_plot.index, y=col, hue=col+"_outlier")  


#Outlier detection function with Local Outlier Factor
def mark_outliers_lof(dataset, col, n_neighbors=20):
    dataset = dataset.copy()
    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    data = dataset[col]
    outliers = lof.fit_predict(data)

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers


 #Visualize the outliers   
df_lof, outliers = mark_outliers_lof(df, outlier_columns)
df_lof = df_plot.reset_index()   
for col in outlier_columns:
        plt.figure()
        sns.scatterplot(data=df_lof, x=df_lof.index, y=col, hue="outlier_lof") 
        

#Compare the outliers detected by the 3 methods, for a specific exercise
for col in outlier_columns:
    exercise = 'bench'
    df_target = df[df['label']==exercise]
    fig, axes = plt.subplots(3,1, figsize=(10,10));
    
    df_chauf = mark_outliers_chauvenet(df_target, col)
    df_chauf = df_chauf.reset_index()
    
    df_iqr = mark_outliers_iqr(df_target, col)
    df_iqr = df_iqr.reset_index()
    
    df_lof, outliers = mark_outliers_lof(df_target, outlier_columns)
    df_lof = df_lof.reset_index() 
    
    
    plt.figure();
    sns.scatterplot(data=df_chauf, x=df_chauf.index, y=col, hue=col+"_outlier", ax=axes[0],palette={True: "red", False: "blue"}, s=5)  
    axes[0].set_title(f"{exercise} {col} Outliers with Chauvenet's criterion");
    axes[1].set_title(f"{exercise} {col} Outliers with IQR");
    sns.scatterplot(data=df_iqr, x=df_iqr.index, y=col, hue=col+"_outlier",ax=axes[1],palette={True: "red", False: "blue"}, s=5)  
    axes[2].set_title(f"{exercise} {col} Outliers with LOF");
    sns.scatterplot(data=df_lof, x=df_lof.index, y=col, hue="outlier_lof",ax=axes[2],palette={True: "red", False: "blue"}, s=5)  



#Outlier imputation using Chauvenet's criterion
outlier_removed_df = df.copy()

for col in outlier_columns:
   for label in df['label'].unique():
       df_target = mark_outliers_chauvenet(df[df['label']==label], col)
       df_target.loc[df_target[col + "_outlier"], col] = np.nan
       
       outlier_removed_df.loc[outlier_removed_df['label']==label, col]= df_target[col]
       n_outliers = len(df) - len(outlier_removed_df[col].dropna())
       print(f"{n_outliers} of outliers removed from {label} {col}")
       

outlier_removed_df.info()

outlier_removed_df.to_pickle("../../data/interim/02_data_outlier_removed.pkl")

         