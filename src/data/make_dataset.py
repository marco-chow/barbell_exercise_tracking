import pandas as pd
from glob import glob

#Read a single file

single_file_acc = pd.read_csv('/Users/marco/Desktop/Projects/quantified_self/data/raw/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv')
single_file_gyr = pd.read_csv('/Users/marco/Desktop/Projects/quantified_self/data/raw/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv')

files = glob('/Users/marco/Desktop/Projects/quantified_self/data/raw/MetaMotion/*.csv')
len(files)


data_path = "/Users/marco/Desktop/Projects/quantified_self/data/raw/MetaMotion/"
f = files[0]


acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()
acc_set = 1
gyr_set = 1
#Read all files into dfs

for f in files:
    participant = f.split("-")[0].replace(data_path,"")
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123")
    
    df = pd.read_csv(f)
    
    df['participant'] = participant
    df['label'] = label
    df['category'] = category
    
    if "Accelerometer" in f:
        df['set'] = acc_set
        acc_set +=1
        acc_df = pd.concat([acc_df, df])
         
    elif "Gyroscope" in f:
        df['set'] = gyr_set
        gyr_set +=1
        gyr_df = pd.concat([gyr_df, df])
        
#Converting the timestamps
acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit='ms')


del acc_df["epoch (ms)"]   
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]
del gyr_df["epoch (ms)"]   
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]

 #Creating a function to encapsulate the above steps
def read_data_from_files(files, data_path = "/Users/marco/Desktop/Projects/quantified_self/data/raw/MetaMotion/"):
    
    #initialize dfs
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()
    
    #Set counting initializer
    acc_set = 1
    gyr_set = 1
    
    #Read all files into dfs
    for f in files:
        participant = f.split("-")[0].replace(data_path,"")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123")
        
        df = pd.read_csv(f)
        
        df['participant'] = participant
        df['label'] = label
        df['category'] = category
        
        if "Accelerometer" in f:
            df['set'] = acc_set
            acc_set +=1
            acc_df = pd.concat([acc_df, df])
            
        elif "Gyroscope" in f:
            df['set'] = gyr_set
            gyr_set +=1
            gyr_df = pd.concat([gyr_df, df])
            
    #Converting the timestamps
    acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
    gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit='ms')


    del acc_df["epoch (ms)"]   
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]
    del gyr_df["epoch (ms)"]   
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]
    
    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)


data_merged = pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1)
data_merged.dropna()

#Reset column names
data_merged.columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'participant', 'label', 'category', 'set']



#Resampling done by day, in 200ms intervals

Sampling = {
    'acc_x': "mean", 
    'acc_y': "mean",
    'acc_z': "mean", 
    'gyr_x': "mean", 
    'gyr_y': "mean", 
    'gyr_z': "mean", 
    'participant': "last", 
    'label': "last", 
    'category': "last",
    'set': "last"
}

days = [g for n,g in data_merged.groupby(pd.Grouper(freq='D'))]
data_resampled = pd.concat([df.resample(rule='200ms').apply(Sampling).dropna() for df in days])
data_resampled['set'] = data_resampled['set'].astype('int')
data_resampled.info()

#Save data
data_resampled.to_pickle("/Users/marco/Desktop/Projects/quantified_self/data/interim/01_data_resampled.pkl")
