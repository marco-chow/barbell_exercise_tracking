import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib as mpl

df = pd.read_pickle("/Users/marco/Desktop/Projects/quantified_self/data/interim/01_data_resampled.pkl")

#Plot data of a set

set_df = df.loc[df["set"]==1]
plt.plot(set_df['acc_y'].reset_index(drop=True))
plt.title("Acc_y")

#Plotting Exercises in a loop

for label in df.label.unique():
    subset = df.loc[df["label"]==label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]['acc_y'].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

#Plot Settings

mpl.rcParams['figure.figsize'] = (20, 5)
mpl.rcParams['figure.dpi'] = 100
mpl.style.use('seaborn-deep')
mpl.rcParams['axes.grid'] = False    


#compare medium vs heavy sets

category_df = df.query("label=='squat'").query("participant=='A'").reset_index()

fig, ax = plt.subplots()
category_df.groupby(['category'])['acc_y'].plot()
ax.set_xlabel("Samples")
ax.set_ylabel("Acc_Y")
plt.legend()

#compare Bench data

participant_df = df.query("label=='bench'").sort_values("participant").reset_index()

fig, ax = plt.subplots()
participant_df.groupby(['participant'])['acc_y'].plot()
ax.set_xlabel("Samples")
ax.set_ylabel("Acc_Y")
plt.legend()

#Plot 3D Acceleromater data


for label in df.label.unique():
    for participant in df.participant.unique():
        all_axis_df = df.query(f"label=='{label}'").query(f"participant=='{participant}'").reset_index()
        
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax)
            ax.set_xlabel("Samples")
            ax.set_ylabel("Acc")
            plt.title(f"{label}({participant})")
            plt.legend() 
            
            
#Plot 3D gyro and acc for each participant and exercise
for label in df.label.unique():
    for participant in df.participant.unique():
        all_axis_df = df.query(f"label=='{label}'").query(f"participant=='{participant}'").reset_index()
        
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            all_axis_df[['acc_x', 'acc_y', 'acc_z']].plot(ax=ax[0])
            all_axis_df[['gyr_x', 'gyr_y', 'gyr_z']].plot(ax=ax[1])
            ax[0].legend(loc='upper center', ncol=3,  bbox_to_anchor=(0.5,1.15), fancybox=True, shadow=True)
            ax[1].legend(loc='upper center',ncol=3, bbox_to_anchor=(0.5,1.15), fancybox=True, shadow=True)
            ax[1].set_xlabel("Samples")
            ax[0].set_title(f"{label}({participant})")
            
            plt.savefig(f"/Users/marco/Desktop/Projects/quantified_self/reports/figures/{label.title()}({participant}).png")
    
    
for label in df.label.unique():
    print(label, label.title())
   