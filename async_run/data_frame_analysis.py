import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

output_dfs = pd.read_csv("/home/jun/HVAC/energy-plus-DRL/Dataframes/dataframes_output_model.csv")
baseline_dfs = pd.read_csv("/home/jun/HVAC/energy-plus-DRL/Dataframes/dataframes_output_test.csv")
variable = 'ppd'


#Take the rows with Datetime containing the value 1984-01-01
#output_dfs = output_dfs[output_dfs['Datetime'].str.contains('1983-12')]

#Drop duplicates of the Datetime column
#output_dfs = output_dfs.drop_duplicates(subset='Datetime', keep='first')
#reset the index
#output_dfs = output_dfs.reset_index(drop=True)


"""
Calculate:
- Average
- Variance, violin plot

"""


y_axis = output_dfs[variable]
y_baseline = baseline_dfs[variable]



# Create a violin plot using matplotlib
fig, ax = plt.subplots()
vplot1 = ax.violinplot(y_axis, points=200, vert=False, widths=1.1,
                     showmeans=True, showextrema=True, showmedians=False,
                     bw_method=0.5)
vplot2 = ax.violinplot(y_baseline, points=200, vert=False, widths=1.1,
                     showmeans=True, showextrema=True, showmedians=False,
                     bw_method=0.5)

ax.set_title('Fan power usage over December (Comfort reward, ext. Fan)')
ax.set_xlabel('Power (Watts)')
ax.set_ylabel('Density')

model_mean = np.mean(y_axis)
baseline_mean = np.mean(y_baseline)
model_label = f"{'RL Agent (mean: '}{model_mean:.2f}{')'}"
baseline_labe = f"{'Baseline controller (mean: '}{baseline_mean:.2f}{')'}"
# Add a legend
ax.legend(handles=[vplot1["bodies"][0], vplot2["bodies"][0]], labels=[model_label, baseline_labe])



# Show the plot
plt.show()



# -- Plot Results --
#plot with date in the x-axis
fig, ax = plt.subplots()
#output_dfs.plot(x='Datetime', y=variable, use_index=True, ax=ax)
#output_dfs.plot(x='Datetime', y='deck_temp', use_index=True, ax=ax)
#output_dfs.plot(x='Datetime', y='post_deck_temp', use_index=True, ax=ax)
output_dfs.plot(x='Datetime', y=variable, use_index=True, ax=ax)
baseline_dfs.plot(x='Datetime', y=variable, use_index=True, ax=ax)
plt.title('Fan electric power (Watts)')
plt.show()

"""
"""

"""
Date of the months:
1984-01
1999-02
1995-03
1991-04
1986-05
1993-06
1985-07
1999-08
1990-09
1983-10
1992-11
1983-12
"""