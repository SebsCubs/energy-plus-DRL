import pandas as pd
import matplotlib.pyplot as plt

output_dfs = pd.read_csv("C:\Projects\SDU\Thesis\pyenergyplus\Dataframes\dataframes_output_model.csv")

#Take the rows with Datetime containing the value 1984-01-01
#output_dfs = output_dfs[output_dfs['Datetime'].str.contains('1983-12')]

#Drop duplicates of the Datetime column
#output_dfs = output_dfs.drop_duplicates(subset='Datetime', keep='first')
#reset the index
#output_dfs = output_dfs.reset_index(drop=True)


print(output_dfs.size)
print(output_dfs.head(10))


# -- Plot Results --
#plot with date in the x-axis
fig, ax = plt.subplots()
output_dfs.plot(x='Datetime', y='air_loop_fan_mass_flow_var', use_index=True, ax=ax)
#output_dfs.plot(x='Datetime', y='deck_temp', use_index=True, ax=ax)
#output_dfs.plot(x='Datetime', y='post_deck_temp', use_index=True, ax=ax)
#output_dfs.plot(x='Datetime', y='air_loop_fan_electric_power', use_index=True, ax=ax)
plt.title('Fan mass flow rate')
plt.show()


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