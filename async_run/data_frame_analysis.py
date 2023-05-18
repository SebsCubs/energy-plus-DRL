import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def plot_results():

    output_dfs = pd.read_csv("/home/jun/HVAC/repo_recover/energy-plus-DRL/Dataframes/dataframes_output_model.csv")
    baseline_dfs = pd.read_csv("/home/jun/HVAC/repo_recover/energy-plus-DRL/Dataframes/dataframes_output_test.csv")


    ########## Temperature ##########

    y_axis = output_dfs['zn0_temp']
    y_baseline = baseline_dfs['zn0_temp']
    fig, ax = plt.subplots()
    vplot1 = ax.violinplot(y_axis, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    vplot2 = ax.violinplot(y_baseline, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)

    ax.set_title('Room temperature over December (Temperature reward, ext. Fan)')
    ax.set_xlabel('Degrees (Celsius)')
    ax.set_ylabel('Density')
    model_mean = np.mean(y_axis)
    baseline_mean = np.mean(y_baseline)
    model_label = f"{'RL Agent (mean: '}{model_mean:.2f}{')'}"
    baseline_labe = f"{'Baseline controller (mean: '}{baseline_mean:.2f}{')'}"
    ax.legend(handles=[vplot1["bodies"][0], vplot2["bodies"][0]], labels=[model_label, baseline_labe])
    plt.show()


    ########## PPD ##########
    y_axis = output_dfs['ppd']
    y_baseline = baseline_dfs['ppd']
    fig, ax = plt.subplots()
    vplot1 = ax.violinplot(y_axis, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    vplot2 = ax.violinplot(y_baseline, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)

    ax.set_title('PPD over December (Temperature reward, ext. Fan)')
    ax.set_xlabel('PPD (percentage)')
    ax.set_ylabel('Density')
    model_mean = np.mean(y_axis)
    baseline_mean = np.mean(y_baseline)
    model_label = f"{'RL Agent (mean: '}{model_mean:.2f}{')'}"
    baseline_labe = f"{'Baseline controller (mean: '}{baseline_mean:.2f}{')'}"
    ax.legend(handles=[vplot1["bodies"][0], vplot2["bodies"][0]], labels=[model_label, baseline_labe])
    plt.show()



    ########## Power usage ##########

    y_axis = output_dfs['air_loop_fan_electric_power']
    y_baseline = baseline_dfs['air_loop_fan_electric_power']
    fig, ax = plt.subplots()
    vplot1 = ax.violinplot(y_axis, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    vplot2 = ax.violinplot(y_baseline, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)

    ax.set_title('Fan power usage over December (Temperature reward, ext. Fan)')
    ax.set_xlabel('Power (Watts)')
    ax.set_ylabel('Density')
    model_mean = np.mean(y_axis)
    baseline_mean = np.mean(y_baseline)
    model_label = f"{'RL Agent (mean: '}{model_mean:.2f}{')'}"
    baseline_labe = f"{'Baseline controller (mean: '}{baseline_mean:.2f}{')'}"
    ax.legend(handles=[vplot1["bodies"][0], vplot2["bodies"][0]], labels=[model_label, baseline_labe])
    plt.show()


    ######### Room temperature time series #########

    # -- Plot Results --
    #plot with date in the x-axis
    fig, ax = plt.subplots()
    output_dfs.plot(x='Datetime', y='zn0_temp', use_index=True, ax=ax)
    output_dfs.plot(x='Datetime', y='oa_db', use_index=True, ax=ax)
    plt.title('Fan electric power (Watts)')
    plt.show()
