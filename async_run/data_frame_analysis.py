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
    temp_model_mean = np.mean(y_axis)
    temp_baseline_mean = np.mean(y_baseline)
    temp_model_label = f"{'RL Agent (mean: '}{temp_model_mean:.2f}{')'}"
    temp_baseline_labe = f"{'Baseline controller (mean: '}{temp_baseline_mean:.2f}{')'}"
    ax.legend(handles=[vplot1["bodies"][0], vplot2["bodies"][0]], labels=[temp_model_label, temp_baseline_labe])
    #save figure in a folder called "Figures"
    plt.savefig('Figures/room_temp.png', dpi=300, bbox_inches="tight")

    #plt.show()


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
    ppd_model_mean = np.mean(y_axis)
    ppd_baseline_mean = np.mean(y_baseline)
    ppd_model_label = f"{'RL Agent (mean: '}{ppd_model_mean:.2f}{')'}"
    ppd_baseline_labe = f"{'Baseline controller (mean: '}{ppd_baseline_mean:.2f}{')'}"
    ax.legend(handles=[vplot1["bodies"][0], vplot2["bodies"][0]], labels=[ppd_model_label, ppd_baseline_labe])
    #save figure in a folder called "Figures"
    plt.savefig('Figures/ppd.png', dpi=300, bbox_inches="tight")



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
    power_model_mean = np.mean(y_axis)
    power_baseline_mean = np.mean(y_baseline)
    power_model_label = f"{'RL Agent (mean: '}{power_model_mean:.2f}{')'}"
    power_baseline_labe = f"{'Baseline controller (mean: '}{power_baseline_mean:.2f}{')'}"
    ax.legend(handles=[vplot1["bodies"][0], vplot2["bodies"][0]], labels=[power_model_label, power_baseline_labe])
    #save figure in a folder called "Figures"
    plt.savefig('Figures/fan_power.png', dpi=300, bbox_inches="tight")


    ######### Room temperature time series #########

    # -- Plot Results --
    #plot with date in the x-axis
    fig, ax = plt.subplots()
    output_dfs.plot(x='Datetime', y='zn0_temp', use_index=True, ax=ax)
    #Reduce the number of x-axis labels
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    output_dfs.plot(x='Datetime', y='oa_db', use_index=True, ax=ax)
    plt.title('Room temperature (Time)')
    
    #save figure in a folder called "Figures"
    plt.savefig('Figures/room_temp_time_series.png', dpi=300, bbox_inches="tight")

    ########## Whole building Power usage ##########

    y_axis = output_dfs['total_hvac_energy']
    y_baseline = baseline_dfs['total_hvac_energy']
    fig, ax = plt.subplots()
    vplot1 = ax.violinplot(y_axis, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    vplot2 = ax.violinplot(y_baseline, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    

    ax.set_title('Whole HVAC power usage over December (Temperature reward)')
    ax.set_xlabel('Power (Watts)')
    ax.set_ylabel('Density')
    model_mean = np.mean(y_axis)
    baseline_mean = np.mean(y_baseline)
    model_label = f"{'RL Agent (mean: '}{model_mean:.2f}{')'}"
    baseline_labe = f"{'Baseline controller (mean: '}{baseline_mean:.2f}{')'}"
    ax.legend(handles=[vplot1["bodies"][0], vplot2["bodies"][0]], labels=[model_label, baseline_labe])
    #save figure in a folder called "Figures"
    plt.savefig('Figures/whole_hvac.png', dpi=300, bbox_inches="tight")
    


    #########Make a figure with all 4 previous plots############
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    #fig.suptitle('December 2017 results (Temperature reward, ext. Fan)')
    ax[0, 0].set_title('PPD')
    ax[0, 0].set_xlabel('PPD (percentage)')
    ax[0, 0].set_ylabel('Density')
    vplot1 = ax[0, 0].violinplot(output_dfs['ppd'], points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    vplot2 = ax[0, 0].violinplot(baseline_dfs['ppd'], points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    
    #Set the y-axis limits 105percent larger in the top
    ylim = ax[0, 0].get_ylim()
    ax[0, 0].set_ylim(ylim[0], ylim[1]*1.15)
    ax[0,0].legend(handles=[vplot1["bodies"][0], vplot2["bodies"][0]], labels=[ppd_model_label, ppd_baseline_labe])



    ax[0, 1].set_title('Fan power usage')
    ax[0, 1].set_xlabel('Power (Watts)')
    ax[0, 1].set_ylabel('Density')
    ax[0, 1].violinplot(output_dfs['air_loop_fan_electric_power'], points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    ax[0, 1].violinplot(baseline_dfs['air_loop_fan_electric_power'], points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    #Set the y-axis limits 105percent larger in the top
    ylim = ax[0, 1].get_ylim()
    ax[0, 1].set_ylim(ylim[0], ylim[1]*1.15)
    ax[0,1].legend(handles=[vplot1["bodies"][0], vplot2["bodies"][0]], labels=[power_model_label, power_baseline_labe])

    ax[1, 0].set_title('Room temperature')
    ax[1, 0].set_xlabel('Temperature (C)')
    ax[1, 0].set_ylabel('Density')
    ax[1, 0].violinplot(output_dfs['zn0_temp'], points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    ax[1, 0].violinplot(baseline_dfs['zn0_temp'], points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    #Set the y-axis limits 105percent larger in the top
    ylim = ax[1, 0].get_ylim()
    ax[1, 0].set_ylim(ylim[0], ylim[1]*1.15)
    ax[1,0].legend(handles=[vplot1["bodies"][0], vplot2["bodies"][0]], labels=[temp_model_label, temp_baseline_labe])

    ax[1, 1].set_title('Room temp. Time series')
    ax[1, 1].set_xlabel('Degrees C')
    ax[1, 1].set_ylabel('Datetime')


    output_dfs.plot(x='Datetime', y='zn0_temp', use_index=True, ax=ax[1, 1])
    output_dfs.plot(x='Datetime', y='oa_db', use_index=True, ax=ax[1, 1])
    ax[1, 1].xaxis.set_major_locator(plt.MaxNLocator(1))
    #Set legend for the time series plot
    ax[1, 1].legend(['Room temp.', 'Outdoor temp.'])
    
    #save figure in a folder called "Figures"
    plt.savefig('Figures/all_violin_plots.png', dpi=300, bbox_inches="tight")




    ############# heating coil power usage #############
    model_heat_coil_power = output_dfs['damper_coil_heating_rate'] + output_dfs['pre_heating_coil_htgrate']
    baseline_heating_coil_power = baseline_dfs['damper_coil_heating_rate'] + baseline_dfs['pre_heating_coil_htgrate']
    fig, ax = plt.subplots()
    #fig.suptitle('December 2017 results (Temperature reward, ext. Fan)')
    ax.set_title('Heating coils heating power usage')
    ax.set_xlabel('Power (Watts)')
    ax.set_ylabel('Density')
    vplot1 = ax.violinplot(model_heat_coil_power, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    vplot2 = ax.violinplot(baseline_heating_coil_power, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    
    #add a legend for each violin plot with the text "Model" and "Baseline" and the mean value of each one, rounded to 2 decimals
    ax.legend(handles=[vplot1["bodies"][0], vplot2["bodies"][0]], labels=["Model (mean: " + str(round(np.mean(model_heat_coil_power), 2)) + ")", "Baseline (mean: " + str(round(np.mean(baseline_heating_coil_power), 2)) + ")"])  
    
    #save figure in a folder called "Figures"
    plt.savefig('Figures/heating_coils_power.png', dpi=300, bbox_inches="tight")

    ############ total hvac power usage #############
    model_total_hvac_power = output_dfs['air_loop_fan_electric_power'] + output_dfs['damper_coil_heating_rate'] + output_dfs['pre_heating_coil_htgrate']
    baseline_total_hvac_power = baseline_dfs['air_loop_fan_electric_power'] + baseline_dfs['damper_coil_heating_rate'] + baseline_dfs['pre_heating_coil_htgrate']
    fig, ax = plt.subplots()
    #fig.suptitle('December 2017 results (Temperature reward, ext. Fan)')
    ax.set_title('Total HVAC power usage')
    ax.set_xlabel('Power (Watts)')
    ax.set_ylabel('Density')
    vplot1 = ax.violinplot(model_total_hvac_power, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    vplot2 = ax.violinplot(baseline_total_hvac_power, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    
    #add a legend for each violin plot with the text "Model" and "Baseline" and the mean value of each one, rounded to 2 decimals
    ax.legend(handles=[vplot1["bodies"][0], vplot2["bodies"][0]], labels=["Model (mean: " + str(round(np.mean(model_total_hvac_power), 2)) + ")", "Baseline (mean: " + str(round(np.mean(baseline_total_hvac_power), 2)) + ")"])

    #save figure in a folder called "Figures"
    plt.savefig('Figures/total_hvac_power.png', dpi=300, bbox_inches="tight")

    ############## total hvac power from energyplus #############
    model_total_hvac_power = output_dfs['total_hvac_energy']
    baseline_total_hvac_power = baseline_dfs['total_hvac_energy']
    fig, ax = plt.subplots()
    #fig.suptitle('December 2017 results (Temperature reward, ext. Fan)')
    ax.set_title('Total HVAC power usage')
    ax.set_xlabel('Power (Watts)')
    ax.set_ylabel('Density')
    vplot1 = ax.violinplot(model_total_hvac_power, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    vplot2 = ax.violinplot(baseline_total_hvac_power, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    
    #add a legend for each violin plot with the text "Model" and "Baseline" and the mean value of each one, rounded to 2 decimals
    ax.legend(handles=[vplot1["bodies"][0], vplot2["bodies"][0]], labels=["Model (mean: " + str(round(np.mean(model_total_hvac_power), 2)) + ")", "Baseline (mean: " + str(round(np.mean(baseline_total_hvac_power), 2)) + ")"])

    #save figure in a folder called "Figures"
    plt.savefig('Figures/total_hvac_power_2.png', dpi=300, bbox_inches="tight")




    
def plot_temp_baseline_vs_model():
    output_dfs = pd.read_csv("/home/jun/HVAC/repo_recover/energy-plus-DRL/Dataframes/dataframes_output_model.csv")
    baseline_dfs = pd.read_csv("/home/jun/HVAC/repo_recover/energy-plus-DRL/Dataframes/dataframes_output_test.csv")
    #plot the temperature time series for the baseline and the model in two separate plots within one figure
    fig, ax = plt.subplots(2, 1)
    ax[0].set_title('Room temp. Time series')
    ax[0].set_ylabel('Degrees C')
    output_dfs.plot(x='Datetime', y='zn0_temp', use_index=True, ax=ax[0])
    output_dfs.plot(x='Datetime', y='oa_db', use_index=True, ax=ax[0])
    ax[0].xaxis.set_major_locator(plt.MaxNLocator(3))
    ax[0].legend(['Room temp.', 'Outdoor temp.'])
    ax[1].set_ylabel('Degrees C')
    #ax[1].set_xlabel('Datetime')
    baseline_dfs.plot(x='Datetime', y='zn0_temp', use_index=True, ax=ax[1])
    baseline_dfs.plot(x='Datetime', y='oa_db', use_index=True, ax=ax[1])
    ax[1].xaxis.set_major_locator(plt.MaxNLocator(3))
    ax[1].legend(['Room temp.', 'Outdoor temp.'])
    #add space between plots
    plt.subplots_adjust(hspace=0.5)
    #save figure in a folder called "Figures"

    plt.savefig('Figures/temp_baseline_vs_model.png', dpi=300, bbox_inches="tight")
    


def plot_only_baseline():
    baseline_dfs = pd.read_csv("/home/jun/HVAC/repo_recover/energy-plus-DRL/Dataframes/dataframes_output_test.csv")
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    #fig.suptitle('December 2017 results (Temperature reward, ext. Fan)')
    ax[0, 0].set_title('PPD')
    ax[0, 0].set_xlabel('PPD (percentage)')
    ax[0, 0].set_ylabel('Density')
    vplot2 = ax[0, 0].violinplot(baseline_dfs['ppd'], points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)



    ax[0, 1].set_title('Fan power usage')
    ax[0, 1].set_xlabel('Power (Watts)')
    ax[0, 1].set_ylabel('Density')
    ax[0, 1].violinplot(baseline_dfs['air_loop_fan_electric_power'], points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)

    ax[1, 0].set_title('Room temperature')
    ax[1, 0].set_xlabel('Temperature (C)')
    ax[1, 0].set_ylabel('Density')
    ax[1, 0].violinplot(baseline_dfs['zn0_temp'], points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)

    ax[1, 1].set_title('Room temp. Time series')
    ax[1, 1].set_xlabel('Degrees C')
    ax[1, 1].set_ylabel('Datetime')


    baseline_dfs.plot(x='Datetime', y='zn0_temp', use_index=True, ax=ax[1, 1])
    baseline_dfs.plot(x='Datetime', y='oa_db', use_index=True, ax=ax[1, 1])
    ax[1, 1].xaxis.set_major_locator(plt.MaxNLocator(1))
    
    #save figure in a folder called "Figures"
    plt.savefig('Figures/baseline_all_violin_plots.png', dpi=300, bbox_inches="tight")


    ######### Room temperature time series #########

    # -- Plot Results --
    #plot with date in the x-axis
    fig, ax = plt.subplots()
    baseline_dfs.plot(x='Datetime', y='zn0_temp', use_index=True, ax=ax)
    #Reduce the number of x-axis labels
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    baseline_dfs.plot(x='Datetime', y='oa_db', use_index=True, ax=ax)
    plt.title('Room temperature (Time)')
    
    #save figure in a folder called "Figures"
    plt.savefig('Figures/baseline_room_temp_time_series.png', dpi=300, bbox_inches="tight")

    ##########3 Energy usage time series ##########
    fig, ax = plt.subplots()
    baseline_dfs.plot(x='Datetime', y='air_loop_fan_electric_power', use_index=True, ax=ax)
    baseline_dfs.plot(x='Datetime', y='damper_coil_heating_rate', use_index=True, ax=ax)
    baseline_dfs.plot(x='Datetime', y='pre_heating_coil_htgrate', use_index=True, ax=ax)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.title('Fan and coils power usage')    
    #save figure in a folder called "Figures"
    plt.savefig('Figures/baseline_heating_power_time_series.png', dpi=300, bbox_inches="tight")

    ############# heating coil power usage #############
    # make an array with the sum of the values of the heating coil and the pre-heating coil
    heating_coil_power = baseline_dfs['damper_coil_heating_rate'] + baseline_dfs['pre_heating_coil_htgrate']
    # make a violin plot with the heating coil power
    fig, ax = plt.subplots()
    ax.violinplot(heating_coil_power, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    ax.set_title('Heating coils power usage')
    ax.set_xlabel('Power (Watts)')
    ax.set_ylabel('Density')
    ax.legend([f'Mean: {heating_coil_power.mean():.2f}'])

    #save figure in a folder called "Figures"
    plt.savefig('Figures/baseline_heating_coils_power.png', dpi=300, bbox_inches="tight")

    ############ total hvac power usage #############
    # make an array with the sum of the values of the heating coil and the pre-heating coil
    total_hvac_power = baseline_dfs['damper_coil_heating_rate'] + baseline_dfs['pre_heating_coil_htgrate'] + baseline_dfs['air_loop_fan_electric_power']
    # make a violin plot with the heating coil power
    fig, ax = plt.subplots()
    ax.violinplot(total_hvac_power, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    ax.set_title('Total HVAC power usage')
    ax.set_xlabel('Power (Watts)')
    ax.set_ylabel('Density')
    ax.legend([f'Mean: {total_hvac_power.mean():.2f}'])

    #save figure in a folder called "Figures"
    plt.savefig('Figures/baseline_total_hvac_power.png', dpi=300, bbox_inches="tight")



if __name__ == '__main__':
    pass
