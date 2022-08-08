
"""Log plotting functions

functions: 1- show_log_in_range_depth (raw_plot DataFrame)
           2- show_log (raw_plot DataFrame)
           3- augmented_logs (augmented_logs DataFrame)
           4- litho_prediction (litho_predictions DataFram)

some require libraries: pandas, numpy, matplotlib
"""
#PLOTTING logs in specific depth
import pandas as pd
def show_log_in_range_depth(
  logs:pd.DataFrame,
  well_num:int,
  top_depth:int,
  bottom_depth:int
  ):
  """Plots the raw logs contained in the original datasets after they have been formated.
  For Plots logs in specific depth.

  Parameters
  ----------
  logs: Dataframe
    The raw logs once the headers and necessary columns have been formated and fixed.
  well_num: int
    The number of the well to be plotted. raw_logs internally defines a list of weells 
    contained by the dataset, each of them could be called by its list index.
  top_depth: int
  bottom_depth: int

  Returns
  ----------
  plot:
    Different tracks having one well log in top and bot depth range 
  """
  from matplotlib import pyplot as plt
  wells = logs['WELL'].unique() # creating a wells list
  logs = logs[logs['WELL'] == wells[well_num]] # selecting well by index number
  logs = logs.sort_values(by='DEPTH_MD') # sorting well log by depth
  fig, ax = plt.subplots(figsize=(15,10))
  fig.patch.set_facecolor('white')
  #Set up the plot axes
  ax1 = plt.subplot2grid((1,6), (0,0), rowspan=1, colspan = 1)
  ax2 = plt.subplot2grid((1,6), (0,1), rowspan=1, colspan = 1, sharey = ax1)
  ax3 = plt.subplot2grid((1,6), (0,2), rowspan=1, colspan = 1, sharey = ax1)
  ax4 = plt.subplot2grid((1,6), (0,3), rowspan=1, colspan = 1, sharey = ax1)
  ax5 = ax3.twiny() #Twins the y-axis for the density track with the neutron track
  ax6 = plt.subplot2grid((1,6), (0,4), rowspan=1, colspan = 1, sharey = ax1)
  ax7 = ax2.twiny()
  
  # As our curve scales will be detached from the top of the track,
  # this code adds the top border back in without dealing with splines
  ax10 = ax1.twiny()
  ax10.xaxis.set_visible(False)
  ax11 = ax2.twiny()
  ax11.xaxis.set_visible(False)
  ax12 = ax3.twiny()
  ax12.xaxis.set_visible(False)
  ax13 = ax4.twiny()
  ax13.xaxis.set_visible(False)
  ax14 = ax6.twiny()
  ax14.xaxis.set_visible(False)
  
  # Gamma Ray track
  ax1.plot(logs["GR"], logs.DEPTH_MD, color = "green", linewidth = 0.5)
  ax1.set_xlabel("Gamma")
  ax1.xaxis.label.set_color("green")
  ax1.set_xlim(0, 200)
  ax1.set_ylabel("Depth (m)")
  ax1.tick_params(axis='x', colors="green")
  ax1.spines["top"].set_edgecolor("green")
  ax1.title.set_color('green')
  ax1.set_xticks([0, 50, 100, 150, 200])
  
  # Resistivity track
  ax2.plot(logs["RDEP"], logs.DEPTH_MD, color = "red", linewidth = 0.5)
  ax2.set_xlabel("Resistivity - Deep")
  ax2.set_xlim(0.2, 2000)
  ax2.xaxis.label.set_color("red")
  ax2.tick_params(axis='x', colors="red")
  ax2.spines["top"].set_edgecolor("red")
  ax2.set_xticks([0.1, 1, 10, 100, 1000])
  ax2.semilogx()
  
  # Density track
  ax3.plot(logs["RHOB"], logs.DEPTH_MD, color = "red", linewidth = 0.5)
  ax3.set_xlabel("Density")
  ax3.set_xlim(1.95, 2.95)
  ax3.xaxis.label.set_color("red")
  ax3.tick_params(axis='x', colors="red")
  ax3.spines["top"].set_edgecolor("red")
  ax3.set_xticks([1.95, 2.45, 2.95])
  
  # Sonic track
  ax4.plot(logs["DTC"], logs.DEPTH_MD, color = "purple", linewidth = 0.5)
  ax4.set_xlabel("Sonic-DTC")
  ax4.set_xlim(140, 40)
  ax4.xaxis.label.set_color("purple")
  ax4.tick_params(axis='x', colors="purple")
  ax4.spines["top"].set_edgecolor("purple")
  
  # Neutron track placed ontop of density track
  ax5.plot(logs["NPHI"], logs.DEPTH_MD, color = "blue", linewidth = 0.5)
  ax5.set_xlabel('Neutron')
  ax5.xaxis.label.set_color("blue")
  ax5.set_xlim(45, -15)
  ax5.set_ylim(4150, 3500)
  ax5.tick_params(axis='x', colors="blue")
  ax5.spines["top"].set_position(("axes", 1.08))
  ax5.spines["top"].set_visible(True)
  ax5.spines["top"].set_edgecolor("blue")
  ax5.set_xticks([45,  15, -15])
  
  # Caliper track
  ax6.plot(logs["CALI"], logs.DEPTH_MD, color = "black", linewidth = 0.5)
  ax6.set_xlabel("Caliper")
  ax6.set_xlim(6, 16)
  ax6.xaxis.label.set_color("black")
  ax6.tick_params(axis='x', colors="black")
  ax6.spines["top"].set_edgecolor("black")
  ax6.set_xticks([6,  11, 16])
  
  # Resistivity track - Curve 2
  ax7.plot(logs["RMED"], logs.DEPTH_MD, color = "green", linewidth = 0.5)
  ax7.set_xlabel("Resistivity - Med")
  ax7.set_xlim(0.2, 2000)
  ax7.xaxis.label.set_color("green")
  ax7.spines["top"].set_position(("axes", 1.08))
  ax7.spines["top"].set_visible(True)
  ax7.tick_params(axis='x', colors="green")
  ax7.spines["top"].set_edgecolor("green")
  ax7.set_xticks([0.1, 1, 10, 100, 1000])
  ax7.semilogx()
  
  
  # Common functions for setting up the plot can be extracted into
  # a for loop. This saves repeating code.
  for ax in [ax1, ax2, ax3, ax4, ax6]:
      ax.set_ylim(bottom_depth, top_depth)
      ax.grid(which='major', color='lightgrey', linestyle='-')
      ax.xaxis.set_ticks_position("top")
      ax.xaxis.set_label_position("top")
      ax.spines["top"].set_position(("axes", 1.02))
      
      
  for ax in [ax2, ax3, ax4, ax6]:
      plt.setp(ax.get_yticklabels(), visible = False)
      
  plt.tight_layout()
  fig.subplots_adjust(wspace = 0.15)
  fig.suptitle('Well Logs '+str(wells[well_num]), fontsize=16,y=1.03)
#PLOTTING log + litology

def show_log(
  logs:pd.DataFrame,
  Logs_name_to_plot:list,
  well_num:int,
  le
  ):
  """Plots the raw logs contained in the original datasets after they have been formated.

  Parameters
  ----------
  logs: DataDrame
    The raw logs once the headers and necessary columns have been formated and fixed.
  Logs_name_to_plot: list
    select your logs name want to plot
  well_num: int
    The number of the well to be plotted. raw_logs internally defines a list of weells 
    contained by the dataset, each of them could be called by its list index.
  le:list
    Not need to change it.It is a labelencoder and we need it for le.classes_

  Returns
  ----------
  plot:
    Different tracks having one well log each and a final track containing the 
    lithofacies interpretation.
  """
  import numpy as np
  import matplotlib.pyplot as plt
  import matplotlib.colors as colors
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  facies_colors = ['tan', 'magenta', 'lawngreen', '#000000', 'gold', 'lightblue', 'lightseagreen', 'cyan', 'darkorange', '#228B22' , 'grey', '#FF4500']
  facies_labels = le.classes_
  facies_color_map = {} # creating facies color map
  for ind, label in enumerate(facies_labels):
      facies_color_map[label] = facies_colors[ind]
  wells = logs['WELL'].unique() # creating a wells list
  logs = logs[logs['WELL'] == wells[well_num]] # selecting well by index number
  logs = logs.sort_values(by='DEPTH_MD') # sorting well log by depth
  cmap_facies = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')
  top = logs.DEPTH_MD.min()
  bot = logs.DEPTH_MD.max()
  
  real_label = np.repeat(np.expand_dims(logs['LITHO'].values, 1), 100, 1)
  f, ax = plt.subplots(nrows=1, ncols=len(Logs_name_to_plot)+1, figsize=(2*len(Logs_name_to_plot), 12))
  f.patch.set_facecolor('white')
  log_colors = ['green', 'blue', '#8A2BE2', '0.5', 'r', 'black', 'm', 'purple', '#D2691E', '#DC143C', '#008B8B', '#9932CC', '#00CED1', '#DAA520', '#7B68EE', '#8B4513' ]
  for i in range(0,len(Logs_name_to_plot)):
    ax[i].plot(logs[Logs_name_to_plot[i]], logs.DEPTH_MD, color=log_colors[i]) # plotting each well log on each track
    ax[i].set_ylim(top, bot)
    ax[i].set_xlabel(Logs_name_to_plot[i])
    ax[i].invert_yaxis()
    ax[i].grid()
    ax[i].tick_params(axis='x', colors=log_colors[i])
    ax[i].spines["top"].set_edgecolor(log_colors[i])  
    ax[i].xaxis.label.set_color(log_colors[i])
    ax[i].xaxis.set_ticks_position("top")
    ax[i].xaxis.set_label_position("top")
    ax[i].spines["top"].set_position(("axes", 1.02))
    ax[0].set_ylabel("Depth (m)")
    ax[i+1].set_yticklabels([])

  im = ax[-1].imshow(real_label, interpolation='none', aspect='auto', cmap=cmap_facies, vmin=0, vmax=11)
  ax[-1].set_xlabel('Real Lith') # creating a facies log on the final track
  ax[-1].set_xticklabels([])
  ax[-1].xaxis.set_label_position("top")
  divider = make_axes_locatable(ax[-1]) # appending legend besides the facies log
  cax = divider.append_axes("right", size="20%", pad=0.05)
  cbar=plt.colorbar(im, cax=cax)
  cbar.set_label((12*' ').join(le.classes_))
  
  cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
  f.suptitle('Well Logs '+str(wells[well_num]), fontsize=16,y=0.97)

#PLOTTING LOGS AUGMENTED BY ML
def augmented_logs(
  logs:pd.DataFrame,
  well_num:int,
  le
  ):
  """Plots the raw, predicted, and augmented wireline logs after applying data augmentation.

  Parameters
  ----------
  logs: DataFrame
    The raw, predicted, and augmented logs.
  well_num: int
    The number of the well to be plotted. augmented_logs internally defines a list of 
    weells contained by the logs dataframe, each of which could be called by its list index.

  Returns
  ----------
  plot:
    Different tracks containing the raw, predicted, and augmented logs.
    Augmented logs mean that the missing values hbeen filled up by machine-learning
    predicted readings.
  """
  #auxiliar libraries
  import numpy as np
  import matplotlib.pyplot as plt
  import matplotlib.colors as colors
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  facies_colors = ['tan', 'magenta', 'lawngreen', '#000000', 'gold', 'lightblue', 'lightseagreen', 'cyan', 'darkorange', '#228B22' , 'grey', '#FF4500']
  facies_labels = le.classes_
  facies_color_map = {}  # creating facies color map
  for ind, label in enumerate(facies_labels):
      facies_color_map[label] = facies_colors[ind]
  wells = logs['WELL'].unique()
  logs = logs[logs['WELL'] == wells[well_num]] # selecting well by index number
  logs = logs.sort_values(by='DEPTH_MD') # sorting well log by depth
  cmap_facies = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')
  
  top = logs.DEPTH_MD.min()
  bot = logs.DEPTH_MD.max()
  
  real_label = np.repeat(np.expand_dims(logs['LITHO'].values, 1), 100, 1)
  f, ax = plt.subplots(nrows=1, ncols=13, figsize=(20, 12))
  f.patch.set_facecolor('white')
  log_colors = ['green', 'blue', '#8A2BE2', '0.5', 'r', 'black', 'm', 'purple', '#D2691E', '#DC143C', '#008B8B', '#9932CC', '#00CED1', '#DAA520', '#7B68EE', '#8B4513' ]
  for i in range(3,15):
    ax[i-3].plot(logs.iloc[:,i], logs.DEPTH_MD, color=log_colors[i]) # plotting raw, predicted, and augmented logs
    ax[i-3].set_ylim(top, bot)
    ax[i-3].set_xlabel(str(logs.columns[i]))
    ax[i-3].invert_yaxis()
    ax[i-3].grid()
    ax[i-3].tick_params(axis='x', colors=log_colors[i])
    ax[i-3].spines["top"].set_edgecolor(log_colors[i])  
    ax[i-3].xaxis.label.set_color(log_colors[i])
    ax[i-3].xaxis.set_ticks_position("top")
    ax[i-3].xaxis.set_label_position("top")
    ax[i-3].spines["top"].set_position(("axes", 1.02))
    ax[0].set_ylabel("Depth (m)")
    ax[i-3+1].set_yticklabels([])
    
  im = ax[-1].imshow(real_label, interpolation='none', aspect='auto', cmap=cmap_facies, vmin=0, vmax=11)
  ax[-1].set_xlabel('LITHO') # creating a facies log on the final track
  ax[-1].set_xticklabels([])
  ax[-1].xaxis.set_label_position("top")
  #ax[-1].spines["top"].set_position(("axes", 1.02))
  divider = make_axes_locatable(ax[-1]) # appending legend besides the facies log
  cax = divider.append_axes("right", size="20%", pad=0.05)
  cbar=plt.colorbar(im, cax=cax)
  cbar.set_label((12*' ').join(le.classes_))
  cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
  f.suptitle('WELL LOGS '+str(wells[well_num]), fontsize=16,y=0.97)

#PLOTTING LITHOFACIES PREDICTION
def litho_prediction(
logs:pd.DataFrame,
well_num:int,
n_pred:int,
le
):
  """Plots the raw logs, the lihtology interpretation, and the n_pred number of predcted 
    lithologies by machine learning.

    Parameters
    ----------
    logs: dataframe
      Dataframe holding the raw wireline logs, true lithology, and n_pred columns 
      containing different ML model predictions each.
    well_num: int
      The number of the well to be plotted. litho_prediction internally defines a list of 
      weells contained by the logs dataframe, each of which could be called by its list index.

    Returns
    ----------
    plot:
      Different track plots representing each wireline log, the true lihtology and the 
      predicted lithologies by dfferent mane-learning models.
    """
  import numpy as np
  import matplotlib.pyplot as plt
  import matplotlib.colors as colors
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  facies_colors = ['tan', 'magenta', 'lawngreen', '#000000', 'gold', 'lightblue', 'lightseagreen', 'cyan', 'darkorange', '#228B22' , 'grey', '#FF4500']
  facies_labels = le.classes_
  facies_color_map = {} #creating facies coorap
  for ind, label in enumerate(facies_labels):
      facies_color_map[label] = facies_colors[ind]
  wells = logs['WELL'].unique() # well names list
  logs = logs[logs['WELL'] == wells[well_num]]
  logs = logs.sort_values(by='DEPTH_MD') # sorting the plotted well logs by depth       
  cmap_facies = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')
  top = logs.DEPTH_MD.min()
  bot = logs.DEPTH_MD.max()
  f, ax = plt.subplots(nrows=1, ncols=(12+n_pred), figsize=(20, 12))
  f.patch.set_facecolor('white')
  log_colors = ['green', 'blue', '#8A2BE2', '0.5', 'r', 'black', 'm', 'purple', '#D2691E', '#DC143C', '#008B8B', '#9932CC', '#00CED1', '#DAA520', '#7B68EE', '#8B4513',
                'green', 'blue', '#8A2BE2', '0.5', 'r', 'black', 'm', 'purple', '#D2691E', '#DC143C', '#008B8B', '#9932CC', '#00CED1', '#DAA520', '#7B68EE', '#8B4513' ]

  for i in range(7,18):
    ax[i-7].plot(logs.iloc[:,i], logs.DEPTH_MD, color=log_colors[i]) # plotting continuous wireline logs
    ax[i-7].set_ylim(top, bot)
    ax[i-7].set_xlabel(str(logs.columns[i]))
    ax[i-7].invert_yaxis()
    ax[i-7].grid()
    ax[i-7].tick_params(axis='x', colors=log_colors[i])
    ax[i-7].spines["top"].set_edgecolor(log_colors[i])  
    ax[i-7].xaxis.label.set_color(log_colors[i])
    ax[i-7].xaxis.set_ticks_position("top")
    ax[i-7].xaxis.set_label_position("top")
    ax[i-7].spines["top"].set_position(("axes", 1.02))
    ax[0].set_ylabel("Depth (m)")
    ax[i-7+1].set_yticklabels([])

  for j in range((-1-n_pred), 0): # ploting the lithology predictions obtainedby ML
    label = np.repeat(np.expand_dims(logs.iloc[:,j].values, 1), 100, 0)
    im = ax[j].imshow(label, interpolation='none', aspect='auto', cmap=cmap_facies, vmin=0, vmax=11)
    ax[j].set_xlabel(str(logs.columns[j]))
    ax[j].set_xticklabels([])
    ax[j].set_yticklabels([])
    ax[j].xaxis.set_label_position("top")
  divider = make_axes_locatable(ax[-1]) # appending lithology legend
  cax = divider.append_axes("right", size="20%", pad=0.05)
  cbar=plt.colorbar(im, cax=cax)
  cbar.set_label((12*' ').join(le.classes_))
  cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
  f.suptitle('WELL LOGS '+str(wells[well_num]), fontsize=14,y=0.97)