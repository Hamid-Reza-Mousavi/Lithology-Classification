
"""Log plotting functions

functions: 1- show_log_in_range_depth (raw_plot DataFrame)
           2- show_log (raw_plot DataFrame)
           3- augmented_logs (augmented_logs DataFrame)
           4- litho_prediction (litho_predictions DataFram)

some require libraries: pandas, numpy, matplotlib
"""
#PLOTTING logs in specific depth
import pandas as pd
import numpy as np
def triple_combo(logs, well_num, column_depth, column_GR, column_resistivity, 
                 column_NPHI, column_RHOB, min_depth, max_depth, 
                 min_GR=0, max_GR=150, sand_GR_line=60,
                 min_resistivity=0.01, max_resistivity=1000, 
                 color_GR='black', color_resistivity='green', 
                 color_RHOB='red', color_NPHI='blue',
                 figsize=(6,10), tight_layout=1, 
                 title_size=15, title_height=1.05):
  """
  Producing Triple Combo log

  Input:

  df is your dataframe
  column_depth, column_GR, column_resistivity, column_NPHI, column_RHOB
  are column names that appear in your dataframe (originally from the LAS file)

  specify your depth limits; min_depth and max_depth

  input variables other than above are default. You can specify
  the values yourselves. 

  Output:

  Fill colors; gold (sand), lime green (non-sand), blue (water-zone), orange (HC-zone)
  """
  
  import matplotlib.pyplot as plt
  from matplotlib.ticker import AutoMinorLocator  
  wells = logs['WELL'].unique() # creating a wells list
  logs = logs[logs['WELL'] == wells[well_num]] # selecting well by index number
  logs = logs.sort_values(by='DEPTH_MD') # sorting well log by depth
  fig, ax=plt.subplots(1,3,figsize=(8,10))
  fig.suptitle('Well Logs '+str(wells[well_num]), fontsize=16,y=0.97)

  ax[0].minorticks_on()
  ax[0].grid(which='major', linestyle='-', linewidth='0.5', color='lime')
  ax[0].grid(which='minor', linestyle=':', linewidth='1', color='black')

  ax[1].minorticks_on()
  ax[1].grid(which='major', linestyle='-', linewidth='0.5', color='lime')
  ax[1].grid(which='minor', linestyle=':', linewidth='1', color='black')
  ax[1].set_yticklabels([])

  ax[2].minorticks_on()
  ax[2].grid(which='major', linestyle='-', linewidth='0.5', color='lime')
  ax[2].grid(which='minor', linestyle=':', linewidth='1', color='black')  
  ax[2].set_yticklabels([])
  # First track: GR
  ax[0].get_xaxis().set_visible(False)
  ax[0].invert_yaxis()   

  gr=ax[0].twiny()
  gr.set_xlim(min_GR,max_GR)
  gr.set_xlabel('GR',color=color_GR)
  gr.set_ylim(max_depth, min_depth)
  gr.spines['top'].set_position(('outward',10))
  gr.tick_params(axis='x',colors=color_GR)
  gr.plot(logs[column_GR], logs[column_depth], color=color_GR)  

  gr.minorticks_on()
  gr.xaxis.grid(which='major', linestyle='-', linewidth='0.5', color='lime')
  gr.xaxis.grid(which='minor', linestyle=':', linewidth='1', color='black') 

  gr.fill_betweenx(logs[column_depth], sand_GR_line, logs[column_GR], where=(sand_GR_line>=logs[column_GR]), color = 'gold', linewidth=0) # sand
  gr.fill_betweenx(logs[column_depth], sand_GR_line, logs[column_GR], where=(sand_GR_line<logs[column_GR]), color = 'lime', linewidth=0) # shale

  # Second track: Resistivity
  ax[1].get_xaxis().set_visible(False)
  ax[1].invert_yaxis()   

  res=ax[1].twiny()
  res.set_xlim(min_resistivity,max_resistivity)
  res.set_xlabel('Resistivity',color=color_resistivity)
  res.set_ylim(max_depth, min_depth)
  res.spines['top'].set_position(('outward',10))
  res.tick_params(axis='x',colors=color_resistivity)
  res.semilogx(logs[column_resistivity], logs[column_depth], color=color_resistivity)    

  res.minorticks_on()
  res.xaxis.grid(which='major', linestyle='-', linewidth='0.5', color='lime')
  res.xaxis.grid(which='minor', linestyle=':', linewidth='1', color='black')   

  # Third track: NPHI and RHOB
  ax[2].get_xaxis().set_visible(False)
  ax[2].invert_yaxis()  

  ## NPHI curve 
  nphi=ax[2].twiny()
  nphi.set_xlim(-0.15,0.45)
  nphi.invert_xaxis()
  nphi.set_xlabel('NPHI',color='blue')
  nphi.set_ylim(max_depth, min_depth)
  nphi.spines['top'].set_position(('outward',10))
  nphi.tick_params(axis='x',colors='blue')
  nphi.plot(logs[column_NPHI], logs[column_depth], color=color_NPHI)

  nphi.minorticks_on()
  nphi.xaxis.grid(which='major', linestyle='-', linewidth='0.5', color='lime')
  nphi.xaxis.grid(which='minor', linestyle=':', linewidth='1', color='black')     

  ## RHOB curve 
  rhob=ax[2].twiny()
  rhob.set_xlim(1.95,2.95)
  rhob.set_xlabel('RHOB',color='red')
  rhob.set_ylim(max_depth, min_depth)
  rhob.spines['top'].set_position(('outward',50))
  rhob.tick_params(axis='x',colors='red')
  rhob.plot(logs[column_RHOB], logs[column_depth], color=color_RHOB)

  # solution to produce fill between can be found here:
  # https://stackoverflow.com/questions/57766457/how-to-plot-fill-betweenx-to-fill-the-area-between-y1-and-y2-with-different-scal
  x2p, _ = (rhob.transData + nphi.transData.inverted()).transform(np.c_[logs[column_RHOB], logs[column_depth]]).T
  nphi.autoscale(False)
  nphi.fill_betweenx(logs[column_depth], logs[column_NPHI], x2p, color="orange", alpha=0.4, where=(x2p > logs[column_NPHI])) # hydrocarbon
  nphi.fill_betweenx(logs[column_depth], logs[column_NPHI], x2p, color="blue", alpha=0.4, where=(x2p < logs[column_NPHI])) # water

  res.minorticks_on()
  res.grid(which='major', linestyle='-', linewidth='0.5', color='lime')
  res.grid(which='minor', linestyle=':', linewidth='1', color='black')

  plt.tight_layout(tight_layout)  
  plt.show()

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

