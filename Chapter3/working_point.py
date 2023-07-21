


def make_plot(save_plot=False,save_folder=None,title_plot=None):
    import numpy as np 
    import os 
    import sys 
    import pandas as pd
    from scipy.ndimage import gaussian_filter
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    sys.path.append('/home/jaime/Desktop/hippocampus/files')
    import file_management
    # from signal_analysis import *

    idcb = np.array([8, 9, 4, 5, 0, 15, 2, 11, 19, 3, 12, 10, 17])
    idcb = np.sort(idcb)

    cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R','Insula_L', 'Insula_R','Cingulate_Ant_L', 
                     'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R','Hippocampus_L', 
                     'Hippocampus_R', 'ParaHippocampal_L','ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                     'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L','Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R','Thalamus_L', 'Thalamus_R']
    
    # subjects 
    subjects = ['NEMOS_035', 'NEMOS_049', 'NEMOS_050', 'NEMOS_058', 'NEMOS_059',
                'NEMOS_064', 'NEMOS_065', 'NEMOS_071', 'NEMOS_075', 'NEMOS_077']

    # loading the data 
    folder_data = '/home/jaime/Desktop/neuromodulacion/paper/cingulum_bundle/data/'
    nsubjects = len(subjects)
    nnodes = 22
    n = 81
    ntrials = 5 

    fpeak = [] 
    power = [] 
    power_norm = []
    plv_alpha = []
    plv_alpha_list = []
    for s in subjects:
        fpeak.append( file_management.load_lzma(os.path.join(folder_data, f"fpeak_{s}.lzma")) )
        power.append( file_management.load_lzma(os.path.join(folder_data, f"power_{s}.lzma")) )
        power_norm.append( file_management.load_lzma(os.path.join(folder_data, f"power_norm_{s}.lzma")) ) 
        plv_alpha.append( file_management.load_lzma(os.path.join(folder_data, f"plv_alpha_{s}.lzma")) )
        plv_alpha_list.append( file_management.load_lzma(os.path.join(folder_data, f"plv_alpha_list_{s}.lzma")) )
        
    fpeak = np.array(fpeak)
    power = np.array(power)
    power_norm = np.array( power_norm )
    plv_alpha = np.array(plv_alpha)
    plv_alpha_list = np.array(plv_alpha_list)

    # loading the plv matrices 
    nnodes = 22
    current_folder = '/home/jaime/Desktop/neuromodulacion/paper/cingulum_bundle/'#os.getcwd()
    folder_files = os.path.join(current_folder, "files")
    folder = os.path.join(folder_files, f"FCrms_{subjects[0]}")
    FClabs = list(np.loadtxt(os.path.join(folder, "roi_labels_rms.txt"),dtype=str))
    FC_cb_idx = [FClabs.index(roi) for roi in cingulum_rois]

    plv_alpha_exp = []
    plv_alpha_exp_list = []
    for s in range(nsubjects):
        folder = os.path.join(folder_files, f"FCrms_{subjects[s]}")
        plv_alpha_exp.append( np.loadtxt(os.path.join(folder, "3-alpha_plv_rms.txt"),delimiter=',') )
        plv_alpha_exp[-1] = plv_alpha_exp[-1][:,FC_cb_idx][FC_cb_idx]
        aux = []
        for j1 in range(nnodes):
            for j2 in range(j1+1,nnodes):
                aux.append(plv_alpha_exp[-1][j1,j2])
        plv_alpha_exp_list.append(aux)

    plv_alpha_exp      = np.array(plv_alpha_exp)
    plv_alpha_exp_list = np.array(plv_alpha_exp_list)
    plv_alpha_exp_mean = np.mean(plv_alpha_exp_list)
    plv_alpha_exp_std  = np.std(plv_alpha_exp_list)

    coupling_factor = np.linspace(0,0.8,81)

    plv_alpha_mean  = np.zeros((nsubjects, n))
    plv_alpha_std   = np.zeros((nsubjects, n))
    fpeak_mean      = np.zeros((nsubjects, n))
    fpeak_std       = np.zeros((nsubjects, n))
    power_mean      = np.zeros((nsubjects, n))
    power_std       = np.zeros((nsubjects, n))
    power_norm_mean = np.zeros((nsubjects, n))
    power_norm_std  = np.zeros((nsubjects, n))

    fpeak_cluster_mean      = np.zeros((nsubjects, n))
    fpeak_cluster_std       = np.zeros((nsubjects, n))
    power_cluster_mean      = np.zeros((nsubjects, n))
    power_cluster_std       = np.zeros((nsubjects, n))
    power_norm_cluster_mean = np.zeros((nsubjects, n))
    power_norm_cluster_std   = np.zeros((nsubjects, n))

    for s in range(nsubjects):
        for i in range(n):
            plv_alpha_mean[s,i]  = np.mean( np.mean( plv_alpha_list[s,i], axis=1 ) )
            plv_alpha_std[s,i]   = np.mean( np.std(  plv_alpha_list[s,i], axis=1 ) )
            fpeak_mean[s,i]      = np.mean( np.mean( fpeak[s,i],axis=1 ) )
            fpeak_std[s,i]       = np.mean( np.std(  fpeak[s,i],axis=1 ) )
            power_mean[s,i]      = np.mean( np.mean( power[s,i],axis=1 ) )
            power_std[s,i]       = np.mean( np.std(  power[s,i],axis=1 ) )
            power_norm_mean[s,i] = np.mean( np.mean( power_norm[s,i],axis=1 ) )
            power_norm_std[s,i]  = np.mean( np.std(  power_norm[s,i],axis=1 ) )
            
            fpeak_cluster_mean[s,i]      = np.mean( np.mean( fpeak[s,i,idcb],axis=1 ) )
            fpeak_cluster_std[s,i]       = np.mean( np.std(  fpeak[s,i,idcb],axis=1 ) )
            power_cluster_mean[s,i]      = np.mean( np.mean( power[s,i,idcb],axis=1 ) )
            power_cluster_std[s,i]       = np.mean( np.std(  power[s,i,idcb],axis=1 ) )
            power_norm_cluster_mean[s,i] = np.mean( np.mean( power_norm[s,i,idcb],axis=1 ) )
            power_norm_cluster_std[s,i]  = np.mean( np.std(  power_norm[s,i,idcb],axis=1 ) )
            
    correlation = np.zeros((nsubjects, n, ntrials))
    for s in range(nsubjects):
        for i in range(n): 
            for k in range(ntrials):
                correlation[s,i,k] = np.corrcoef( plv_alpha_exp_list[s], plv_alpha_list[s,i,:,k] )[0,1]
    correlation_mean = np.mean(correlation,axis=2)
    correlation_std  = np.std(correlation, axis=2)


    # making the subplots ###################################################33
    fig = make_subplots(
        rows=7, cols=2,
        specs=[[{"type":"scatter"}, {"type":"scatter"}],
              [{"type":"scatter"}, {"type":"scatter"}],
              [{"rowspan":2,"colspan": 2,"type":"bar"},None],
              [None, None],
              [{"rowspan": 2, "colspan": 2,"type":"scatter"},None],
              [None,None],
              [{"colspan": 2,"type":"scatter"},None]], 
              subplot_titles=(None, None, None, None, "Raster plot - Coupling constant = 0.25", 
                              "Raster plot - Coupling constant = 0.60","Oscillation frequency per region")) 

    coupling_factor = np.linspace(0,0.8,81)
    cf0 = 0.4

    columns = ["subject", "mode", "coup","rPLV", "plv_mean", "plv_std", "amplitude_fpeak", "amplitude_fpeak_std", 
               "fpeak", "fpeak_std", "fpeak_cluster", "fpeak_cluster_std",
               "amplitude_fpeak_cluster", "amplitude_fpeak_cluster_std",
               "amplitude_norm_fpeak", "amplitude_norm_fpeak_std", 
               "amplitude_norm_fpeak_cluster", "amplitude_norm_fpeak_cluster_std"]
    
    colors = ['rgb(31,  119, 180)', 'rgb(255, 127, 14)',  'rgb(44,  160, 44)',
              'rgb(214,  39,  40)', 'rgb(148, 103, 189)', 'rgb(140,  86, 75)',
              'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)',
              'rgb(23,  190, 207)']
    
    rows = dict.fromkeys(columns)
    for key in columns:
        rows[key] = [] 
        
    subjects = ['NEMOS_035', 'NEMOS_049', 'NEMOS_050', 'NEMOS_058', 'NEMOS_059',
                'NEMOS_064', 'NEMOS_065', 'NEMOS_071', 'NEMOS_075', 'NEMOS_077']    

    #for ii,s in enumerate(subjects[:-1]):
    ii = 7
    s = "NEMOS_071"
    x = list(coupling_factor)
    # correlation 
    mu1  = correlation_mean[ii]
    std1 = correlation_std[ii]
    y_upper = list(mu1+std1)
    y_lower = list(mu1-std1)
    trace1 = go.Scatter(x=x,y=mu1,mode='markers+lines',
                        showlegend=False, line=dict(color=colors[0]))
    trace1_fill = go.Scatter(x=x+x[::-1], y=y_upper+y_lower[::-1],
                             fill='toself',hoverinfo="skip",showlegend=False, 
                             opacity=0.35, line=dict(color=colors[0]))

    # plv 
    mu2  = plv_alpha_mean[ii]
    std2 = plv_alpha_std[ii]
    y_upper = list(mu2+std2)
    y_lower = list(mu2-std2)
    trace2      = go.Scatter(x=x,y=mu2,mode='markers+lines',
                             showlegend=False,line=dict(color=colors[0]))
    
    trace2_fill = go.Scatter(x=x+x[::-1], y=y_upper+y_lower[::-1],
                             fill='toself',hoverinfo="skip",showlegend=False, 
                             opacity=0.35,line=dict(color=colors[0]))

    mu4  = fpeak_mean[ii]
    std4 = fpeak_std[ii]
    y_upper = list(mu4+std4)
    y_lower = list(mu4-std4)
    trace3      = go.Scatter(x=x,y=mu4,mode='markers+lines',showlegend=False,line=dict(color=colors[0])) #line=dict(color=colors[i]),name=subject)
    trace3_fill = go.Scatter(x=x+x[::-1], y=y_upper+y_lower[::-1],fill='toself',hoverinfo="skip",showlegend=False, opacity=0.35,line=dict(color=colors[0]))

    # axs['b'].set_title(s)

    cost = np.zeros(n)
    dcost = np.zeros(n)
    x0 = 10 
    y0 = correlation_mean[ii].max() 
    z0 = 0.45#plv_alpha_mean[ii].min()
    for i,(xx,yy,zz) in enumerate(zip(fpeak_mean[ii],correlation_mean[ii], plv_alpha_mean[ii])): # power_mean[0]/power_mean[0].max() )):
        cost[i] = 0.5*((xx-x0)/x0)**2+0.25*((yy-y0)/y0)**2+0.25*((zz-z0)/z0)**2
        dcost[i] = ((xx-x0)/x0)*std4[i]+0.5*((yy-y0)/y0)*std1[i]+0.5*((zz-z0)/z0)*std2[i]
    cost_smooth = gaussian_filter(cost, sigma=3)
    w = (fpeak_mean[ii]<10.5) & (fpeak_mean[ii]>9.5)

    aux = np.r_[True, cost_smooth[1:] < cost_smooth[:-1]] & np.r_[cost_smooth[:-1] < cost_smooth[1:], True]
    icmin = np.where(aux)[0][0]

    y_upper = list(cost+dcost)
    y_lower = list(cost-dcost)
    trace4      = go.Scatter(x=x,y=cost,mode='markers',showlegend=False,line=dict(color=colors[0]))#line=dict(color=colors[i]),name=subject)
    trace4_fill = go.Scatter(x=x+x[::-1], y=y_upper+y_lower[::-1],fill='toself',hoverinfo="skip",showlegend=False, opacity=0.35,line=dict(color=colors[0]))
    trace4_fit  = go.Scatter(x=x,y=cost_smooth,mode='lines',showlegend=False,line=dict(color=colors[3],width=3))

    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace1_fill, row=1,col=1)

    fig.add_trace(trace2, row=1, col=2)
    fig.add_trace(trace2_fill, row=1, col=2)

    fig.add_trace(trace3, row=2, col=1)
    fig.add_trace(trace3_fill, row=2, col=1)

    fig.add_trace(trace4, row=2, col=2)
    fig.add_trace(trace4_fill, row=2, col=2)
    fig.add_vline(x=x[icmin],line=dict(color=colors[2]))
    fig.add_vline(x=x[60],line=dict(color=colors[7],dash="dash"))

    fig.add_trace(trace4_fit, row=2, col=2)

    #fig.add_trace(trace5, row=2, col=2)
    # fig.show()

    fpeakmean = np.mean(fpeak[ii,icmin],axis=1)
    fpeakstd  = np.std(fpeak[ii,icmin],axis=1)
    fpeakmean_60 = np.mean(fpeak[ii,60],axis=1)
    fpeakstd_60  = np.std(fpeak[ii,60],axis=1)

    data = {} 
    data["Frequency"] = [] 
    data["Region"] = []
    data["Coupling factor"] =  []
    data["Frequency"].append(fpeakmean)
    data["Frequency"].append(fpeakmean_60)
    data["Region"].append( cingulum_rois*2 )
    data["Coupling factor"].append( [icmin]*22 )
    data["Coupling factor"].append( [60]*22 )

    for lb in data.keys():
        data[lb] = np.concatenate(data[lb])
    df = pd.DataFrame(data)

    import plotly.express as px
    data = []
    data1 =  go.Bar(x=df[df["Coupling factor"]==icmin]["Region"], 
                    y=df[df["Coupling factor"]==icmin]["Frequency"],
                    name="str(icmin)", 
                    marker_color=colors[2],
                    showlegend=False) 
    data2 =  go.Bar(x=df[df["Coupling factor"]==60]["Region"],    
                    y=df[df["Coupling factor"]==60]["Frequency"], 
                    name="60",
                    marker_color=colors[7],
                    showlegend=False) 

    fig.add_trace(data1,col=1,row=7)
    fig.add_trace(data2,col=1,row=7)

    # fig.update_layout(barmode="group")
    data = file_management.load_lzma(os.path.join( os.path.join(current_folder, s), f"data_spikes_{ii}_{icmin}.lzma"))
    w = ( data["tspikes_0"]>=10000 ) & (data["tspikes_0"]<12000)
    trace5 = go.Scatter(x=data["tspikes_0"][w]/1e3, y=data["ncell_0"][w],mode='markers',marker=dict(color=colors[0],size=1),showlegend=False)

    icmin =60
    data = file_management.load_lzma(os.path.join( os.path.join(current_folder, s), f"data_spikes_{ii}_{icmin}.lzma"))
    w = ( data["tspikes_0"]>=10000 ) & (data["tspikes_0"]<12000)
    trace6 = go.Scatter(x=data["tspikes_0"][w]/1e3, y=data["ncell_0"][w],mode='markers',marker=dict(color=colors[0],size=1),showlegend=False)

    fig.add_trace(trace5, row=3, col=1)
    fig.add_trace(trace6, row=5, col=1)

    fig.update_layout( height=1900, width=1400, template="plotly_white",
                       title=dict(text="Working point selection - Subject 08",font=dict(size=30)))

    fig.update_xaxes(title="Coupling factor [a.u.]", row=1, col=1, title_font=dict(size=25))
    fig.update_xaxes(title="Coupling factor [a.u.]", row=1, col=2, title_font=dict(size=25))
    fig.update_xaxes(title="Coupling factor [a.u.]", row=2, col=1, title_font=dict(size=25))
    fig.update_xaxes(title="Coupling factor [a.u.]", row=2, col=2, title_font=dict(size=25))
    fig.update_xaxes(title="Time [s]",               row=3, col=1, title_font=dict(size=25),
                     tickvals=[10.0,10.25,10.5,10.75,11.0,11.25,11.5,11.75,12.0])
    fig.update_xaxes(title="Time [s]",               row=5, col=1, title_font=dict(size=25),
                     tickvals=[10.0,10.25,10.5,10.75,11.0,11.25,11.5,11.75,12.0])

    fig.update_yaxes(title="Correlation eFC-sFC", row=1, col=1, title_font=dict(size=25))
    fig.update_yaxes(title="PLV mean",            row=1, col=2, title_font=dict(size=25))
    fig.update_yaxes(title="Frequency mean [Hz]", row=2, col=1, title_font=dict(size=25))
    fig.update_yaxes(title="Cost function",       row=2, col=2, title_font=dict(size=25))
    fig.update_yaxes(title="# Neuron",            row=3, col=1, title_font=dict(size=25))
    fig.update_yaxes(title="# Neuron",            row=5, col=1, title_font=dict(size=25))
    fig.update_yaxes(title="Frequency [Hz]",      row=7, col=1, title_font=dict(size=25))
    
    for i in range(7):
        fig.layout[f"xaxis{i+1}"]["tickfont"]["size"] = 17
        fig.layout[f"yaxis{i+1}"]["tickfont"]["size"] = 17
    for i,ann in enumerate(fig.layout.annotations):
        ann["font"]["size"] = 25
        
    if save_plot:
        fig.write_image(os.path.join(save_folder,title_plot))
    else:
        fig.show()