

def make_plot(save_plot=False, save_folder=None,title_plot=None):
    # figura de la current - frequency
    import pandas as pd
    import numpy as np
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px
    import os 
    import sys
    sys.path.append('/home/jaime/Desktop/hippocampus/files/')
    import file_management

    current_folder = os.getcwd()
    named_colorscales = px.colors.named_colorscales()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
              '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
              '#393b79', '#637939']

    cingulum_rois = ['Frontal_Mid_2_L', 'Frontal_Mid_2_R',
                    'Insula_L', 'Insula_R',
                    'Cingulate_Ant_L', 'Cingulate_Ant_R', 'Cingulate_Post_L', 'Cingulate_Post_R',
                    'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L',
                    'ParaHippocampal_R', 'Amygdala_L', 'Amygdala_R',
                    'Parietal_Sup_L', 'Parietal_Sup_R', 'Parietal_Inf_L',
                    'Parietal_Inf_R', 'Precuneus_L', 'Precuneus_R',
                    'Thalamus_L', 'Thalamus_R']

    subjects = ['NEMOS_035', 'NEMOS_049', 'NEMOS_050', 'NEMOS_058', 'NEMOS_059',
                'NEMOS_064', 'NEMOS_065', 'NEMOS_071', 'NEMOS_075', 'NEMOS_077']

    folder_data = '/home/jaime/Desktop/neuromodulacion/paper/cingulum_bundle/OzCz_densities_wstimfit2_examples/'

    titles = ["Subject 01","Subject 02","Subject 03","Subject 04","Subject 05",
              "Subject 06","Subject 07","Subject 08","Subject 09","Subject 10"]

    for ii,s in enumerate(subjects):
        folder = os.path.join(folder_data, s)
        filename = f"output_{s}_7_0_0.lzma"
        file = os.path.join(folder, filename)
        data = file_management.load_lzma(file)

        data["time"] = data["time"]/1e3
        w1 = (data["time"]>=14) & (data["time"]<16)
        w2 = (data["time"]>=16) & (data["time"]<=20)
        fig = make_subplots(rows=1,cols=1,column_titles=[f"Local field potentials - tACS - {titles[ii]}"])

        mean = np.mean(data["lfp"][0],axis=1)
        std  = np.std(data["lfp"][0],axis=1)
        ns = 0
        print( len(data["lfp"][0]) )
        for i in range(22):
            data["lfp"][0][i] = data["lfp"][0][i]-mean[i]
            
        amp = np.zeros(22)
        for i in range(1,22):
            amp[i] = 0.5*(mean[i-1]+ns*std[i-1])+0.5*(mean[i]+ns*std[i])
        amp = np.max(amp)/5
        traces = []

        for k in range(22):
            traces.append(go.Scatter(x=data["time"][w1], y=data["lfp"][0][k][w1]+(21-k)*amp,
                                     mode='lines',line=dict(color=colors[k]),showlegend=False, 
                                     name=cingulum_rois[k],opacity=0.5)) 
            fig.add_trace(traces[-1],col=1,row=1)

            traces.append(go.Scatter(x=data["time"][w2], y=data["lfp"][0][k][w2]+(21-k)*amp, 
                                     mode='lines',line=dict(color=colors[k]),showlegend=True, 
                                     name=cingulum_rois[k])) 
            fig.add_trace(traces[-1],col=1,row=1)
        
        fig.add_vline(x=16, opacity=0.6, line=dict(dash="dash", color="black"),  row=1, col=1)
        
        fig.update_layout( height=800, width=1400, template="plotly_white",
                           legend=dict(y=1.0,x=1.02,font=dict(size=19)))

        fig.layout.annotations[0]["font"] = dict(size=35)

        fig.update_xaxes(title="Time [s]",title_font=dict(size=35), row=1, col=1)
        fig.update_xaxes(tickvals=[14,14.5,15,15.5,16,16.5,17,17.5,18,18.5,19,19.5,20,20.5])
        fig.update_yaxes(tickvals=[])
        fig.layout["xaxis1"]["tickfont"]["size"] = 20
        if save_plot:
            fig.write_image(os.path.join(save_folder,f"lfps_stimulation_{ii}.png"))
        else:
            fig.show()