


def make_plot(save_plot=True, save_folder=None,title_plot=None):
    
    import numpy as np
    from scipy import io
    import os
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import scipy.stats as stats

    # current_dir = os.getcwd()
    # folder = os.path.join(current_dir, "files")
    # folder = os.path.join(folder, "histogramas")
    # folder_densities = os.path.join(folder, "Densities")
    folder = '/home/jaime/Desktop/neuromodulacion/paper/cingulum_bundle/files/histogramas'
    folder_densities = '/home/jaime/Desktop/neuromodulacion/paper/cingulum_bundle/files/histogramas/Densities'
    subjects = ["NEMOS_035", "NEMOS_049", "NEMOS_050", "NEMOS_058", "NEMOS_059",
                "NEMOS_064", "NEMOS_065", "NEMOS_071", "NEMOS_075", "NEMOS_077"]

    models = ["OzCzModel"]

    title = "-ROIvals_orth-roast_"
    data = dict.fromkeys(models)
    data_ = dict.fromkeys(models)
    data_reduced = dict.fromkeys(models)
    for model in models:
        data[model] = dict.fromkeys(subjects)
        data_[model] = dict.fromkeys(subjects)
        data_reduced[model] = dict.fromkeys(subjects)
    for model in models:
        for s in subjects:
            file = os.path.join(folder,s+title+model+".mat")
            data[model][s] = io.loadmat(file)

    cingulum_rois = ['Frontal_Mid_2_L', 
                    'Frontal_Mid_2_R',
                    'Insula_L', 
                    'Insula_R',
                    'Cingulate_Ant_L', 
                    'Cingulate_Ant_R', 
                    'Cingulate_Post_L', 
                    'Cingulate_Post_R',
                    'Hippocampus_L', 
                    'Hippocampus_R', 
                    'ParaHippocampal_L',
                    'ParaHippocampal_R', 
                    'Amygdala_L', 
                    'Amygdala_R',
                    'Parietal_Sup_L', 
                    'Parietal_Sup_R', 
                    'Parietal_Inf_L',
                    'Parietal_Inf_R', 
                    'Precuneus_L', 
                    'Precuneus_R',
                    'Thalamus_L', 
                    'Thalamus_R']

    files_dir = '/home/jaime/Desktop/neuromodulacion/paper/cingulum_bundle/files/'
    files_dir = os.path.join(files_dir, f"{subjects[0]}_AAL2_pass")
    files = ['tract_lengths.txt','weights.txt','centres.txt']
    SClabs = []

    for line in open(os.path.join(files_dir, files[2])):
        SClabs.append( line.split()[0] )

    SC_cb_idx = [SClabs.index(roi) for roi in cingulum_rois]  # find indexes in SClabs that matches cortical_rois
    SClabs = np.array( SClabs )

    nnodes = 120
    for m in models:
        for s in subjects:
            data_[m][s] = [ [] for i in range(nnodes)]
            a = data[m][s]["ROIvals"][0]
            for i,a1 in enumerate(a): # por nodo
                data_[m][s][i] = np.concatenate(np.concatenate(a1))
                
    nnodes = 22 
    for m in models: 
        for s in subjects: 
            data_reduced[m][s] = [ [] for i in range(nnodes)]
            for i, ii in enumerate(SC_cb_idx): 
                data_reduced[m][s][i] = data_[m][s][ii]

    data_accumulated = dict.fromkeys(models)
    for model in models: 
        data_accumulated[model] = [ [] for i in range(len(SC_cb_idx)) ]

    for m in models: 
        for s in subjects:
            for i,ii in enumerate(SC_cb_idx): 
                data_accumulated[m][i].append( data_[m][s][ii] )
                
    for i in range(len(SC_cb_idx)):
        data_accumulated[m][i] = np.concatenate(data_accumulated[m][i])

    titles = cingulum_rois[:18]+[" "] +cingulum_rois[18:]+[" "]

    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)',
            'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
            'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)',
            'rgb(23, 190, 207)']

    fig = make_subplots(rows=4, cols=6, subplot_titles=titles,
                        x_title="Normal components of the electric field [V/m]",
                        y_title="Probability density")

    for m in models:
        element = data_accumulated[m]
        xmin = -0.4#np.min(np.concatenate(element))
        xmax = 0.4#np.max(np.concatenate(element))

        k = 0 
        for i in range(3):
            for j in range(6):
                example = element[k]
                bins = len(np.histogram_bin_edges(example, bins="fd"))
                y,x = np.histogram(example,bins=bins,density=True)
                datax  = x[1:]-np.diff(x)[0]/2.0
                datay  = y

                datap1 = [np.max(y), np.median(example), np.std(example),
                        np.max(y), np.median(example), np.std(example)]
                datap2 = [np.max(y), np.median(example), np.std(example)]
                xseq = np.linspace(xmin, xmax, 1001)

                dx = np.diff(xseq)[0]
                fig.add_trace(go.Histogram(x=example, nbinsx=bins,opacity=0.25, marker=dict(color=colors[0]), histnorm="probability density", showlegend=False), row=i+1, col=j+1)
                _ = np.histogram(example, bins = bins, density=True)
                counts, x = _[0], _[1]
                x = x[1:]-np.diff(x)[0]/2
                kde = stats.gaussian_kde(example)
                yfit = kde(xseq)
                yfit = yfit/np.sum(yfit*dx)

                fig.add_trace(go.Scatter(x=xseq,y=yfit,line=dict(color=colors[0]),showlegend=False),row=i+1,col=j+1)
                fig.add_vline(x=np.mean(example), opacity=0.6, line=dict(color="red", width=1),  row=i+1, col=j+1)

                fig.add_vline(x=0,opacity=0.6, line=dict(dash="dash",color="black",width=1),row=i+1,col=j+1)
                k += 1

        for i in range(3,4):
            for j in range(1,5):
                example = element[k]
                bins = len(np.histogram_bin_edges(example, bins="fd"))
                y,x = np.histogram(example,bins=bins,density=True)
                datax  = x[1:]-np.diff(x)[0]/2.0
                datay  = y

                datap1 = [np.max(y), np.median(example), np.std(example),
                        np.max(y), np.median(example), np.std(example)]
                datap2 = [np.max(y), np.median(example), np.std(example)]
                xseq = np.linspace(xmin, xmax, 1001)

                dx = np.diff(xseq)[0]
                fig.add_trace(go.Histogram(x=example, nbinsx=bins, opacity=0.25, marker=dict(color=colors[0]), histnorm="probability density", showlegend=False), row=i+1, col=j+1)
                _ = np.histogram(example, bins = bins, density=True)

                counts, x = _[0], _[1]
                x = x[1:]-np.diff(x)[0]/2
                kde = stats.gaussian_kde(example)
                yfit = kde(xseq)
                yfit = yfit/np.sum(yfit*dx)

                fig.add_trace(go.Scatter(x=xseq,y=yfit,line=dict(color=colors[0]),showlegend=False),row=i+1,col=j+1)
                fig.add_vline(x=np.mean(example), opacity=0.6,line=dict(color="red", width=1),  row=i+1, col=j+1)
                fig.add_vline(x=0 ,opacity=0.6,line=dict(dash="dash",color=colors[0],width=1),row=i+1,col=j+1)
                k += 1
        #plt.savefig(m+"_"+s+f"_{k}.png",dpi=300,bbox_inches='tight')
        #plt.close()
    fig.update_layout(barmode="overlay", template="plotly_white", height=1000, width=1400)
    fig.update_annotations(font_size=20)
    n = len(fig.layout.annotations)
    for i in range(n-2):
        fig.layout.annotations[i]["y"] += 0.04

    #         yaxis13=dict(title='Probability Density',anchor="free"),
    #         xaxis21=dict(title='Normal components of the electric field [V/m]',anchor="free"),
    #                      title_font=dict(size=20))
    fig.layout.annotations[-1]["font"]["size"] = 35
    fig.layout.annotations[-2]["font"]["size"] = 35
    for i in range(24):
        fig.layout[f"xaxis{i+1}"]["tickvals"] = [-0.2,0.0,0.2]
        fig.layout[f"xaxis{i+1}"]["tickfont"]["size"] = 18
        fig.layout[f"yaxis{i+1}"]["tickfont"]["size"] = 18

    if save_plot:
        fig.write_image(os.path.join(save_folder,title_plot))
    else:
        fig.show()
