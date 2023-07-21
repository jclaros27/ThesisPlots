

def make_plot(save_plot=False, save_folder=None,title_plot=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    from scipy import stats
    import numpy as np
    import sys
    import os 
    sys.path.append('/home/jaime/Desktop/hippocampus/files/')
    import file_management

    folder = '/home/jaime/Desktop/neuromodulacion/paper/one_node/tacs_densities_examples/data'
    files = ["data_spikes_0_0_0_no_filter.lzma","data_spikes_1_0_0_no_filter.lzma"]

    data = []
    data.append(file_management.load_lzma(os.path.join(folder, files[0])))
    data.append(file_management.load_lzma(os.path.join(folder, files[1])))

    files = ["data_lfp_0_0_0_no_filter.lzma","data_lfp_1_0_0_no_filter.lzma"]
    lfp = []
    lfp.append(file_management.load_lzma(os.path.join(folder, files[0])))
    lfp.append(file_management.load_lzma(os.path.join(folder, files[1])))


    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)']
    fig = make_subplots(rows=3, cols=2, column_titles=["Zero-mean <br>symmetric bimodal", "Positive-shifted <br>symmetric bimodal"], shared_xaxes=False,
                        vertical_spacing=0.1, horizontal_spacing=0.11,
                        x_title="Period interval [ms]")#, y_title="Probability density") 
    
    cmap_p = px.colors.qualitative.Pastel2
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)']
    legend = ["50","100","150"]
    
    xseq = np.linspace(0, 100, 1001)

    for col, i in enumerate(range(2)):
        for row, j in enumerate([2,1,0]):

            xx = np.mod(data[i]["time"][j][data[i]["time"][j]>5000],100)
            if i ==0 :
                fig.add_trace(go.Histogram(x=xx, histnorm='probability density', marker_color=colors[j],opacity=0.25,showlegend=True,name=legend[j]), row=row+1, col=col+1)
                
                bins = len(np.histogram_bin_edges(xx, bins="fd"))
                _ = np.histogram(xx, bins = bins, density=True)

                counts, x = _[0], _[1]
                x = x[1:]-np.diff(x)[0]/2
                dx = np.diff(x)[0]
                kde = stats.gaussian_kde(xx)
                yfit = kde(xseq)

                fig.add_trace(go.Scatter(x=xseq,y=yfit,line=dict(color=colors[j]),showlegend=False),row=row+1,col=col+1)
            else:
                fig.add_trace(go.Histogram(x=xx, histnorm='probability density',marker_color=colors[j],opacity=0.25,showlegend=False), row=row+1, col=col+1)

                bins = len(np.histogram_bin_edges(xx, bins="fd"))
                _ = np.histogram(xx, bins = bins, density=True)

                counts, x = _[0], _[1]
                x = x[1:]-np.diff(x)[0]/2
                dx = np.diff(x)[0]
                kde = stats.gaussian_kde(xx)
                yfit = kde(xseq)

                fig.add_trace(go.Scatter(x=xseq,y=yfit,line=dict(color=colors[j]),showlegend=False),row=row+1,col=col+1)

    fig.update_yaxes(range=[0,0.035], row=row+1, col=col+1)
    fig.update_xaxes(tickvals=[0,25,50,75,100])

    fig.update_layout(barmode="overlay", template="plotly_white", height=960, width=1400, 
                      legend=dict(title="Stimulation <br>intensity",y=1.0,x=1.02,font=dict(size=25)))

    # fig.update_yaxes( title="Density", tickvals=[0,0.01,0.02,0.03,0.04])
    #fig.layout.annotations[-1]["x"] = -0.02
    # position of ylabel for the subplots below
    fig.update_yaxes(title_text="Probability density", title_font=dict(size=35),row=2, col=1)

    for i,ann in enumerate(fig.layout.annotations):
        ann["font"]["size"] = 35
    
    for i in range(6):
        fig.layout[f"xaxis{i+1}"]["tickfont"]["size"] = 18
        fig.layout[f"yaxis{i+1}"]["tickfont"]["size"] = 18
        fig.layout[f"yaxis{i+1}"]["range"] = [0,0.03]
        fig.layout[f"yaxis{i+1}"]["tickvals"] = [0,0.01,0.02,0.03]

    print(fig.layout.annotations[-1])
    if save_plot:
        fig.write_image(os.path.join(save_folder,title_plot))
    else:
        fig.show()
