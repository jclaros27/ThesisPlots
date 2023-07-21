

def make_plot(save_folder=None,save_plot=False,title_plot=None):

    # figura de la current - frequency
    import numpy as np
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import os
    import sys
    sys.path.append('/home/jaime/Desktop/hippocampus/files/')
    import file_management

    folder = '/home/jaime/Desktop/neuromodulacion/paper/one_node/current-frequency-curve/'
    data = []
    for i in range(10):
        filename = f"data_{i}.lzma"
        file = os.path.join(folder,filename)  
        data.append( file_management.load_lzma(file) )

    fpeak = []
    psd = [] 
    for i in range(10):
        fpeak.append( data[i]["fpeak"] )
        psd.append( data[i]["psd"])
    fpeak = np.array(fpeak)
    fpeak_mean = np.mean(fpeak,axis=0)
    fpeak_err  = np.std(fpeak,axis=0)
    rate_seq = np.linspace(1600,3200,21)

    #(10, 21, 2049)
    psd = np.array(psd).real
    psd_mean = np.mean(psd,axis=0)
    psd_err = np.std(psd,axis=0)
    fr = data[0]["fr"]

    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)',
            'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
            'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)',
            'rgb(23, 190, 207)']

    lfp1 = data[0]["lfp"][5]
    lfp2 = data[0]["lfp"][10]
    lfp3 = data[0]["lfp"][15]

    time = np.arange(0,len(lfp1),1)/1e3
    
    w = (time>=8) & (time<=10)
    fig = make_subplots(
        rows=3, cols=5,
        specs=[[{"type":"scatter","rowspan":3,"colspan":2},None, {"type":"scatter","colspan":2},None,{"type":"scatter"}],
            [None, None, {"type":"scatter","colspan":2},None, {"type":"scatter"}],
            [None, None, {"type":"scatter","colspan":2},None, {"type":"scatter"}]],
            subplot_titles =("", "Rate = 2000 Hz", "","Rate = 2400 Hz","", "Rate = 2800 Hz",""),
            vertical_spacing=0.11, horizontal_spacing=0.1
    )

    trace00 =  go.Scatter(x=rate_seq, y=fpeak_mean,showlegend=False, mode="lines", line=dict(color=colors[0])) 
    trace01 =  go.Scatter(x=rate_seq, y=fpeak_mean,showlegend=False, mode="markers",line=dict(color=colors[0])) 

    x, y= list(rate_seq), list(fpeak_mean)
    yerr = list(fpeak_err)
    y_upper = fpeak_mean+fpeak_err
    y_lower = fpeak_mean-fpeak_err
    y_upper, y_lower = list(y_upper), list(y_lower)

    trace02 = go.Scatter(x=x+x[::-1], # x, then x reversed
                                y=y_upper+y_lower[::-1], # upper, then lower reversed
                                fill='toself',
                                hoverinfo="skip",
                                fillcolor=colors[0],
                                line=dict(color=colors[0]),
                                showlegend=False, 
                                opacity=0.15)

    trace1 =  go.Scatter(x=time[w], y=lfp1[w], showlegend=False, name=f"{int(rate_seq[5])} Hz", line=dict(color=colors[1])) 
    trace2 =  go.Scatter(x=time[w], y=lfp2[w], showlegend=False, name=f"{int(rate_seq[10])} Hz",line=dict(color=colors[2])) 
    trace3 =  go.Scatter(x=time[w], y=lfp3[w], showlegend=False, name=f"{int(rate_seq[15])} Hz",line=dict(color=colors[3])) 

    fig.add_trace(trace00,  row=1, col=1)
    fig.add_trace(trace01,  row=1, col=1)
    fig.add_trace(trace02,  row=1, col=1)

    fig.add_trace(trace1, row=1, col=3)
    fig.add_trace(trace2, row=2, col=3)
    fig.add_trace(trace3, row=3, col=3)


    trace = []
    w = (fr>0) & (fr<40)
    for ii,i in enumerate([5,10,15]):
        x, y = list(fr[w]), list(psd_mean[i,w])
        yerr = list(psd_err[i,w])
        y_upper = np.array(y)+np.array(yerr)
        y_lower = np.array(y)-np.array(yerr)
        y_upper, y_lower = list(y_upper), list(y_lower)

        trace.append( go.Scatter(x=x, y=y,  showlegend=False, line=dict(color=colors[ii+1])) )

        trace.append( go.Scatter(x=x+x[::-1], # x, then x reversed
                                    y=y_upper+y_lower[::-1], # upper, then lower reversed
                                    fill='toself',
                                    hoverinfo="skip",
                                    fillcolor=colors[0],
                                    line=dict(color=colors[ii+1]),
                                    showlegend=False, 
                                    opacity=0.15))
        
        fig.add_trace(trace[-2], row=ii+1, col=5)
        fig.add_trace(trace[-1], row=ii+1, col=5)

    fig.update_layout( height=500, width=1100, template="plotly_white")#,
                    #    title_text="Single node dynamics")

    fig.update_xaxes(title_text="Rate [Hz]",     title_font=dict(size=35),row=1, col=1)
    fig.update_xaxes(title_text="Time [ms]",     title_font=dict(size=35),row=3, col=3)
    fig.update_xaxes(title_text="Frequency [Hz]",title_font=dict(size=35),row=3, col=5)

    fig.update_yaxes(title_text="Oscillatory frequency [Hz]", title_font=dict(size=35),row=1, col=1)
    fig.update_yaxes(title_text="LFP [pA]",                   title_font=dict(size=35),row=2, col=3)#, matches='y6')
    fig.update_yaxes(title_text="PSD [pA<sup>2</sup>/Hz]",    title_font=dict(size=35),row=2, col=5)#, matches='y7')
    fig.update_layout(template="plotly_white", height=650, width=1400, font=dict(size=20))

    for i in range(7):
        fig.layout[f"xaxis{i+1}"]["tickfont"]["size"] = 18
        fig.layout[f"yaxis{i+1}"]["tickfont"]["size"] = 18

    for i in range(3):
        fig.layout.annotations[i]["font"]["size"]= 20

    if save_plot:
        fig.write_image(os.path.join(save_folder,title_plot))
    else:
        fig.show()