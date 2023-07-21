
def make_plot(save_plot=False,save_folder=None,title_plot=None):

    import pandas as pd
    import numpy as np
    import os
    from scipy.stats import rv_histogram

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.io as pio
    import plotly.express as px

    current_folder = '/home/jaime/Desktop/neuromodulacion/paper/one_node/'
    folder = os.path.join(current_folder, "tacs_densities")
    file1  = os.path.join(folder,"one_node_tacs_theoretical_densities2.txt")
    file2  = os.path.join(current_folder,"theoretical_distributions2.txt")

    dists_vals = pd.read_csv(file2, index_col=0, delimiter="\t")
    df = pd.read_csv(file1, delimiter="\t", index_col=0)
    df = df[df["fex"]>=5]
    df_spk = df.groupby(["simulation","histogram", "weight", "fex"]).mean().reset_index()

    x = np.linspace(-0.35,0.35,10001)
    sigma = 0.05
    mu = 0.075
    shift = 0.1

    # bimodal simetrica
    y1  = bimodal(x, 1.0, -mu, sigma, 1.0, mu,  sigma)
    y1s = bimodal(x, 1.0, -mu+shift, sigma, 1.0, mu+shift,  sigma)

    # bimodal asimetrica
    y2  = bimodal(x, 1.0, -mu-0.0415,       sigma, 3.5, mu-0.0415,  sigma)
    y2s = bimodal(x, 1.0, -mu-0.0415+shift, sigma, 3.5, mu-0.0415+shift,  sigma)

    # gaussian
    y3  = bimodal(x, 1.0, 0.0, sigma, 1.0, 0.0, sigma)
    y3s = bimodal(x, 1.0, shift, sigma, 1.0, shift, sigma)

    g = []
    dx = np.diff(x)[0]
    xn = np.array(list(x-dx/2)+[x[-1]+dx/2])
    np.random.seed(1993)
    for i,y in enumerate([y1,y1s,y2,y2s,y3,y3s]):
        rv = rv_histogram((y,xn))
        g.append(rv.rvs(size=1000))

    freq_min, freq_max, freq_colorscale = df_spk["fpeak"].min(), df_spk["fpeak"].max(), px.colors.diverging.balance
    pow_min, pow_max, pow_colorscale = df_spk["amplitude_fpeak"].min(), df_spk["amplitude_fpeak"].max(), px.colors.sequential.Sunsetdark

    freq_colorbar = dict(title="Hz", thickness=10, x=1)
    pow_colorbar = dict(title="dB", thickness=10, x=1.1)

    cmap_p = px.colors.qualitative.Pastel2
    cmap_s = px.colors.qualitative.Set2
    cmap_d = px.colors.qualitative.Dark2

    simulation = "lfp"
    columns, index = "fex", "weight"

    fig = make_subplots(rows=3, cols=5, column_titles=["Frequency", "Power", "", "Frequency", "Power"], #specs=[ [{type: "heatmap"},{},{},{},{}], [{},{},{},{},{}],[{},{},{},{},{}]], 
    vertical_spacing=0.08, horizontal_spacing=0.08,
    x_title = "Stimulation Frequency [Hz]    Electric Field [V/m]    Stimulation Frequency [Hz]")

    for i, dist in enumerate([y1, y1s, y2, y2s, y3, y3s]):
        row = i*2//4
        color = 0 if i % 2 == 0 else cmap_p[1]
        # Add histograms and theoretical distribution to the central column: blue left (centered), red right (shifted)
        fig.add_trace(go.Histogram(x=dists_vals.iloc[:, i].values, opacity=0.5, name="Dist " + str(i), marker=dict(color=cmap_p[i%2-1]), histnorm="probability density", showlegend=False), row=row+1, col=3)
        fig.add_trace(go.Scatter(x=x, y=dist, line=dict(color=cmap_s[i%2-1]), name="Theo " + str(i), showlegend=False), row=row+1, col=3)
        if i%2==0:
            fig.add_vline(x=0, opacity=0.6, line=dict(dash="dash",color="black", width=1),  row=row+1, col=3)
        else:
            fig.add_vline(x=shift, opacity=0.6, line=dict(dash="dash",color=cmap_d[i%2-1]), row=row+1, col=3)

        # Add arnold tongue
        w = (df_spk["simulation"] == simulation) & (df_spk["histogram"]==i)

        xx = np.sort(np.unique(df_spk[w][columns]))
        yy = np.sort(np.unique(df_spk[w][index]))
        fpeak = df_spk[w].pivot(index=index,columns=columns, values="fpeak").values
        amplitude_fpeak = df_spk[w].pivot(index=index,columns=columns, values="amplitude_fpeak").values

        s_col = 1 if i%2 == 0 else 4
        ss = True if i%2 == 0 else None
        fig.add_trace(go.Heatmap(x=xx, y=yy, z=fpeak, colorbar=dict(title="Hz", thickness=12, x=1.010,
                                 tickfont=dict(size=18)), showscale=ss, colorscale=freq_colorscale, zmin=freq_min, zmax=freq_max), row=row+1, col=s_col)
        fig.add_trace(go.Heatmap(x=xx, y=yy, z=amplitude_fpeak, colorbar=dict(title="dB", thickness=12, x=1.05,
                                 tickfont=dict(size=18)), showscale=ss, colorscale=pow_colorscale, zmin=pow_min, zmax=pow_max), row=row+1, col=s_col+1)

    fig.update_layout(barmode="overlay", template="plotly_white", height=770, width=1400, legend=dict(x=1.2),)
                    # xaxis11=dict(title="."+" "*36+"Stimulation Frequency [Hz]", anchor="free"), 
                    # xaxis15=dict(title="Stimulation Frequency [Hz]"+" "*32+"."))

    fig.update_yaxes(title_text="Stimulation intensity", title_font=dict(size=35),row=2, col=1)
    fig.update_yaxes(title_text="Stimulation intensity", title_font=dict(size=35),title_standoff=0.05,row=2, col=4)
    fig.update_yaxes(title_text="Probability density",   title_font=dict(size=35),row=2, col=3)
    #fig.update_xaxes(title_text="Electric Field [V/m]",  title_font=dict(size=35),row=3, col=3)

    fig.add_annotation(text="Simulations with <br>zero-mean distributions", x=0.02, y=1.19, xref="paper", yref="paper", showarrow=False, font=dict(size=32, color=cmap_d[-1]))

    fig.add_annotation(text="Simulations with <br>positive-shifted distributions", x=1.02, y=1.19, xref="paper", yref="paper", showarrow=False, font=dict(size=32, color=cmap_s[0]))
    
    fig.layout.annotations[0]["font"]["size"] = 20
    fig.layout.annotations[1]["font"]["size"] = 20
    fig.layout.annotations[2]["font"]["size"] = 20
    fig.layout.annotations[3]["font"]["size"] = 20
    fig.layout.annotations[4]["font"]["size"] = 35

    for i in range(15):
        fig.layout[f"xaxis{i+1}"]["tickfont"]["size"] = 18
        fig.layout[f"yaxis{i+1}"]["tickfont"]["size"] = 18
    if save_plot:
        fig.write_image(os.path.join(save_folder,title_plot))
    else:
        fig.show()

def bimodal(x, a1, mu1, sigma1, a2, mu2, sigma2):
    import numpy as np
    y = a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)) + a2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
    dx = np.diff(x)[0]
    return y/np.sum(y*dx)