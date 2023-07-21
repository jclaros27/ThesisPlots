

def make_plot(save_plot=False, save_folder=None,title_plot=None):

    import numpy as np 
    import os 
    import pandas as pd
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.express as px 
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    freq_colorscale = px.colors.diverging.balance
    pow_colorscale  = px.colors.sequential.Sunsetdark

    subjects = ['NEMOS_035', 'NEMOS_049', 'NEMOS_050', 'NEMOS_058', 'NEMOS_059','NEMOS_064', 'NEMOS_065', 'NEMOS_071', 'NEMOS_075', 'NEMOS_077']

    folder_coupled   = '/home/jaime/Desktop/neuromodulacion/paper/cingulum_bundle/OzCz_densities_coupled'
    folder_precuneus = '/home/jaime/Desktop/neuromodulacion/paper/cingulum_bundle/OzCz_densities_precuneus'
    folder_cingulate = '/home/jaime/Desktop/neuromodulacion/paper/cingulum_bundle/OzCz_densities_cingulate'
    folder_uncoupled = '/home/jaime/Desktop/neuromodulacion/paper/cingulum_bundle/OzCz_densities'
    folder_coupled_no_precuneus = '/home/jaime/Desktop/neuromodulacion/paper/cingulum_bundle/OzCz_densities_no_precuneus'

    # title = f"stimulation_OzCz_densities_nodes_uncoupled.txt"
    # new_df1 = pd.read_csv(os.path.join(folder_uncoupled, title), sep="\t")
    title = f"stimulation_OzCz_densities_nodes_precuneus.txt"
    new_df2 = pd.read_csv(os.path.join(folder_precuneus, title), sep="\t")
    title = f"stimulation_OzCz_densities_nodes_coupled.txt"
    new_df3 = pd.read_csv(os.path.join(folder_coupled, title), sep="\t")
    # title = f"stimulation_OzCz_densities_nodes_cingulate.txt"
    # new_df4 = pd.read_csv(os.path.join(folder_cingulate, title), sep="\t")
    title = f"stimulation_OzCz_densities_nodes_no_precuneus.txt"
    new_df5 = pd.read_csv(os.path.join(folder_coupled_no_precuneus, title), sep="\t")

    df1 = new_df2[new_df3["node"] == "Precuneus_R"] # only precuneus stimulated
    df2 = new_df3[new_df2["node"] == "Precuneus_R"] # all nodes stimulated
    df3 = new_df5[new_df5["node"] == "Precuneus_R"] # all nodes stimulated but the precuneus
    aux1 = df1[df1["node"]=="Precuneus_R"].groupby(["w","fex"])["fpeak","amp_fpeak"].mean().reset_index()   
    aux2 = df2[df2["node"]=="Precuneus_R"].groupby(["w","fex"])["fpeak","amp_fpeak"].mean().reset_index()
    aux3 = df3[df3["node"]=="Precuneus_R"].groupby(["w","fex"])["fpeak","amp_fpeak"].mean().reset_index()

    matrix11 = aux1.pivot(index="w", columns="fex", values="fpeak")
    matrix21 = aux1.pivot(index="w", columns="fex", values="amp_fpeak")
    matrix12 = aux2.pivot(index="w", columns="fex", values="fpeak")
    matrix22 = aux2.pivot(index="w", columns="fex", values="amp_fpeak")
    matrix13 = aux3.pivot(index="w", columns="fex", values="fpeak")
    matrix23 = aux3.pivot(index="w", columns="fex", values="amp_fpeak")

    vmin_fpeak = np.min([np.min(matrix11), np.min(matrix12),np.min(matrix13)])
    vmax_fpeak = np.max([np.max(matrix11), np.max(matrix12),np.max(matrix13)])
    vmin_amp_fpeak = np.min([np.min(matrix21), np.min(matrix22),np.min(matrix12)])
    vmax_amp_fpeak = np.max([np.max(matrix21), np.max(matrix22),np.max(matrix23)])

    x = aux1["fex"].unique()
    y = aux1["w"].unique()

    fig = make_subplots(rows=2, cols=3,
                        column_titles=["Only Precuneus_R", "All ROIs", "All ROIs <br>no Precuneus_R"],
                        vertical_spacing=0.1,horizontal_spacing=0.1,
                        x_title="Frequency [Hz]", y_title="Calibration constant &#923;")

    trace11 = go.Heatmap(x=x,y=y, z=matrix11, zmin=vmin_fpeak, zmax=vmax_fpeak, showscale=False, colorscale=freq_colorscale)
    trace12 = go.Heatmap(x=x,y=y, z=matrix12, zmin=vmin_fpeak, zmax=vmax_fpeak, colorscale=freq_colorscale,colorbar=dict(len=0.5, title="Hz", thickness=12, y=0.75+0.05, tickfont=dict(size=18)))#, y=0.75+0.075))
    trace13 = go.Heatmap(x=x,y=y, z=matrix13, zmin=vmin_fpeak, zmax=vmax_fpeak, showscale=False, colorscale=freq_colorscale)

    trace21 = go.Heatmap(x=x,y=y, z=matrix21, zmin = vmin_amp_fpeak, zmax=vmax_amp_fpeak, showscale=False, colorscale=pow_colorscale)
    trace22 = go.Heatmap(x=x,y=y, z=matrix22, zmin = vmin_amp_fpeak, zmax=vmax_amp_fpeak, colorscale=pow_colorscale, colorbar=dict(len=0.5, title="dB", thickness=12,y =0.25,
    tickfont=dict(size=18)))
    trace23 = go.Heatmap(x=x,y=y, z=matrix23, zmin = vmin_amp_fpeak, zmax=vmax_amp_fpeak,
    showscale=False, colorscale=pow_colorscale)

    fig.add_trace(trace11, row=1, col=1)
    fig.add_trace(trace12, row=1, col=2)
    fig.add_trace(trace13, row=1, col=3)

    fig.add_trace(trace21, row=2, col=1)
    fig.add_trace(trace22, row=2, col=2)
    fig.add_trace(trace23, row=2, col=3)

    # fig.update_xaxes(title_text="Frequency [Hz]", row=2, col=1)
    # fig.update_xaxes(title_text="Frequency [Hz]", row=2, col=2)
    # fig.update_xaxes(title_text="Frequency [Hz]", row=2, col=3)

    # fig.update_yaxes(title_text=r"."+" "*40 + r"Calibration constant ($\Lambda$)", row=2, col=1)
    # fig.update_yaxes(title_text=r"Calibration constant ($\Lambda$)", row=2, col=1)
    fig.update_layout(height=800, width=1400, 
                        annotations=dict(
            # Don't specify y position, because yanchor="middle" should do it
            x=1.22,
            align="right",
            valign="top",
            text='Colorbar Title',
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            # Parameter textangle allow you to rotate annotation how you want
            textangle=90
            ))
    #                   coloraxis=dict(colorscale='deep_r', colorbar_x=0.43, colorbar_thickness=23),
    #                   coloraxis2=dict(colorscale='matter_r', colorbar_x=1.0075, colorbar_thickness=23))
    for i,ann in enumerate(fig.layout.annotations):
        ann["font"]["size"] = 35
    
    for i in range(6):
        fig.layout[f"xaxis{i+1}"]["tickfont"]["size"] = 20
        fig.layout[f"yaxis{i+1}"]["tickfont"]["size"] = 20

    if save_plot:
        fig.write_image(os.path.join(save_folder,title_plot))
    else:
        fig.show()