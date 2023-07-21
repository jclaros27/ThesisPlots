    
    
def make_plot(save_plot=False, save_folder=None, title_plot=None):
    import numpy as np
    import pandas as pd
    import os
    # import plotly.graph_objects as go
    import plotly.graph_objs as go
    import plotly.io as pio

    from plotly.subplots import make_subplots
    import plotly.express as px
    import glob
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    ######  SPIKING

    ## 0b. Load SPK data
    sim_tag = "stimWfit\\"
    fname = "stimulation_OzCz_densities_nodes.txt"
    folder = "/home/jaime/Desktop/neuromodulacion/paper/cingulum_bundle/OzCz_densities_wstimfit"

    # cargar los datos
    # stimWfit_spk_pre = pd.read_csv(folder + sim_tag + fname, delimiter="\t", index_col=0)
    stimWfit_spk_pre = pd.read_csv(os.path.join(folder, fname), delimiter="\t",index_col=0)
    empCluster_rois = ['Precentral_L', 'Frontal_Sup_2_L', 'Frontal_Sup_2_R', 'Frontal_Mid_2_L',
                    'Frontal_Inf_Oper_L', 'Frontal_Inf_Oper_R', 'Frontal_Inf_Tri_L', 'Frontal_Inf_Tri_R',
                    'Frontal_Inf_Orb_2_L', 'Rolandic_Oper_L', 'Rolandic_Oper_R', 'Frontal_Sup_Medial_L',
                    'Frontal_Sup_Medial_R', 'Rectus_L', 'OFCmed_L', 'Insula_L', 'Insula_R', 'Cingulate_Ant_L',
                    'Cingulate_Ant_R',
                    'Hippocampus_L', 'Hippocampus_R', 'ParaHippocampal_L', 'ParaHippocampal_R',
                    'Amygdala_L', 'Calcarine_L', 'Calcarine_R', 'Cuneus_L', 'Cuneus_R', 'Lingual_L', 'Lingual_R',
                    'Occipital_Sup_R', 'Occipital_Mid_L', 'Occipital_Mid_R', 'Occipital_Inf_L',
                    'Occipital_Inf_R', 'Fusiform_L', 'Fusiform_R', 'Postcentral_L', 'Parietal_Sup_R',
                    'Parietal_Inf_R', 'Angular_R', 'Precuneus_R', 'Temporal_Sup_L',
                    'Temporal_Sup_R', 'Temporal_Pole_Sup_L', 'Temporal_Pole_Sup_R', 'Temporal_Mid_L',
                    'Temporal_Mid_R', 'Temporal_Pole_Mid_L', 'Temporal_Inf_L', 'Temporal_Inf_R']
    stimWfit_spk = stimWfit_spk_pre.loc[stimWfit_spk_pre["node"].isin(empCluster_rois)].copy()

    stimWfit_spk = stimWfit_spk.groupby(["subject", "trial", "w"]).mean().reset_index()

    n_trials = stimWfit_spk["trial"].max() + 1

    # Calculate percentage
    baseline_spk = stimWfit_spk.loc[stimWfit_spk["w"] == 0].groupby("subject").mean().reset_index()

    stimWfit_spk["percent"] = [(row["amp_fpeak"] - baseline_spk.loc[baseline_spk["subject"] == row["subject"]].amp_fpeak.values[0]) / baseline_spk.loc[baseline_spk["subject"] == row["subject"]].amp_fpeak.values[0] * 100 for i, row in stimWfit_spk.iterrows()]
    # stimWfit_spk["percent"] = [((row["amp_fpeak"] / baseline_spk.loc[baseline_spk["subject"] == row["subject"]].amp_fpeak.values[0]) - 1) * 100 for i, row in stimWfit_spk.iterrows()]

    stimWfit_spk_avg = stimWfit_spk.groupby(["subject", "w"]).mean().reset_index()
    stimWfit_spk_err = stimWfit_spk.groupby(["subject", "w"]).std().reset_index()

    # Just show half the calibration constants to make a clearer picture
    include_w = [sorted(set(stimWfit_spk_avg.w))[0]] + sorted(set(stimWfit_spk_avg.w))[3:-4:1]
    stimWfit_spk_sub = stimWfit_spk_avg[stimWfit_spk_avg["w"].isin(include_w)]
    stimWfit_spk_err_sub = stimWfit_spk_avg[stimWfit_spk_err["w"].isin(include_w)]

    ## Substitute internal coding NEMOS by subject
    labels = []
    for i, subj in enumerate(sorted(set(stimWfit_spk_sub.subject))):
        new_name = "Subject " + str(i+1).zfill(2)
        labels.append(new_name)
        stimWfit_spk_sub["subject"].loc[stimWfit_spk_sub["subject"] == subj] = new_name
        #stimWfit_spk_err_sub["subject"].loc[stimWfit_spk_sub["subject"] == subj] = new_name

    stimWfit_spk_sub["percent_err"] = stimWfit_spk_err_sub["percent"]/np.sqrt(n_trials)
    ### A. Scatter plot with Mean line for percentage
    sim="SPK"
    #fig = px.Scatter(stimWfit_spk_sub, x="w", y="percent", error_y="percent_err", error_y_mode = "band",color="subject", labels={"subject": ""}, log_x=False)#, markers=True)
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)',
            'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
            'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)',
            'rgb(23, 190, 207)']

    fig = go.Figure()
    for i, subject in enumerate(labels):
        subdata = stimWfit_spk_sub[stimWfit_spk_sub["subject"] == subject]
        x, y, yerr = subdata["w"].values, subdata["percent"].values, subdata["percent_err"].values
        y_upper, y_lower = y+yerr, y-yerr

        x, y = list(x), list(y)
        yerr = list(yerr)
        y_upper, y_lower = list(y_upper), list(y_lower)

        fig.add_trace(go.Scatter(x=x,y=y,mode='lines',
                                 line=dict(color=colors[i],width=5), name=subject))
        fig.add_trace(go.Scatter(x=x,y=y,mode='markers',
                                 line=dict(color=colors[i]),marker=dict(size=15),showlegend=False))

        fig.add_trace(go.Scatter(x=x+x[::-1], y=y_upper+y_lower[::-1], fill='toself',
                                 hoverinfo="skip",fillcolor=colors[i],line=dict(color=colors[i]),
                                 showlegend=False, opacity=0.15))

    w = np.asarray(stimWfit_spk_sub.groupby("w").mean().reset_index()["w"])
    mean = np.asarray(stimWfit_spk_sub.groupby("w").mean()["percent"])
    median = np.asarray(stimWfit_spk_sub.groupby("w").median()["percent"])
    err = np.asarray(stimWfit_spk_err_sub.groupby("w").median()["percent"])

    fig.add_trace(go.Scatter(x=w, y=mean, mode="lines", name="mean", 
                             line=dict(color='darkslategray', width=5)))
    fig.add_trace(go.Scatter(x=w, y=mean, mode="markers", name="mean", showlegend=False,
                             line=dict(color='darkslategray'),marker=dict(size=15)))

    fig.add_shape(x0=32.5, x1=37.5, y0=-50, y1=100, fillcolor="lightgray", 
                  opacity=0.3, line=dict(width=1))

    fig.update_layout(height=940, width=1400, template="plotly_white")
    fig.update_xaxes(title="Calibration constant &#923;",  title_font=dict(size=35))
    fig.update_yaxes(title="Alpha band power change (%)",  title_font=dict(size=35))
    fig.update_layout(legend=dict(font=dict(size=25),x=1.0,y=1.0))
    fig.layout["xaxis1"]["tickfont"]["size"] = 20
    fig.layout["yaxis1"]["tickfont"]["size"] = 20
    fig.layout["yaxis1"]["tickvals"] = [-50, -25, 0, 25, 50, 75, 100, 125, 150, 175]

    if save_plot:
        fig.write_image(os.path.join(save_folder,title_plot))
    else:
        fig.show()