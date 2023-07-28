def make_plot(save_plot=False,save_folder=None,title_plot=None):
    import numpy as np 
    import matplotlib.pyplot as plt
    import os
    import sys 
    import plot_parameters as pc
    from matplotlib.colorbar import Colorbar 
    sys.path.append('/home/jaime/Desktop/hippocampus/files/')
    import file_management

    # Your main script or Jupyter Notebook
    # Set the custom parameters for matplotlib
    plt.rcParams['font.size'] = pc.fontsize_labels
    plt.rcParams['axes.labelsize'] = pc.fontsize_labels
    plt.rcParams['axes.titlesize'] = pc.fontsize_titles
    plt.rcParams['legend.fontsize'] = pc.fontsize_labels
    plt.rcParams['xtick.labelsize'] = pc.fontsize_labels
    plt.rcParams['ytick.labelsize'] = pc.fontsize_labels

    if pc.use_latex:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx}'

    xticks = [0,np.pi/2,np.pi,3*np.pi/2,2*np.pi]
    xtickslabels = [r"$0$",r"$\pi/2$",r"$\pi$",r"$3\pi/2$",r"$2\pi$"]
    yticks_colorbar = [-np.pi, -np.pi/2,0,np.pi/2,np.pi]
    ytickslabels_colorbar = [r"$-\pi$",r"$-\pi/2$","0",r"$\pi/2$",r"$\pi$"]

    folder = '/home/jaime/Desktop/synchronization/2motif/'
    data = file_management.load_lzma(os.path.join(folder,f"2motif_different_omega.lzma"))

    dlist = data[0]
    xlist = data[1]
    jlist = data[2]

    nd = 100 
    no = 21
    delta_array = np.linspace(0,2*np.pi,nd)
    nfixed_points = [ [] for ii in range(no)]

    domega_array = np.linspace(0.5,1.5,no)-1.0
    stability = np.zeros((no,nd))
    theta12 = np.ones((no,nd))*np.nan

    for ii in range(no):
        x = np.concatenate(xlist[ii])
        det = np.concatenate(jlist[ii])
        delta = np.concatenate(dlist[ii])

        # # Compute the stability of the fixed points
        # fixed_points = np.vstack((x, delta)).T
        # matrices = det
        mask = det < 0.0 # compute_stability(fixed_points, matrices)
        # mask = compute_stability(fixed_points)

        xmasked = x[mask]
        deltamasked = delta[mask]
        detmasked = det[mask]
        duniques = np.unique(deltamasked)
        index = np.nonzero(np.in1d(delta_array, duniques))[0]

        stability[ii,index] = np.array([ len(xmasked[deltamasked==d]) for d in duniques])
        # stability[stability>=2] = 2

        for id, d in zip(index,duniques):
            if len(xmasked[deltamasked==d]) == 1:
                theta12[ii,id] = xmasked[deltamasked==d]
                #print(id,xmasked[deltamasked==d])
            else:
                #print(id,xmasked[deltamasked==d])
                theta12[ii,id] = np.nan
    
    # smoothing theta12
    # give a list of the position of nans

    ii,jj = np.where(np.isnan(theta12[5:10,:22]))
    for i,j in zip(ii,jj):
        i = i+5
        theta12[i,j] = np.nanmean(theta12[i-1:i+2,j-1:j+2])

    ii,jj = np.where(np.isnan(theta12[5:10,78:]))
    for i,j in zip(ii,jj):
        i = i+5
        j = j+78
        theta12[i,j] = np.nanmean(theta12[i-1:i+2,j-1:j+2])

    ii, jj = np.where(np.isnan(theta12[10:16,27:72]))
    for i,j in zip(ii,jj):
        i = i+10
        j = j+27
        values = theta12[i-1:i+2,j-1:j+2]
        values[values<0] += 2*np.pi
        values_mean = np.mod(np.nanmean(values),2*np.pi)
        if values_mean > np.pi:
            values_mean -= 2*np.pi
        theta12[i,j] = values_mean

    fig, ax = plt.subplot_mosaic([['0','bar']],gridspec_kw={'width_ratios':[1,0.05]})   
    pcolor = ax["0"].pcolor(delta_array, domega_array, theta12, cmap="twilight",vmin=-np.pi,vmax=np.pi)
    ax["0"].set_yticks([-0.5,-0.25,0.0,0.25,0.5])
    ax["0"].set_xticks(xticks,xtickslabels)
    cb = Colorbar(ax=ax["bar"], mappable=pcolor, orientation='vertical', ticklocation='right',ticks=yticks_colorbar)
    cb.ax.set_yticklabels(ytickslabels_colorbar)
    ax["0"].set_xlabel(r"$\delta$ [rad]")
    ax["0"].set_ylabel(r"$\Delta\Omega/\Omega_0$")
    ax["0"].set_title(r"$\theta_{12}$")

    if save_plot:
        plt.savefig(os.path.join(save_folder,title_plot), bbox_inches='tight')
    else:
        plt.show()



