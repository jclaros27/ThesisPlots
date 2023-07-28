
def make_plot(l1=0,save_plot=False,save_folder=None,title_plot=None):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import sys
    import scipy as sp
    import functions
    import copy
    from matplotlib.colorbar import Colorbar 
    import plot_parameters as pc
    import matplotlib.transforms as mtransforms

    sys.path.append('/home/jaime/Desktop/hippocampus/files')
    import file_management

    folder = '/home/jaime/Desktop/synchronization/cmotif'

    pc.fontsize_ticklabels = 10.0
    plt.rcParams['font.size'] = pc.fontsize_labels
    plt.rcParams['axes.labelsize'] = pc.fontsize_labels
    plt.rcParams['axes.titlesize'] = pc.fontsize_titles
    plt.rcParams['legend.fontsize'] = pc.fontsize_labels
    plt.rcParams['xtick.labelsize'] = pc.fontsize_ticklabels
    plt.rcParams['ytick.labelsize'] = pc.fontsize_ticklabels

    if pc.use_latex:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx}'

    colorbar_ticks = [-np.pi,-np.pi/2,0.0,np.pi/2,np.pi]
    colorbar_ticklabels = [r"$-\pi$",r"$-\pi/2$","$0$",r"$\pi/2$",r"$\pi$"]
    xticks = [0,np.pi/2,np.pi,3*np.pi/2,2*np.pi]
    xticklabels = [r"$0$",r"$\pi/2$",r"$\pi$",r"$3\pi/2$",r"$2\pi$"]
    yticks = [-0.5,-0.25,0.0,0.25,0.5]
    yticklabels = [r"$-0.50$",r"$-0.25$","$0.00$",r"$0.25$",r"$0.50$"]

    if l1 == 0:
        ylabel = r"$\Delta\Omega_1/\Omega_0$"
    else:
        ylabel = r"$\Delta\Omega_2/\Omega_0$"

    # mosaic plot
    mosaic_list = [['10','11','bar'],['20','21','bar'],['30','31','bar']]  
    fig, axs = plt.subplot_mosaic(mosaic_list,
                                  gridspec_kw={'height_ratios': [1,1,1],
                                               'width_ratios': [1,1,0.05]},
                                  layout='constrained')

    for l2,kp in zip([1,2,3],[0.25,0.5,0.75]):
        title = f'cmotif_different_omega_different_delta_{l1}_{l2}.lzma'
        data = file_management.load_lzma(os.path.join(folder,title))
        
        dlist = data[0]
        xlist = data[1]
        ylist = data[2]
        jlist = data[3]
        tlist = data[4]
        nd = 101 
        no = 21

        delta_array = np.linspace(0,2*np.pi,nd)
        nfixed_points = [ [] for ii in range(no)]
        domega_array = np.linspace(0.5,1.5,no)-1.0
        stability = np.zeros((no,nd))
        theta12 = np.ones((no,nd))*np.nan
        theta13 = np.ones((no,nd))*np.nan 

        theta12_list = [ [] for i in range(no)]
        theta13_list = [ [] for i in range(no)]
        delta_list   = [ [] for i in range(no)]
        
        k_matrix = np.array([  [0.0, 1.0, kp], [1.0, 0.0, 1.0], [kp, 1.0, 0.0] ])

        for ii in range(no):
            x = np.concatenate(xlist[ii])
            y = np.concatenate(ylist[ii])
            det = np.concatenate(jlist[ii])
            trace = np.concatenate(tlist[ii])
            delta = np.concatenate(dlist[ii])

            delta_matrix = [ [] for jj in range(len(delta))]
            for jj in range(len(delta)):
                delta_matrix[jj] = np.array([ [0.0, delta[jj], delta[jj]], [delta[jj], 0.0, delta[jj]], [delta[jj], delta[jj], 0.0] ])
            if l1 == 0:
                omega_array = [domega_array[ii]+1.0,1.0,1.0]
            if l1 == 1:
                omega_array = [1.0,domega_array[ii]+1.0,1.0]
            
            # Compute the stability of the fixed points
            fixed_points = np.vstack((x, y, delta)).T
            matrices = np.vstack((det, trace)).T
            
            mask = functions.determine_stability(x=x,y=y,delta=delta_matrix,omega=omega_array,k=k_matrix)
            # mask = compute_stability(fixed_points,matrices)
            #print(mask)
            xmasked = x[mask]
            ymasked = y[mask]
            deltamasked = delta[mask]
            detmasked = det[mask]
            tracemasked = trace[mask]

            duniques = np.unique(deltamasked)
            index = np.nonzero(np.in1d(delta_array, duniques))[0]

            stability[ii,index] = np.array([ len(xmasked[deltamasked==d]) for d in duniques])
            for id, d in zip(index,duniques):
                if len(xmasked[deltamasked==d])==1:
                    theta12[ii,id] = xmasked[deltamasked==d]
                    theta13[ii,id] = ymasked[deltamasked==d]
                else:
                    theta12[ii,id] = np.nan
                    theta13[ii,id] = np.nan     

        stability[stability>2] = 2

        # theta12 = functions.correct_matrices(stability,theta12,units='radians')
        # theta13 = functions.correct_matrices(stability,theta13,units='radians')

        stability, theta12, theta13, lines0, lines1 = functions.detect_outliers(stability, theta12, theta13, plot=False)
        
        if l1 == 0:
            if l2 == 1:
                theta12[-1,68] = np.nan
                theta13[-1,68] = np.nan
            if l2 == 2:
                print(theta12[10,30:60])
                theta12[10,44:56] += np.pi
                theta13[10,44:56][theta13[10,44:56]>= np.pi] -= 2*np.pi
            if l2 == 3:
                theta12[0,42] = theta12[0,43]
                theta13[0,42] = theta13[0,43]

        sta = copy.copy(stability).astype(float)
        sta[sta<2] = np.nan
        axs[f"{l2}0"].pcolor(delta_array, domega_array, sta,vmax=2,alpha=0.05,cmap="summer")
        axs[f"{l2}1"].pcolor(delta_array, domega_array, sta,vmax=2,alpha=0.05,cmap="summer")

        subfig1 = axs[f"{l2}0"].pcolormesh(delta_array, domega_array, theta12, cmap="twilight",vmin=-np.pi,vmax=np.pi)
        subfig2 = axs[f"{l2}1"].pcolormesh(delta_array, domega_array, theta13, cmap="twilight",vmin=-np.pi,vmax=np.pi) 
        axs[f"{l2}0"].set_yticks(yticks)
        axs[f"{l2}1"].set_yticks(yticks)
        axs[f"{l2}0"].set_yticklabels([])
        axs[f"{l2}1"].set_yticklabels([])
        axs[f"{l2}0"].set_xticks(xticks)
        axs[f"{l2}1"].set_xticks(xticks)
        axs[f"{l2}0"].set_xticklabels([])
        axs[f"{l2}1"].set_xticklabels([])

    axs["10"].set_yticklabels(yticklabels)
    axs["20"].set_yticklabels(yticklabels)
    axs["30"].set_yticklabels(yticklabels)
    axs["30"].set_xticklabels(xticklabels)
    axs["31"].set_xticklabels(xticklabels)
    
    axs["10"].set_ylabel(ylabel)
    axs["20"].set_ylabel(ylabel)
    axs["30"].set_ylabel(ylabel)
    axs["30"].set_xlabel(r"$\delta$ [rad]")
    axs["31"].set_xlabel(r"$\delta$ [rad]")

    axs["10"].set_title(r"$\theta_{12}$")
    axs["11"].set_title(r"$\theta_{13}$")

    cb = Colorbar(ax = axs["bar"], mappable = subfig1, orientation = 'vertical', ticklocation = 'right',ticks=colorbar_ticks)
    cb.ax.set_yticklabels(colorbar_ticklabels)


    trans = mtransforms.ScaledTranslation(-30/72, -5/72, fig.dpi_scale_trans)
    for label,title in zip(["10","20","30"],[r"\textbf{A}",r"\textbf{B}",r"\textbf{C}"]):
        axs[label].text(-0.1, 1.0, title, transform=axs[label].transAxes + trans, fontsize='medium', va='bottom', fontfamily='serif',fontweight="bold")

    trans = mtransforms.ScaledTranslation(-5/72, -5/72, fig.dpi_scale_trans)
    for label,title in zip(["11","21","31"],[r"\textbf{D}",r"\textbf{E}",r"\textbf{F}"]):
        axs[label].text(-0.1, 1.0, title, transform=axs[label].transAxes + trans,fontweight="bold")

    if save_plot:
        plt.savefig(os.path.join(save_folder,title_plot))
    else:
        plt.show()