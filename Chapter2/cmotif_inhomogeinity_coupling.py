
def make_plot(save_plot=False, save_folder=None,title_plot=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import sys
    import scipy as sp
    import functions
    import copy
    from matplotlib.colorbar import Colorbar 
    import matplotlib.transforms as mtransforms

    import plot_parameters as pc 
    sys.path.append('/home/jaime/Desktop/hippocampus/files')
    import file_management

    folder = '/home/jaime/Desktop/synchronization/cmotif'

    pc.fontsize_ticklabels = 12.0
    pc.fontsize_labels = 18.0
    plt.rcParams['font.size'] = pc.fontsize_labels
    plt.rcParams['axes.labelsize'] = pc.fontsize_labels
    plt.rcParams['axes.titlesize'] = pc.fontsize_titles
    plt.rcParams['legend.fontsize'] = pc.fontsize_legend
    plt.rcParams['xtick.labelsize'] = pc.fontsize_ticklabels
    plt.rcParams['ytick.labelsize'] = pc.fontsize_ticklabels

    if pc.use_latex:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{siunitx}'

    colorbar_ticks = [-np.pi,-np.pi/2,0.0,np.pi/2,np.pi]
    colorbar_ticklabels = [r"$-\pi$",r"$-\pi/2$","$0$",r"$\pi/2$",r"$\pi$"]
    xticks = [0,np.pi/2,np.pi,3*np.pi/2,2*np.pi]
    xticklabels = [r"$0$",r"$\pi/2$",r"$\pi$",r"$3\pi/2$",r"$2\pi$"]
    yticks = [0,0.5,1.0,1.5,2.0,2.5]
    yticklabels = ['0.0','0.5','1.0','1.5','2.0','2.5']

    mosaic_list = [['0','0','0','1','1','1','bar'],
                   ['0','0','0','1','1','1','bar'],
                   ['0','0','0','1','1','1','bar'],
                   ['a','a','a','a','a','a','aux'],
                   ['2','2','2','2','2','2', 'aux'], 
                   ['2','2','2','2','2','2', 'aux'],
                   ['2','2','2','2','2','2', 'aux'],
                   ['b','b','b','b','b','b', 'aux'],
                   ['3','3','4','4','5','5', 'aux'],
                   ['3','3','4','4','5','5',    'aux']]                 

    fig, axs = plt.subplot_mosaic(mosaic_list,
                                gridspec_kw={'height_ratios': [1,1,1,1.2,1,1,1,1.5,1,1],'width_ratios': [1,1,1,1,1,1,0.25],'wspace':2.0},
                                layout='constrained',
                                figsize=(6,7))  

    # parte de los colormaps
    axs["a"].axis("off")
    axs["b"].axis("off")
    data = file_management.load_lzma(os.path.join(folder,f"cmotif_same_omega_same_delta.lzma"))

    dlist = data[0]
    xlist = data[1]
    ylist = data[2]
    jlist = data[3]
    tlist = data[4]
    nx = 100
    ny = 51

    yarray = np.linspace(0.0,2.5,ny)
    xarray = np.linspace(0,2*np.pi,nx)

    stability = np.zeros((ny,nx))
    theta12 = np.ones((ny,nx))*np.nan
    theta13 = np.ones((ny,nx))*np.nan 

    theta12_list = [ [] for i in range(ny)]
    theta13_list = [ [] for i in range(ny)]
    delta_list   = [ [] for i in range(ny)]

    for ii in range(ny):
        x = np.concatenate(xlist[ii])
        y = np.concatenate(ylist[ii])
        det = np.concatenate(jlist[ii])
        trace = np.concatenate(tlist[ii])
        delta = np.concatenate(dlist[ii])

        # Compute the stability of the fixed points
        fixed_points = np.vstack((x, y, delta)).T
        matrices = np.vstack((det, trace)).T
        mask = functions.compute_stability(fixed_points, matrices)
        # mask = compute_stability(fixed_points)

        xmasked = x[mask]
        ymasked = y[mask]
        deltamasked = delta[mask]
        detmasked = det[mask]
        tracemasked = trace[mask]

        duniques = np.unique(deltamasked)
        index = np.nonzero(np.in1d(xarray, duniques))[0]

        stability[ii,index] = np.array([ len(xmasked[deltamasked==d]) for d in duniques])
        stability[stability>=2] = 2

        theta12_list[ii] = xmasked
        theta13_list[ii] = ymasked 
        delta_list[ii]   = deltamasked
        for id, d in zip(index,duniques):

            if len(xmasked[deltamasked==d])==1:
                theta12[ii,id] = xmasked[deltamasked==d]
                theta13[ii,id] = ymasked[deltamasked==d]

    # w = np.where(deltamasked == delta_array[50]) 
    # plt.figure()
    # plt.plot(deltamasked[w],theta12_list[w])
    stability[-1,25:75][(stability[-1,25:75]==0)[0]] = 2
    stability, theta12, theta13, lines0, lines1 = functions.detect_outliers(stability, theta12, theta13, plot=False)
    # plot stability
    sta = copy.copy(stability)
    sta=sta.astype(float)
    sta[sta<2] = np.nan
    # plot theta12 and theta13
    axs["0"].pcolor(xarray, yarray, sta,vmax=2.0,alpha=0.05,cmap="summer")
    axs["1"].pcolor(xarray, yarray, sta,vmax=2.0,alpha=0.05,cmap="summer")
    th12 = axs["0"].pcolormesh(xarray, yarray, theta12, cmap="twilight",vmin=-np.pi,vmax=np.pi)
    th13 = axs["1"].pcolormesh(xarray, yarray, theta13, cmap="twilight",vmin=-np.pi,vmax=np.pi)
    cb = Colorbar(axs["bar"], th12, ticks=colorbar_ticks,orientation="vertical",ticklocation="right")
    cb.ax.set_yticklabels(colorbar_ticklabels)

    axs["0"].set_xlabel(r"$\delta$ [rad]")#,fontsize=15)
    axs["0"].set_ylabel(r"$k'/k$")#,fontsize=15)
    axs["0"].set_title(r"$\theta_{12}$")#,fontsize=15)
    axs["1"].set_title(r"$\theta_{13}$")#,fontsize=15)
    axs["1"].set_xlabel(r"$\delta$ [rad]")#,fontsize=15)
    axs["0"].set_xticks(xticks)
    axs["0"].set_xticklabels(xticklabels)
    axs["1"].set_xticks(xticks)
    axs["1"].set_xticklabels(xticklabels)
    axs["1"].set_yticks(yticks)
    axs["1"].set_yticklabels([])
    axs["0"].set_yticks(yticks)
    axs["0"].set_yticklabels(yticklabels)

    axs["0"].axvline(np.pi,linestyle="--",color="tab:blue")
    axs["1"].axvline(np.pi,linestyle="--",color="tab:orange")

    axs["aux"].axis("off")
    ############################################################
    # parte no colormaps 

    xticks = [0,0.5,1,1.5,2,2.5]
    xticklabels = ["0.0","0.5","1.0","1.5","2.0","2.5"]
    yticks = [-np.pi,-np.pi/2,0,np.pi/2,np.pi]
    yticklabels = [r"$-\pi$",r"$-\pi/2$",r"$0$",r"$\pi/2$",r"$\pi$"]

    eps = 1.0
    cluster_size = 100
    eps_theta = 1e-2
    x = np.linspace(-np.pi-eps_theta,np.pi+eps_theta,1001,endpoint=False)
    y = np.linspace(-np.pi-eps_theta,np.pi+eps_theta,1001,endpoint=False)
    X,Y = np.meshgrid(x,y)

    omega = [1.0,1.0,1.0]
    k0 = 1.0
    delt  = np.pi
    deltp = np.pi
    delta = np.array([ [0.0,   delt, deltp],
                       [delt,  0.0,  delt],
                       [deltp, delt, 0.0] ])

    i = 50
    auxx =  [ yarray[ii]*np.ones(len(theta12_list[ii][delta_list[ii]==xarray[i]])) for ii in range(ny)]
    auxx2 = [ yarray[ii]*np.ones(len(theta13_list[ii][delta_list[ii]==xarray[i]])) for ii in range(ny)]
    auxy  = [ theta12_list[ii][delta_list[ii]==xarray[i]] for ii in range(ny)]
    auxy2 = [ theta13_list[ii][delta_list[ii]==xarray[i]] for ii in range(ny)]

    auxx.append( [0.5])
    auxx2.append( [0.5])
    auxy.append([-np.pi])
    auxy2.append( [0.0])
    axs["2"].plot(np.concatenate(auxx), np.concatenate(auxy),'o',label="12")
    axs["2"].plot(np.concatenate(auxx2), np.concatenate(auxy2),'o',label="13")
    axs["2"].set_yticks(yticks)
    axs["2"].set_yticklabels(yticklabels)
    axs["2"].set_xticks(xticks)
    axs["2"].set_xticklabels(xticklabels)
    #axs["0"].axvline(0.25,linestyle="--",color="black")
    #axs["0"].axvline(0.75,linestyle="--",color="black")
    axs["2"].set_xlabel(r"$k'/k$")#,fontsize=15)
    axs["2"].set_ylabel(r"$\theta^{*}_{ij}$")#,fontsize=15)
    axs["2"].legend(loc="best")
    axs["2"].grid(True)
    delt_array = np.linspace(0,2*np.pi,nx)
    # kp_array = np.linspace(0,1.5,31)


    xticks = [-np.pi,0.0,np.pi]
    xticklabels = [r"$-\pi$",r"$0$",r"$\pi$"]
    yticks = xticks
    yticklabels = xticklabels

    for kp,label in zip([0.25,0.5,1.25],["3","4","5"]):

        k = np.array([  [0.0, k0, kp],
                        [k0, 0.0, k0],
                        [kp, k0, 0.0] ])

        dx,dy = functions.phase_locked_states(X,Y,omega,delta,k)
        det, trace = functions.det_and_trace_from_jacobian_matrix(X,Y,omega, delta, k)
        results = functions.get_intersection_points(X,Y, dx,dy,det, trace, delt, eps=1.0, cluster_size=100, create_plot=False)
        
        xlist = results[0]
        ylist = results[1]
        jlist = results[2]
        tlist = results[3]
        dlist = results[4]

        #axs[label].contourf(X,Y,det, levels=[0,np.max(det)])
        #axs[label].contour(X,Y,det,  levels=[0],colors='purple')
        axs[label].streamplot(X,Y, dx, dy, linewidth=0.5, density=.8,color="gray")
        axs[label].contour(X,Y,dy, levels=[0],colors='tab:orange')
        axs[label].contour(X,Y,dx, levels=[0],colors='tab:blue')

        axs[label].scatter( xlist, ylist, s=40, color="black")
        w = (jlist>0) & (tlist<0)
        axs[label].scatter( xlist[w], ylist[w], s=200, color="red",facecolors="none")

        axs[label].set_yticks(yticks)
        axs[label].set_yticklabels([])
        axs[label].set_xticks(xticks)
        axs[label].set_xticklabels(xticklabels)

    axs["3"].set_yticklabels(yticklabels)
    axs["3"].set_xlabel(r"$\theta_{12}$ [rad]")
    axs["4"].set_xlabel(r"$\theta_{12}$ [rad]")
    axs["5"].set_xlabel(r"$\theta_{12}$ [rad]")
    axs["3"].set_ylabel(r"$\theta_{13}$ [rad]")
    axs["3"].set_title(r"$k'/k = 0.25$")
    axs["4"].set_title(r"$k'/k = 0.50$")
    axs["5"].set_title(r"$k'/k = 1.25$")


    trans = mtransforms.ScaledTranslation(-40/72, -5/72, fig.dpi_scale_trans)
    axs["0"].text(-0.1, 1.0, r"\textbf{A}", transform=axs["0"].transAxes + trans,fontweight="bold")  

    trans = mtransforms.ScaledTranslation(-25/72, -5/72, fig.dpi_scale_trans)
    axs["2"].text(-0.1, 1.0, r"\textbf{C}", transform=axs["2"].transAxes + trans,fontweight="bold")  
    
    trans = mtransforms.ScaledTranslation(-45/72, -3/72, fig.dpi_scale_trans)
    axs["3"].text(-0.1, 1.0, r"\textbf{D}", transform=axs["3"].transAxes + trans,fontweight="bold")  
    trans = mtransforms.ScaledTranslation(-15/72, -3/72, fig.dpi_scale_trans)
    axs["4"].text(-0.1, 1.0, r"\textbf{E}", transform=axs["4"].transAxes + trans,fontweight="bold")  
    axs["5"].text(-0.1, 1.0, r"\textbf{F}", transform=axs["5"].transAxes + trans,fontweight="bold")  

    trans = mtransforms.ScaledTranslation(-15/72, -5/72, fig.dpi_scale_trans)
    axs["1"].text(-0.1, 1.0, r"\textbf{B}", transform=axs["1"].transAxes + trans,fontweight="bold")    

    fig.align_ylabels()  

    if save_plot:
        plt.savefig(os.path.join(save_folder,title_plot),bbox_inches='tight')
    else:
        plt.show()