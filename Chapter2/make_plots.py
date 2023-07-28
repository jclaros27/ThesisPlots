import two_motif_different_omega
import vmotif_inhomogeneity_frequency
import cmotif_inhomogeinity_frequency
import cmotif_inhomogeinity_coupling
import cmotif_homogeneity_different_delta
import os 

current_folder = os.getcwd() 
save_folder = os.path.join(current_folder, "plots")
# title_plot = "2motif_different_omega.png"
# two_motif_different_omega.make_plot(save_plot=True, save_folder=save_folder,title_plot=title_plot)

# for l1 in [0,1]:
#     title_plot = f"vmotif_different_omega_{l1}.png"
#     vmotif_inhomogeneity_frequency.make_plot(l1=l1,save_plot=True, save_folder=save_folder,title_plot=title_plot)

# for l1 in [0,1]:
#     title_plot = f"cmotif_different_omega_{l1}.png"
#     cmotif_inhomogeinity_frequency.make_plot(l1=l1,save_plot=True,save_folder=save_folder,title_plot=title_plot)

title = f"cmotif_different_coupling.png"
cmotif_inhomogeinity_coupling.make_plot(save_plot=True,save_folder=save_folder,title_plot=title)

# title = f"cmotif_different_delta.png"
# cmotif_homogeneity_different_delta.make_plot(save_plot=True,save_folder=save_folder,title_plot=title)