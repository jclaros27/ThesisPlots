import os
import single_node_dynamics
import normal_components_distributions
import interregional_communication
import theoretical_distributions_effect
import histograms
import stimwfit
import simulation_ouput
import working_point
# import plv_dynamics
# import empirical_matrices


current_folder = os.getcwd() 
save_folder = os.path.join(current_folder,"plots")

# methods plots

title = "single-node-dynamics.png"
# single_node_dynamics.make_plot(save_plot=True, save_folder=save_folder,title_plot=title)

title = "histograms_electric_field.png"
# normal_components_distributions.make_plot(save_plot=True,save_folder=save_folder,title_plot=title)

# results plots
title = "effect_of_neighboor_stimulation.png"
# interregional_communication.make_plot(save_plot=True,save_folder=save_folder,title_plot=title)

title = "theoretical_distributions_effect.png"
# theoretical_distributions_effect.make_plot(save_plot=True,save_folder=save_folder,title_plot=title)

title = "histograms.png"
# histograms.make_plot(save_plot=True,save_folder=save_folder,title_plot=title)

title = "alpha_rise_cluster.png"
# stimwfit.make_plot(save_plot=True,save_folder=save_folder,title_plot=title)

title = "lfps_stimulation.png"
# simulation_ouput.make_plot(save_plot=True,save_folder=save_folder,title_plot=title)

title = "working_point.png"
# working_point.make_plot(save_plot=True,save_folder=save_folder,title_plot=title)

title = "plv_dynamics.png"ue,save_folder=save_folder,title_plot=title)
# plv_dynamics.make_plot(save_plot=True,save_folder=save_folder,title_plot=title)

title = "empirical_matrices.png"
# empirical_matrices.make_plot(save_plot=Tr