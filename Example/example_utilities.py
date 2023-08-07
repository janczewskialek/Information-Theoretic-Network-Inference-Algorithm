import numpy as np
import os

import networkx as nx 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import seaborn as sns


def time_to_ms(hour, minute):
    return hour*60*60*1000 + minute*60*1000



def simulate3dealers(time_set, couplingBtoA, couplingCtoB ):
    nb_of_timepoints = np.shape(time_set)[0]

    initial_price = 1.0

    A_dealer = np.zeros(nb_of_timepoints)
    B_dealer = np.zeros(nb_of_timepoints)
    C_dealer = np.zeros(nb_of_timepoints)

    A_dealer[0] = initial_price
    B_dealer[0] = initial_price
    C_dealer[0] = initial_price

    alpha = 1
    beta  = 1 - couplingBtoA
    gamma = 1 - couplingCtoB

    rand_mean = 0.001
    rand_scale = 0.05

    for t in range(1, nb_of_timepoints):
            rand_A = np.random.normal(loc=rand_mean, scale=rand_scale)
            rand_B = np.random.normal(loc=rand_mean, scale=rand_scale)
            rand_C = np.random.normal(loc=rand_mean, scale=rand_scale)

            A_dealer[t] = alpha * A_dealer[t-1] + rand_A
            B_dealer[t] = beta  * B_dealer[t-1] + couplingBtoA * A_dealer[t-1] + rand_B
            C_dealer[t] = gamma * C_dealer[t-1] + couplingCtoB * B_dealer[t-1] + rand_C

    return A_dealer, B_dealer, C_dealer


# Disclaimer: the plotting function manipulate (if necessary) only for the purpose of the example that it is used for
def plot_map(beta_inefficiency, entropy, information_share, markets, path, SAVE):
    
    ConStyle = 'arc3, rad=0.12'
    ArrStyle = '-|>'
    nodes_dealers_size = 800
    
    fontkwargs = {'font_size':16}
    fontkwargs1 = {'fontsize':16}

    BI = beta_inefficiency
    EC = entropy
    SL = information_share

    # Manipulation to ensure that the simulated example does not break the figure
    EC[np.isnan(EC)] = 0
    BI[BI>1] = 1
    
    # Basic setting for the figure
    infoshare_alpha = 1
    infoshare_vmax = np.max(SL) 
    bilateral_vmax = np.max(EC)
    bilateral_vmin = 0
    infoshare_vmin = 0

    fig, ax = plt.subplots(figsize=(15,15))

    # Shift for nodes labels
    x_shift=0.095
    y_shift=0.095


    bilateral_colorbar_ticks = np.arange(0, bilateral_vmax+0.01, bilateral_vmax/4)
    infoshare_colorbar_ticks = np.arange(infoshare_vmin, infoshare_vmax+10, infoshare_vmax/4)
    #####################################################################################
    # COLORS

    circe_colors = "#78909C"
    center_color = "#212529"
    nodes_border_color = sns.color_palette("tab10")[-3]
    nodes_infoshare_color = sns.color_palette("blend:#ADB5BD,#FFD43B,#2F9E44", as_cmap=True)
    nodes_infoshare_color = sns.color_palette("blend:#ADB5BD,#378DBD,#1E5288,#0C234B", as_cmap=True)
    
    CMAP =  sns.color_palette("blend:#F7BC50,#FC6B36,#DE3F67,#9D1B3B", as_cmap=True)

    #####################################################################################
    # CIRCLES
    radiuses_range = [1, 0.75, 0.5, 0.25]

    for rr, radius in enumerate(radiuses_range):
        if radius == 1:
            G_circle_r1_nodes = 200
            node_size = 10
            alpha = 1
            edge_width = 2
        
        elif radius == 0.75:
            G_circle_r1_nodes = 150
            node_size = 7.5
            alpha = 0.75
            edge_width = 1

        elif radius == 0.50:
            G_circle_r1_nodes = 100
            node_size = 5
            alpha = 0.6
            edge_width = 1
        else:
            G_circle_r1_nodes = 50
            node_size = 2.5
            alpha = 0.50
            edge_width = 1

        G_circle_r1 = nx.cycle_graph(G_circle_r1_nodes)
        G_circle_r1_nodes_position = np.zeros((G_circle_r1_nodes,2))

        for i in range(G_circle_r1_nodes):
            xpos = np.cos(i*np.pi*2/G_circle_r1_nodes+np.pi)*radius
            ypos = np.sin(i*np.pi*2/G_circle_r1_nodes+np.pi)*radius
            
            G_circle_r1_nodes_position[i, 0] = xpos
            G_circle_r1_nodes_position[i, 1] = ypos

        if radius==1:
            nx.draw(G_circle_r1, G_circle_r1_nodes_position, node_size = 0, alpha=1, node_color="white", edge_color="white", ax=ax)

        nx.draw_networkx_edges(G_circle_r1,G_circle_r1_nodes_position,
            edge_color = circe_colors,
            style = "-.",         
            alpha = alpha,
            width= edge_width,
            ax=ax,)
    #####################################################################################

    G_map = nx.from_numpy_matrix(EC, create_using=nx.DiGraph)
    G_map.remove_edges_from(nx.selfloop_edges(G_map))

    edges,weights = zip(*nx.get_edge_attributes(G_map,'weight').items())
    weights = np.array(weights)

    node_label_dic = {
        v:k[0:3] for v, k in enumerate(markets)}

    nnodes=len(markets)

    pos = np.zeros((nnodes,2))
    pos2 = np.zeros((nnodes,2))
    pos_label = np.zeros((nnodes,2))
    
    
    for i in range(nnodes):
        if (np.isnan(BI[i])):
            radius = 1
        else:
            radius = BI[i]

        xpos = np.cos(i*np.pi*2/nnodes+np.pi)*radius
        ypos = np.sin(i*np.pi*2/nnodes+np.pi)*radius
        xpos2 = np.cos(i*np.pi*2/nnodes+np.pi)*(radius-0.012)
        ypos2 = np.sin(i*np.pi*2/nnodes+np.pi)*(radius-0.012)

        if radius >=1:
            xpos_label = np.cos(i*np.pi*2/nnodes+np.pi)*(radius - x_shift)
            ypos_label = np.sin(i*np.pi*2/nnodes+np.pi)*(radius - y_shift)
        else:
            xpos_label = np.cos(i*np.pi*2/nnodes+np.pi)*(radius + x_shift)
            ypos_label = np.sin(i*np.pi*2/nnodes+np.pi)*(radius + y_shift)
        
        pos[i, 0] = xpos
        pos[i, 1] = ypos
        pos2[i, 0] = xpos2
        pos2[i, 1] = ypos2
        pos_label[i, 0] = xpos_label
        pos_label[i, 1] = ypos_label


    #####################################################################################
    # NETWORK PLOTS

    nx.draw_networkx_nodes(G_map,pos,
        node_color = SL,
        node_size = nodes_dealers_size,
        cmap = nodes_infoshare_color,
        vmin=infoshare_vmin,
        vmax=infoshare_vmax,
        alpha=infoshare_alpha,
        edgecolors= nodes_border_color,
        linewidths = 2,
        ax=ax
    )
    nx.draw_networkx_edges(G_map,pos2,
        arrows=True,
        connectionstyle = ConStyle,
        arrowstyle = ArrStyle,
        edge_color = weights,         
        edge_cmap = CMAP,
        edge_vmin=bilateral_vmin, 
        edge_vmax=bilateral_vmax,
        alpha = 0.75,
        width=weights*100,
        ax=ax,
        )

    nx.draw_networkx_labels(G_map, pos=pos_label, labels = node_label_dic, ax=ax, **fontkwargs)    
    
    #####################################################################################
    # CENTER DOT
    G_center=nx.Graph()
    G_center.add_node("")
    nx.draw_networkx_nodes(G_center, node_shape="h" ,node_size = 30,label=False, node_color = center_color, pos={'':(0,0)}, ax=ax)


    #####################################################################################
    # COLORBAR 1 - Flows
    cbaxes1 = fig.add_axes([0.86, 0.40, 0.02, 0.2])
    sm1 = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin = bilateral_vmin, vmax=bilateral_vmax))
    sm1._A = []

    cb1 = plt.colorbar(sm1, cax=cbaxes1, shrink=1, aspect = 10, ticks=bilateral_colorbar_ticks)
    cb1.ax.tick_params(labelsize=14)

    # COLORBAR 2 - Information Share
    cbaxes2 = fig.add_axes([0.13, 0.40, 0.02, 0.2])
    sm2 = plt.cm.ScalarMappable(cmap=nodes_infoshare_color, norm=plt.Normalize(vmin = infoshare_vmin, vmax=infoshare_vmax))
    sm2._A = []

    cb2 = plt.colorbar(sm2, cax=cbaxes2, shrink=1, aspect = 10, ticks=infoshare_colorbar_ticks)
    cb2.ax.set_title('Information \n Share', pad=20, **fontkwargs1)
    cb2.ax.tick_params(labelsize=14)
    #####################################################################################
    # SAVE FIGURE

    if SAVE ==True:
        fig.savefig(path, bbox_inches="tight")
    plt.show()



def create_directories_for_output(save_base_path, save_fullpath):
    try:
        os.mkdir(save_base_path)
    except OSError:
        print("Main directory %s already exists!" % save_base_path)
    else:
        print ("Main directory %s created!" % save_base_path)

    try:
        os.mkdir(save_fullpath)
        os.mkdir(save_fullpath + "/arrays")
        os.mkdir(save_fullpath + "/arrays/raw")
    except OSError:
        print ("Creation of the directory %s failed" % save_fullpath)
    else:
        print ("Successfully created the directory %s " % save_fullpath)


    # Creating main directory for the data
    try:
        os.mkdir(f"{save_fullpath}/figures")
    except OSError:
        print ("Creation of the directory %s failed" % save_fullpath)
    else:
        print ("Main directory created!")