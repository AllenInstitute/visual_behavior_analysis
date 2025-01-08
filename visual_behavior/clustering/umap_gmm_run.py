"""
Vars needed here are set in umap_set_vars.py

Here we run umap and pca on the data matrix created in umap_set_vars.py.

After this script run umap_plots.py to make plots.


Created on Mon Jun 29 11:31:25 2020
@author: farzaneh

"""

from multiscope_fn.def_funs import *
from umap_funs import *

################################################################################################    
################################################################################################        
#%% Run PCA on all_sess_ns_fof_this_cre
################################################################################################    
################################################################################################    

from sklearn.decomposition import PCA
varexpmax = .99 # 1 # .9

pc_all_cre = []
pca_variance_all_cre = []

for icre in range(len(cre_lines)): # icre=0
    cre = cre_lines[icre]    
    all_sess_ns_fof_thisCre = all_sess_ns_fof_all_cre[icre] # neurons_allExp_thisCre x 24(frames)
    print(f'Running PCA on {cre}, matrix size: {np.shape(all_sess_ns_fof_thisCre)}')

    x_train_pc, pca = doPCA(all_sess_ns_fof_thisCre, varexpmax=varexpmax, doplot=1)
    pca_variance = pca.explained_variance_ratio_
#     x_train_pc.shape

    pc_all_cre.append(x_train_pc)
    pca_variance_all_cre.append(pca_variance)
    
    
################################################################################################    
################################################################################################    
#%% Run umap on all_sess_ns_fof_thisCre
################################################################################################
################################################################################################

import umap    
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

ncomp = 2 # 3 # number of umap components

embedding_all_cre = []

for icre in range(len(cre_lines)): # icre = 2    
    cre = cre_lines[icre]    
    all_sess_ns_fof_thisCre = all_sess_ns_fof_all_cre[icre] # neurons_allExp_thisCre x 24(frames)
    print(f'Running UMAP on {cre}')
    
    sp = 2
    neigh = 7
    embedding = umap.UMAP(spread= sp, n_neighbors = neigh, n_components = ncomp).fit_transform(all_sess_ns_fof_thisCre)
    print(f'embedding size: {embedding.shape}')
    embedding_all_cre.append(embedding)
    
# embedding_all_cre_3d = copy.deepcopy(embedding_all_cre)

    
    
##########################################################################################
#%% After this script, run umap_plot.py to make plots for umap/pca analysis done above.    
##########################################################################################




########################################################################
#%% Try umap on a range of parameteres, and do GMM on each embedding, if desired
########################################################################

from sklearn import mixture
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

doGMM = 0 # if 1, run GMM (with different parameters) on each embedding found below

ncomp = 50 #2 # number of umap components
min_dist_vals = [.1, .3, .7]
neigh_vals = np.concatenate(([10, 50], np.arange(200, int(all_sess_ns_fof_thisCre.shape[0]/10), 500))) # neigh_vals = list(range(2,20))
print(neigh_vals)
# sp_vals = [1, 3, 7] # np.arange(1,10)

# embedding_all_cre = []
for icre in [0]: #range(len(cre_lines)): # icre = 2
    
    cre = cre_lines[icre]    
    all_sess_ns_fof_thisCre = all_sess_ns_fof_all_cre[icre] # neurons_allExp_thisCre x 24(frames)
    print(f'Running UMAP on {cre}')
#     sp = 2 #     neigh = 7
    

    # creating dataframe of BIC score output
    best_clust = []
    best_type = []
    umap_neigh = []
    spreads = []
    umap_mindist = []
    bic_list = []

    # Iterating through UMAP configurations
    
    embedding_neigh_sp = [] # (len_minDistVals x len_neigh) x num_samps x num_comps
    
    for neigh in neigh_vals:
        for mindist in min_dist_vals: # for sp in sp_vals:
            
#             print(f'neighbors: {neigh}, spread: {sp}')
            print(f'neighbors: {neigh}, min_distance: {mindist}')
  
            embedding = umap.UMAP(min_dist = mindist, n_neighbors = neigh, n_components = ncomp).fit_transform(all_sess_ns_fof_thisCre)
#             embedding = umap.UMAP(spread = sp, n_neighbors = neigh, n_components = ncomp).fit_transform(all_sess_ns_fof_thisCre)
#             print(f'embedding size: {embedding.shape}')
#             embedding_all_cre.append(embedding)
            
            embedding_neigh_sp.append(embedding)

        
        
            #%% Iterating through GMM configurations, 5 fold validation
            
            if doGMM:
                X = embedding # cluster the umap embedding

                kf = KFold(5)
                n_components_range = np.arange(3,11) #range(2, 20)
                cv_types = ['full'] #['spherical', 'tied', 'diag', 'full']

                bic_array = []
                lowest_bic = np.infty
                bic = []

                for cv_type in cv_types:
                    for n_components in n_components_range:
                        print(f'cv_type: {cv_type}, n_components: {n_components}')

                        bic_temp = []
                        for train_index, test_index in kf.split(X):

                            X_train = X[train_index]
                            X_test = X[test_index]
                            gmm = mixture.GaussianMixture(n_components=n_components,
                                                          covariance_type=cv_type)
                            gmm.fit(X_train)
                            bic_temp.append(gmm.bic(X_test))

                        bic.append(np.average(bic_temp))
                        bic_array.append(bic_temp)

                        if bic[-1] < lowest_bic:
                            lowest_bic = bic[-1]
                            best_gmm = gmm
                            best_cv = cv_type

    #             color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])
                clf = best_gmm
                bars = []
                bic = np.array(bic)
                std_bars = np.std(bic_array, axis = 1)

                best_clust.append(best_gmm.n_components)
                best_type.append(best_gmm.covariance_type)
                umap_neigh.append(neigh)
    #             spreads.append(sp)
                umap_mindist.append(mindist)
                bic_list.append(np.min(bic))

    

emb = np.reshape(embedding_neigh_sp, (len(min_dist_vals), len(neigh_vals), np.shape(embedding_neigh_sp)[1], np.shape(embedding_neigh_sp)[2]), order='F')
emb.shape

if doGMM:
    bic_listr = np.reshape(bic_list, (len(min_dist_vals), len(neigh_vals)), order='F')
    bic_listr.shape
    bic_listr

    #%% Set a dataframe related to GMM configs
    # spreads = umap_mindist    
    df_grid = pd.DataFrame([umap_neigh, spreads, best_clust, bic_list, best_type]).T    
    df_grid.columns = ['umap_neighbor_config', 'umap_spread_config', 'n_best_cluster', 'bic', 'best_gmm_type']
    # df_grid = pd.DataFrame([best_clust, best_type, umap_neigh, spreads]).T
    # df_grid.columns = ['n_best_cluster', 'best_gmm_type', 'umap_neighbor_config', 'umap_spread_config']



#%% Save all the embeddings and GMM output

now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
dest_ct = f'/allen/programs/braintv/workgroups/nc-ophys/Farzaneh/'
np_corr_fold = os.path.join(dest_ct, f'umap')
np_corr_file = os.path.join(np_corr_fold, f'umap_sp_neigh_{now}.h5')

if not os.path.exists(np_corr_fold):
    os.makedirs(np_corr_fold)

    
f = open(np_corr_file, 'wb')        
pickle.dump(emb, f)    
if doGMM:
    pickle.dump(df_grid, f)
f.close()

# f = np_corr_file
# pkl = open(f, 'rb')
# this_sess_emb = pickle.load(pkl)
# this_sess = pickle.load(pkl)
# this_sess_bic = pickle.load(pkl)

    
    

########################################################################    
#%% Umap ran on different values of n_neighbors and spread; make scatter plots for each parameter
########################################################################

#%% Set vars for making the scatter plot (umap ran on all cre lines concatenated)

color_metric = [[cres]] # if umap was run on a matrix that included all cre lines                 
color_labs = ['cre']
dosavefig = 1


amp = color_metric[0][0] # cres    
clab = color_labs[0]

                 
lab_analysis='UMAP'
cut_axes = 0

same_norm_fo = 0
fign_subp = 'cre_'

now = (datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")
dir_now = 'umap_cluster'
fmt = '.pdf'



#%% Plot all embeddings ran with different UMAP parameters

# color by cre

fig = plt.figure(figsize=(21,7)) #(figsize=(10,3)) (14,5)
# plt.suptitle(f'{cre[:3]}', y=.995, fontsize=22) 

for isp in range(np.shape(embedding_neigh_sp)[0]): # isp=0 
#     fig = plt.figure(figsize=(21,7)) #(figsize=(10,3)) (14,5)
    xy = embedding_neigh_sp[isp]
    
    x = xy[:, 0]
    y = xy[:, 1]
    z = xy[:, 2]

#     ax = fig.add_subplot(len(min_dist_vals), len(neigh_vals), isp+1)
    ax = fig.add_subplot(len(min_dist_vals), len(neigh_vals), isp+1, projection='3d')    
#     ax = fig.add_subplot(1,1,1)
        
    fig_ax = [fig, ax]
    
    scatter_umap(x, y, amp, clab, fig_ax, lab_analysis, cut_axes, same_norm_fo, 0, fign_subp, pc_exp_var=np.nan, whatSess = '_AallBall', fgn='', dim=[3,z])

    
plt.subplots_adjust(wspace=0.4, hspace=0.4)

if dosavefig:
    cre_short = 'allCre_allUMAPconfig'
    fgn_umap = f'{lab_analysis}_scatter_{fign_subp}allNeurSess'
#         if useSDK:
#             fgn_umap = f'{fgn_umap}_sdk_'
    nam = '%s_%s%s_%s_%s' %(cre_short, fgn_umap, '_AallBall', '', now)
    fign = os.path.join(dir0, dir_now, nam+fmt)     

    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    


    

#%% plot umap with a specific configuration

spread = 0 #7
neigh = 1
embedding_all_cre = emb[spread,neigh]
# embedding_all_cre = [emb[-1,-1]]

z = embedding_all_cre[:,2]
plot_scatter_fo([cre_lines[0]], [embedding_all_cre], color_metric, color_labs, lab_analysis, cut_axes, same_norm_fo, 0, fign_subp, pc_exp_var=np.nan, whatSess='_AallBall', fgn='', dim=[3,z])



########################################################################    
#%% Make a number of plots to visualize GMM output
########################################################################

# df_grid = pd.DataFrame([best_clust, best_type, umap_neigh, spreads]).T
# df_grid.columns = ['n_best_cluster', 'best_gmm_type', 'umap_neighbor_config', 'umap_spread_config']
df_grid

sns.countplot(df_grid['best_gmm_type'])
plt.title('Covariance Types per Best GMM:\n All Projects', fontsize = 20)
sns.despine(left=True, bottom=True, right=True)
plt.show()


sns.countplot(df_grid['n_best_cluster'])
plt.title('# Cluster Fit per Best GMM:\n All Projects', fontsize = 20)
sns.despine(left=True, bottom=True, right=True)
plt.show()


df_grid2 = df_grid[df_grid['best_gmm_type']=='full']

sns.countplot(data = df_grid2, x = 'n_best_cluster')
plt.title('# Cluster Fit per Best GMM (Full Covariance):\n All Projects', fontsize = 20)
sns.despine(left=True, bottom=True, right=True)
plt.show()



##### kde plots
sns.kdeplot(df_grid2['umap_neighbor_config'], df_grid2['umap_spread_config'], cmap = "Blues", shade = True)
plt.title('Stability of spread/n_neighbor configs:\n All Projects: Full Covariance', fontsize = 20)
plt.show()


df_grid.n_best_cluster.unique()


sns.kdeplot(df_grid['umap_neighbor_config'], df_grid['n_best_cluster'], cmap = "Blues", shade = True)
plt.title('Stability of # Clusters by Neighbor Config:\nAll Projects', fontsize = 20)
plt.show()

sns.kdeplot(df_grid['umap_spread_config'], df_grid['n_best_cluster'], cmap = "Blues", shade = True)
plt.title('Heatmap of Best GMMs using Correlation Covariance')
plt.title('Stability of # Clusters by Spread Config:\nAll Projects', fontsize = 20)
plt.show()






##########################################################################################
#%% Run UMAP on your choice of configuration (based on the GMM outputs)
##########################################################################################

n_neighbors = 50 #10 #50
min_dist = .1
n_epochs = 500 #200 #500

embedding = umap.UMAP(min_dist = min_dist, n_neighbors = n_neighbors, n_components = 50, n_epochs = n_epochs).fit_transform(all_sess_ns_fof_all_cre[icre])

X = embedding
print(X.shape)

# make umap scatter plot
z = X[:,2]
plot_scatter_fo([cre_lines[0]], [X], color_metric, color_labs, lab_analysis, cut_axes, same_norm_fo, 0, fign_subp, pc_exp_var=np.nan, whatSess='_AallBall', fgn='', dim=[3,z])

# df['umap-1'] = embedding[:,0]
# df['umap-2'] = embedding[:,1]
# df['umap-3'] = embedding[:,2]
# X = df[['umap-1', 'umap-2', 'umap-3']]


################################################################################
################ Use GMM to cluster the UMAP embedding that you chose ################
################################################################################
#%% on the embedding found above, iterate GMM (different covariance types and # of clusters) with 5 fold validation

kf = KFold(5)
lowest_bic = np.infty
bic_array_n = []
bic_n = []
# n_components_range = range(2, 20)
# cv_types = ['spherical', 'tied', 'diag', 'full']
cv_types = ['spherical', 'diag', 'full'] # cv_type = 'tied'
for cv_type in cv_types:
    for n_components in n_components_range:
        print(f'{cv_type}, {n_components}')
        
        bic_temp =[]
        for train_index, test_index in kf.split(X):
            X_train = X[train_index]
            X_test = X[test_index]
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X_train)
            bic_temp.append(gmm.bic(X_test))

        bic_n.append(np.average(bic_temp))
        bic_array_n.append(bic_temp)
        


#%% Plot the output of the above analysis: BIC scores per GMM (most of this I pulled from someone elses code online)

import itertools        

bic = np.average(bic_array_n, axis =1)
std_bars = np.std(bic_array_n, axis = 1)

color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])

bars = []
start = 0

plt.figure(figsize=(12, 6))
spl = plt.subplot(1, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    end = ((i+1)*len(n_components_range))
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):(i + 1) * len(n_components_range)], width=.2, color=color, yerr = std_bars[start:end]))
    start = end

plt.xticks(n_components_range)
# plt.ylim([bic.min() * 0.99, bic.max()*1.01])
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + .2 * np.floor(bic.argmin() / len(n_components_range))
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types, loc=0)
#plt.savefig('GMM_BIC_scores.svg', format='svg', dpi=330)

plt.title('BIC Scores per GMM Configuration: 5 Fold Cross Validated\n(min_dist = .1, neighbors = 50)', fontsize = 20)
# plt.show()


if dosavefig:
    cre_short = f'allCre_BIC_allGMM_minDist{min_dist}_neigh{n_neighbors}'
    fgn_umap = f'{lab_analysis}_scatter_{fign_subp}allNeurSess'
#         if useSDK:
#             fgn_umap = f'{fgn_umap}_sdk_'
    nam = '%s_%s%s_%s_%s' %(cre_short, fgn_umap, '_AallBall', '', now)
    fign = os.path.join(dir0, dir_now, nam+fmt)     

    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    

    
    
    
################################################################
#%% Finally, again run GMM on umap embedding X, using for GMM your choice of cv type and n components, based on the bic values we got above
################################################################

covariance_type = 'spherical' #'full' #
n_components = 3
gmm = mixture.GaussianMixture(n_components = n_components, covariance_type = covariance_type)
# gmm = mixture.GaussianMixture(n_components=3, covariance_type='spherical')

gmm.fit(X)
labels = gmm.predict(X)



################################################################
#%% Plot UMAP; subplot 1: color using GMM clusters; subplot 2: color using cre lines
################################################################

x = X[:,0] #['umap-1']
y = X[:,1] #['umap-2']
z = X[:,2]# z =X['umap-3']


fig = plt.figure(figsize=(10,5))


#### Subplot 1: color using GMM clusters ####

colors = list(plt.get_cmap('tab10').colors)
color_list = []
for n in labels:
    color_list.append(colors[n])
    
# X['color'] = color_list
# df['color'] = color_list

ax3D = fig.add_subplot(121, projection='3d')
ax3D.scatter(x, y, z, s=3, c=color_list, marker='o')
# ax3D.scatter(x, y, s=3, c=color_list, marker='o')
# ax3D.scatter(x, y, z, s=10, c=X['color'], marker='o')

#ax3D.w_xaxis.set_ticks(np.arange(-5, 15, 5))
#ax3D.w_xaxis.set_label('UMAP Axis 1 (Arbitrary Units)')
ax3D.set_xlabel('UMAP Axis 1', fontsize = 11)
ax3D.set_ylabel('UMAP Axis 2', fontsize = 11)
ax3D.set_zlabel('UMAP Axis 3', fontsize = 11)

plt.title('UMAP Clusters (GMM)', fontsize = 11)
# plt.show()



#### Subplot 2: color using cre lines ####

cmap = plt.cm.jet #viridis #bwr #or any other colormap 
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

u = np.unique(amp)
c_value = np.full((len(amp)), colors[0]) # when colors is strings, eg '#1f77b4'
for ia in range(len(u)):
    c_value[amp==u[ia]] = colors[ia]

# fig = plt.figure()
ax3D = fig.add_subplot(122, projection='3d')
ax3D.scatter(x, y, z, s=3, c=c_value, marker='o')

plt.title('UMAP Clusters (cre line)', fontsize = 11)


if dosavefig:
    cre_short = f'allCre_bestGMMcolor_{covariance_type}_{n_components}comp_minDist{min_dist}_neigh{n_neighbors}'
    fgn_umap = f'{lab_analysis}_scatter_{fign_subp}allNeurSess'
#         if useSDK:
#             fgn_umap = f'{fgn_umap}_sdk_'
    nam = '%s_%s%s_%s_%s' %(cre_short, fgn_umap, '_AallBall', '', now)
    fign = os.path.join(dir0, dir_now, nam+fmt)     

    plt.savefig(fign, bbox_inches='tight') # , bbox_extra_artists=(lgd,)    

    

'''
xy = X
x = xy[:, 0]
y = xy[:, 1]
z = xy[:, 2]

plot_scatter_fo([cre_lines[0]], [X], color_metric, color_labs, lab_analysis, cut_axes, same_norm_fo, dosavefig, fign_subp, pc_exp_var=np.nan, whatSess='_AallBall', fgn='', dim=[3,z])

# plot_scatter_fo([cre_lines[0]], [X], color_metric, color_labs, lab_analysis, cut_axes, same_norm_fo, dosavefig, fign_subp, pc_exp_var=np.nan, whatSess='_AallBall', fgn='', dim=[2,2])
'''


# get the average traces for each cluster.
# maybe could help to cluster on a single session type? like ophys 1 images A, since we know VIP has omission responses for that session and should be quite distinct from Sst and Slc
# create github issue
# create presentation
# i think take the averages for each of the 4 gmm clusters
