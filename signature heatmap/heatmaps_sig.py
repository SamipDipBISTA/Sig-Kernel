#%%
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
###########################################################################################
from datetime import datetime

import esig
import matplotlib.pyplot as plt

import math

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

from quantick2.adjcoefs_df_from_bars_df import adjcoefs_df_from_bars_df
from quantick2.adjust_bars import adjust_bars
from quantick2.bars_df_from_trades_df import bars_df_from_trades_df
from quantick2.bars_df_from_quotes_df import bars_df_from_quotes_df
from quantick2.fetchquantick import fetchdata
from quantick2.get_cdh_boto3_session import get_cdh_boto3_session
from quantick2.read_configs import read_configs
from quantick2.ticks_df_from_quantick import ticks_df_from_quantick 

from helper import *
from run_pipeline import *

#%%
quantick_location = "local"


under = "TTF-Hi-Cal-51.6-ICE-ENDEX"
matu = "MA"
tz = "Europe--Paris"
from_ymd = "2025-01-01"
to_ymd = "2025-12-31"

with_SatSun = False

trades_df, quotes_df=fetchdata(quantick_location, under, matu, tz, from_ymd, to_ymd, postproc="None", with_SatSun=False)
#%%
dt = 60*2
daily_bars_hor = 60*dt  # duration in seconds of timebar (if 0, then 1 bar per day)
daily_bars_from_hms = "08:00:00"
daily_bars_to_hms = "18:00:00"



tbars_df_ = bars_df_from_trades_df(trades_df,
                                       daily_bars_hor,
                                       daily_bars_from_hms,
                                       daily_bars_to_hms)



tbars_df_['to_datetime'] = pd.to_datetime(tbars_df_['to_datetime'])
tbars_df_['close_datetime'] = pd.to_datetime(tbars_df_['close_datetime'])
tbars_df_['date'] = tbars_df_['to_datetime'].dt.date
#%%
tbars_df_['log_returns'] = np.log(tbars_df_['close']/tbars_df_['close'].shift(1))
tbars_df_.dropna(inplace=True)
tbars_df_['log_returns'].plot()
#%%
#tbars_df_.to_csv('tbars_df_.csv', index=False)
#%%
signatures_df = signatures(tbars_df=tbars_df_, level=12, cols_to_include=['close'],start_time=daily_bars_from_hms, end_time=daily_bars_to_hms, sig_type='sig')
signatures_df['date'] = pd.to_datetime(signatures_df['date'])
signatures_df.head()
#%%
#signatures_df.to_csv('signatures_df.csv', index=False)
#%%
plot_signatures_and_close(signatures_df, tbars_df_)
#%%
####--------------------------------------#####
####1 Clustering and heatmaps with scaling#####
####--------------------------------------#####
#Whole signatures with scaling
signatures_df=signatures_df.fillna(0)
X_scaled = scale_data(signatures_df)

umap_plot(signatures_df, X_scaled, color_by='month')

elbow_plot(X_scaled)
pca_clustering_workflow(signatures_df, scale_data, color_by='month') #PCA and then Cluster


#%%

#For few months--------------------------------------------------------------------------------------------------------------
X_scaled = scale_data(signatures_df)
#Cosine Similarity
cos_sim_matrix = cosine_similarity(X_scaled)
dates = signatures_df['date'].astype(str) 

#L2 distance matrix

norms = np.linalg.norm(X_scaled, axis=1).reshape(-1,1)
X_scaled = X_scaled/norms
sq_norms = np.sum(X_scaled ** 2, axis=1)

sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * X_scaled @ X_scaled.T
sq_dists = np.maximum(sq_dists, 0)

l2_dist_matrix_normalized = sq_dists
#l2_dist_matrix_normalized = cdist(X_scaled, X_scaled, metric='euclidean')
#l2_dist_matrix_normalized = l2_dist_matrix**2/(norms @ norms.T)
#Clip the matrix
#q_low, q_high = np.quantile(l2_dist_matrix_normalized, [0.10, 0.90])
#l2_dist_matrix_normalized = np.clip(l2_dist_matrix_normalized, q_low, q_high)

#%%
#Plots for few months
#plot_cosine_similarity_matrix(cos_sim_matrix, dates)
#plot_l2_distance_matrix(l2_dist_matrix_normalized, dates, cmap='coolwarm')
#----------------------------------------------------------------------------------------------------------------------------
#%%
#Plots for year
dates = signatures_df['date']
date_str = dates.dt.strftime("%Y-%m-%d")

# indices of first day of each month
first_days = dates.groupby([dates.dt.year, dates.dt.month]).head(1).index
print('Standard scaling')
#Cosine similarity
plot_cosine_similarity_matrix_long(cos_sim_matrix, first_days, date_str)
#L2 dist
plot_l2_distance_matrix_long(l2_dist_matrix_normalized, first_days, date_str, cmap='coolwarm_r')

#%%
########################################################################################################################
########################################################################################################################
########################################################################################################################
####--------------------------------------#####
###2 Heatmaps without scaling###
####--------------------------------------#####

#For few months--------------------------------------------------------------------------------------------------------------
X = signatures_df.iloc[:,1:].to_numpy()
#cos sim
cos_sim_matrix = cosine_similarity(X)
#L2
norms = np.linalg.norm(X, axis=1).reshape(-1,1)
X = X/norms
sq_norms = np.sum(X ** 2, axis=1)
sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * X @ X.T
sq_dists = np.maximum(sq_dists, 0)

l2_dist_matrix_normalized = sq_dists
#%%
#Plot inter month
plot_cosine_similarity_matrix(cos_sim_matrix, dates)

plot_l2_distance_matrix(l2_dist_matrix_normalized, dates, cmap='coolwarm_r')
#----------------------------------------------------------------------------------------------------------------------------
#%%
print('No scaling')
# For a year
dates = signatures_df['date']
date_str = dates.dt.strftime("%Y-%m-%d")

# indices of first day of each month
first_days = dates.groupby([dates.dt.year, dates.dt.month]).head(1).index
#Cosine similarity
plot_cosine_similarity_matrix_long(cos_sim_matrix, first_days, date_str)
#L2 dist
plot_l2_distance_matrix_long(l2_dist_matrix_normalized, first_days, date_str, cmap='coolwarm_r')
#%%

###############################################
#Plot over few months

plot_cos_l2_for_all_months(signatures_df, scale=True)
#%%

#%%
# Clustering --> PCA
'''
labels = perform_kmeans(X_scaled, n_clusters=2)
pca_emb = perform_pca(X_scaled, n_components=2)
#plot_pca(signatures_df, pca_emb, labels=labels, color_by='month')
best_score = -1
best_k = 2
for k in range(2, 11):
    labels_test = perform_kmeans(X_scaled, n_clusters=k)
    score = silhouette_score(X_scaled, labels_test)
    if score > best_score:
        best_score = score
        best_k = k

labels_best = perform_kmeans(X_scaled, n_clusters=best_k)
plot_pca(signatures_df, pca_emb, labels=labels_best, color_by='month')
print(f"Best k: {best_k}, Silhouette score: {best_score:.3f}")

unique_clusters = np.unique(labels)
for cluster in unique_clusters:
    mask = (labels == cluster)
    print(f'Plot for cluster {cluster}')
    plot_pca(signatures_df.iloc[mask],pca_emb[mask],labels=None,color_by='month')
'''

#%% 
# signatures with 9,10,12 dropped
X_scaled_dropped = scale_data(signatures_df.drop(columns=['sig_9', 'sig_10', 'sig_12']))
umap_plot(signatures_df, X_scaled_dropped, color_by='month')
elbow_plot(X_scaled_dropped) 
#%%
# PCA -> Clustering
pca_clustering_workflow(signatures_df.drop(columns=['sig_9', 'sig_10', 'sig_12']), scale_data)
# %%
