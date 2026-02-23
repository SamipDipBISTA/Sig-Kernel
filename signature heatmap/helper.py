#%%
from datetime import datetime
import iisignature

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

import umap


#Signatures functions
def compute_signatures(df, level, time, sig_type, *cols):
    if time:
        sequence_time = df[time].apply(lambda t: t.hour*3600 + t.minute*60 + t.second + t.microsecond/1e6)/3600
        sequence_time = np.insert(sequence_time.to_numpy(), 0, 0)
        sequences = [sequence_time] + [np.insert(df[col].to_numpy(), 0, 0) for col in cols]
    else:
        sequences = [np.insert(df[col].to_numpy(), 0, 0) for col in cols]
    path = np.column_stack(sequences)
    if sig_type == 'sig':
        return iisignature.sig(path,level)#esig.stream2sig(path, level)
    elif sig_type == 'logsig':
        return None#esig.stream2logsig(path, level)

def signatures(tbars_df, level, cols_to_include, date_col, time_col = 'close_datetime', start_time=None, end_time=None, sig_type='sig'):
    if start_time and end_time:
        start_time = datetime.strptime(start_time, "%H:%M:%S").time()
        end_time = datetime.strptime(end_time, "%H:%M:%S").time()

    signature_results = []
    for day, group in tbars_df.groupby(date_col):
        if start_time and end_time:
            group = group[
                (group['to_datetime'].dt.time >= start_time) &
                (group['to_datetime'].dt.time <= end_time)
            ]
        if len(group) == 0:
            continue
        sig = compute_signatures(group, level, time_col, sig_type, *cols_to_include)
        sig_dict = {'date': day}
        sig_dict.update({f'sig_{i+1}': val for i, val in enumerate(sig)})
        signature_results.append(sig_dict)

    signatures_df = pd.DataFrame(signature_results)
    return signatures_df

def signatures_per_hours(tbars_df, level, cols_to_include, sig_type):
    signature_results = []
    for hour, group in tbars_df.groupby('hour'):
        if len(group) == 0:
            continue
        sig = compute_signatures(group, level, 'close_datetime', sig_type, *cols_to_include)
        sig_dict = {'hour': hour}
        sig_dict.update({f'sig_{i+1}': val for i, val in enumerate(sig)})
        signature_results.append(sig_dict)
    signatures_df = pd.DataFrame(signature_results)
    return signatures_df


#Plot functions
def plot_signatures_and_close(df, tbars_df, n_cols=3):
    date_col = df.columns[0]
    cols_to_plot = [col for col in df.columns[1:] if df[col].std() != 0]

    n_rows = math.ceil(len(cols_to_plot) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cols_to_plot):
        axes[i].plot(df[date_col], df[col])
        axes[i].set_title(col)
        axes[i].set_ylabel('Value')
        axes[i].tick_params(axis='x', rotation=45)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(tbars_df['close'])
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.title('Close over time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#Clustering 

def scale_data(df):
    X = df.iloc[:, 1:]
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def umap_plot(df, X_scaled, color_by='month'):
    dates = pd.to_datetime(df.iloc[:, 0])
    if color_by == 'month':
        colors = dates.dt.month
    elif color_by == 'day':
        colors = dates.dt.day
    elif color_by == 'year':
        colors = dates.dt.year
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap='viridis', s=10)
    plt.colorbar(label=color_by.capitalize())
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.show()

def elbow_plot(X_scaled, k_range=range(1, 10)):
    inertia = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

def perform_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca

def perform_kmeans(X, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X) + 1
    return labels

def plot_pca(df, pca_embedding, labels=None, color_by='day'):
    dates = pd.to_datetime(df.iloc[:, 0])
    if color_by == 'month':
        colors = dates.dt.month
    elif color_by == 'day':
        colors = dates.dt.day
    elif color_by == 'year':
        colors = dates.dt.year
    else:
        raise ValueError("color_by must be 'day' or 'month' or 'year.")

    n_subplots = 1 + (labels is not None)
    fig, axes = plt.subplots(1, n_subplots, figsize=(6 * n_subplots, 5))

    if labels is not None:
        scatter0 = axes[0].scatter(pca_embedding[:, 0], pca_embedding[:, 1], c=labels, cmap='Set1', s=10)
        axes[0].set_xlabel('PCA 1')
        axes[0].set_ylabel('PCA 2')
        axes[0].set_title('Clusters')
        fig.colorbar(scatter0, ax=axes[0], label='Cluster')

        scatter1 = axes[1].scatter(pca_embedding[:, 0], pca_embedding[:, 1], c=colors, cmap='viridis', s=10)
        axes[1].set_xlabel('PCA 1')
        axes[1].set_ylabel('PCA 2')
        axes[1].set_title(f'PCA colored by {color_by}')
        fig.colorbar(scatter1, ax=axes[1], label=color_by.capitalize())
    else:
        scatter = axes.scatter(pca_embedding[:, 0], pca_embedding[:, 1], c=colors, cmap='viridis', s=10)
        axes.set_xlabel('PCA 1')
        axes.set_ylabel('PCA 2')
        axes.set_title(f'PCA colored by {color_by}')
        fig.colorbar(scatter, ax=axes, label=color_by.capitalize())

    plt.tight_layout()
    plt.show()

#Cosine sim and L2 plots
def plot_cos_sim_intraday(cos_sim_matrix, hours, annot=True):
    plt.figure(figsize=(12,10))
    sns.heatmap(cos_sim_matrix, annot=annot, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Cosine Similarity Heatmap")
    plt.xlabel("Hour")
    plt.ylabel("Hour")
    plt.xticks(ticks=range(len(hours)), labels=hours, rotation=0)
    plt.yticks(ticks=range(len(hours)), labels=hours, rotation=0)
    plt.tight_layout()
    plt.show()

def plot_cosine_similarity_matrix(cos_sim_matrix, dates, figsize=(12,10), annot=True, cmap="coolwarm"):
    plt.figure(figsize=figsize)
    sns.heatmap(cos_sim_matrix, annot=annot, fmt=".2f", cmap=cmap, square=True,
                xticklabels=dates, yticklabels=dates)
    plt.title("Cosine Similarity Heatmap")
    plt.xlabel("Date")
    plt.ylabel("Date")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_l2_distance_matrix(l2_dist_matrix, dates, figsize=(12,10), annot=True, cmap="coolwarm"):
    plt.figure(figsize=figsize)
    sns.heatmap(l2_dist_matrix, annot=annot, fmt=".2f", cmap=cmap, square=True,
                xticklabels=dates, yticklabels=dates)
    plt.title("L2 Distance Heatmap")
    plt.xlabel("Dates")
    plt.ylabel("Dates")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_cosine_similarity_matrix_long(cos_sim_matrix, tick_indices, tick_labels, figsize=(12,10), annot=False, cmap="coolwarm"):
    plt.figure(figsize=figsize)
    sns.heatmap(cos_sim_matrix, annot=annot, fmt=".2f", cmap=cmap, square=True)
    plt.title("Cosine Similarity Heatmap")
    plt.xlabel("Date")
    plt.ylabel("Date")
    plt.xticks(tick_indices, tick_labels[tick_indices], rotation=45)
    plt.yticks(tick_indices, tick_labels[tick_indices])
    plt.tight_layout()
    plt.show()

def plot_l2_distance_matrix_long(l2_dist_matrix, tick_indices, tick_labels, figsize=(12,10), annot=False, cmap="coolwarm"):
    plt.figure(figsize=figsize)
    sns.heatmap(l2_dist_matrix, annot=annot, fmt=".2f", cmap=cmap, square=True)
    plt.title("L2 Distance Heatmap")
    plt.xlabel("Dates")
    plt.ylabel("Dates")
    plt.xticks(tick_indices, tick_labels[tick_indices], rotation=45)
    plt.yticks(tick_indices, tick_labels[tick_indices])
    plt.tight_layout()
    plt.show()


def plot_cos_l2_for_all_months(df, scale=False,  cmap='coolwarm_r'):
    df = df.copy()
    df['months'] = pd.to_datetime(df['date']).dt.to_period('M')
    dates = df['date']
    for month, group in df.groupby('months'):
        X = group.drop(columns=['date', 'months']).values
        if scale:
            scaler = StandardScaler()
            X=scaler.fit_transform(X)
        cos_sim_matrix = cosine_similarity(X)
        print(f"Heatmap for {month}")
        plot_cosine_similarity_matrix(cos_sim_matrix, group['date'])
        '''
        norms = np.linalg.norm(X, axis=1).reshape(-1,1)
        X = X/norms
        sq_norms = np.sum(X ** 2, axis=1)
        sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2 * X @ X.T
        sq_dists = np.maximum(sq_dists, 0)

        l2_dist_matrix_normalized = sq_dists
        plot_l2_distance_matrix(l2_dist_matrix_normalized, group['date'], cmap=cmap)
        '''
    
def plot_cos_l2_subplots_for_all_months(df, scale=False, cmap='coolwarm'):
    df = df.copy()
    df['months'] = pd.to_datetime(df['date']).dt.to_period('M')

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    for ax in axes[12:]:
        ax.set_visible(False)
    i = 0
    for month, group in df.groupby('months'):
        X = group.drop(columns=['date', 'months']).values
        if scale:
            X = StandardScaler().fit_transform(X)
        cos_sim = cosine_similarity(X)
        ax = axes[i]
        ax.set_title(str(month))
        im = ax.imshow(cos_sim, cmap=cmap, aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])
        i+=1
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def gk_daily(df):
    H = df['high']
    L = df['low']
    O = df['open']
    C = df['close']
    
    log_HL = np.log(H / L)
    log_CO = np.log(C / O)
    
    daily_var = 0.5 * log_HL**2 - (2 * np.log(2) - 1) * log_CO**2
    daily_var = np.clip(daily_var, a_min=0, a_max=None)
    
    daily_vol = np.sqrt(daily_var)
    return daily_vol
#%%
if __name__ == "__main__":
#%%
    df_test = pd.read_csv('signatures_df.csv')
#%%
    plot_cos_l2_for_all_months(df_test, scale=True)
#%%
    plot_cos_l2_subplots_for_all_months(df_test, scale=False)
# %%
