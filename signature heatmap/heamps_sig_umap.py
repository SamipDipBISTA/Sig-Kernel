#%%
from datetime import datetime
import iisignature
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.metrics import silhouette_score

import umap
#%%
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
#%% 
df = pd.read_csv('signature_data_5min.csv')
df['to_datetime'] = pd.to_datetime(df['to_datetime'])
df['time'] = df['to_datetime'].dt.time
df['date'] = pd.to_datetime(df['to_datetime'].dt.date)

df = df[df['date']<pd.to_datetime('2025-01-01')]
#%%
cols = ['cumul_log_ret', 'local_volatility', 'volume','vwap_deviation']

signatures_df = signatures(df, 5, cols,date_col='date',time_col= 'to_datetime')
signatures_df = signatures_df.set_index("date").reset_index()
signatures_half_day_morning_df = signatures(df, 5, cols,date_col='date',time_col= 'to_datetime', start_time="08:00:00", end_time="13:00:00")
signatures_half_day_evening_df = signatures(df, 5, cols,date_col='date',time_col= 'to_datetime', start_time="13:00:00", end_time="18:00:00")
#%%
#signatures_df = signatures_half_day_evening_df
#%%

time = signatures_df['date'].dt.year #+ (signatures_df['date'].dt.month-1)/12
X = signatures_df.iloc[:, 1:]
scaler = StandardScaler()
scaler.fit_transform(X)
dates = pd.to_datetime(signatures_df.iloc[:, 0])

color_by = 'year'
if color_by == 'month':
    colors = dates.dt.month
elif color_by == 'day':
    colors = dates.dt.day
elif color_by == 'year':
    colors = dates.dt.year

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X)
plt.scatter(embedding[:, 0], embedding[:, 1], c=time, cmap='viridis', s=10)
plt.colorbar(label='Time')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()

#%%
x = embedding[:, 0]
y = embedding[:, 1]
z = time #signatures_df['date'].dt.year #

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(-x, y, z, c=time, cmap='winter', s=5, alpha=0.7)

ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('')

# colorbar
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)

plt.show()
# %%
plt.scatter(z,x)
#%%
plt.scatter(z,y)

#%%
def kmeans_silhouette_pca(X, k_min=2, k_max=5, model='kmeans', eps=0.5, min_samples=5):
    best_score = -1
    best_k = k_min

    for k in range(k_min, k_max):
        kmeans = KMeans(n_clusters=k, n_init=50, random_state=42).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_k = k

    kmeans = KMeans(n_clusters=best_k, n_init=5, random_state=42).fit(X)
    spectral = SpectralClustering(n_clusters=6).fit(X)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    
    if model=='kmeans':
        chosen_model = kmeans
        print('Best k', best_k)
    elif model=='spectral':
        chosen_model = spectral
    else:
        chosen_model = dbscan
        
    plt.figure(figsize=(8,6))
    plt.scatter(X[:,0], X[:,1], c=chosen_model.labels_, cmap='tab20', s=50)
    plt.colorbar()
    #plt.title(f'Best k (for kmeans): {best_k}')
    plt.show()
    return chosen_model
# %%
model = kmeans_silhouette_pca(embedding, k_min=2, k_max=10, model = 'dbscan', eps= 0.5)


# %%
model.labels_