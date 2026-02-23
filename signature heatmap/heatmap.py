#%%
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

from helper import *
from run_pipeline import *

#%% 
df = pd.read_csv('signature_data_5min.csv')
df['to_datetime'] = pd.to_datetime(df['to_datetime'])
df['time'] = df['to_datetime'].dt.time
df['date'] = df['to_datetime'].dt.date
df.head()
#%%
cols = ['cumul_log_ret', 'local_volatility', 'volume','vwap_deviation']

signatures_df = signatures(df, 2, cols,date_col='date',time_col= 'to_datetime')
signatures_df = signatures_df.set_index("date").reset_index()
signatures_half_day_morning_df = signatures(df, 2, cols,date_col='date',time_col= 'to_datetime', start_time="08:00:00", end_time="13:00:00")
signatures_half_day_evening_df = signatures(df, 2, cols,date_col='date',time_col= 'to_datetime', start_time="13:00:00", end_time="18:00:00")


#%%
def run_signature_analysis(df, scale_func, color_by='year'):
    X_scaled = scale_func(df)
    elbow_plot(X_scaled)
    X_pca, labels, k, score = pca_clustering_workflow(df, scale_func, color_by=color_by)

    cos_sim_matrix = cosine_similarity(X_scaled)
    dates = pd.to_datetime(df['date'])
    date_str = dates.dt.strftime("%Y-%m-%d")
    first_days = dates.groupby([dates.dt.year, dates.dt.month]).head(1).index

    plot_cosine_similarity_matrix_long(cos_sim_matrix, first_days, date_str)

    return X_scaled, X_pca, labels, k, score, cos_sim_matrix
#%%
run_signature_analysis(signatures_df, scale_data, color_by='year')
#%%
X_scaled = scale_data(signatures_df)
#umap_plot(signatures_df, X_scaled, color_by='year')
#%% 
run_signature_analysis(signatures_half_day_morning_df, scale_data, color_by='day')
X_scaled = scale_data(signatures_half_day_morning_df)
#umap_plot(signatures_half_day_morning_df, X_scaled, color_by='year')
#%%
run_signature_analysis(signatures_half_day_evening_df, scale_data, color_by='year')
# %%



signatures_1 = signatures(df, 2, cols,date_col='date',time_col= 'to_datetime', start_time="08:00:00", end_time="10:00:00")
signatures_2 = signatures(df, 2, cols,date_col='date',time_col= 'to_datetime', start_time="10:00:00", end_time="12:00:00")
signatures_3 = signatures(df, 2, cols,date_col='date',time_col= 'to_datetime', start_time="12:00:00", end_time="14:00:00")
signatures_4 = signatures(df, 2, cols,date_col='date',time_col= 'to_datetime', start_time="14:00:00", end_time="16:00:00")
signatures_5 = signatures(df, 2, cols,date_col='date',time_col= 'to_datetime', start_time="16:00:00", end_time="18:00:00")
run_signature_analysis(signatures_1, scale_data, color_by='year')
run_signature_analysis(signatures_2, scale_data, color_by='year')
run_signature_analysis(signatures_3, scale_data, color_by='year')
run_signature_analysis(signatures_4, scale_data, color_by='year')
run_signature_analysis(signatures_5, scale_data, color_by='year')