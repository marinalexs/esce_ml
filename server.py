from tkinter.font import names
import streamlit as st
import pandas as pd
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt

st.title('ESCE Viewer')


@st.cache
def get_available_results():
    return glob.glob('results/stats/*/*/*')


available_results = get_available_results()

df = pd.DataFrame(available_results, columns=['full_path'])
df[['model', 'dataset', 'features', 'target', 'n', 'stratify', 'grid']] = df['full_path'].str.replace(
    'results/stats/', '').str.replace('.csv', '').str.replace('/', '_').str.split('_', expand=True)

available_targets = df['target'].unique()
selected_targets = st.multiselect(label='Targets:', options=available_targets)

available_features = df[df['target'].isin(
    selected_targets)]['features'].unique()
selected_features = st.multiselect(
    label='Features:', options=available_features)

available_models = df[(df['target'].isin(
    selected_targets)) & (df['features'].isin(
        selected_features))]['model'].unique()
selected_models = st.multiselect(label='Models:', options=available_models)


grid = st.sidebar.selectbox(label='HP grid', options=['default'], index=0)

st.sidebar.number_input(
    label='Extrapolation sample size', value=100000, step=1000)

classification_metrics = ['r2_test', 'mse_test', 'mae_test',  'r2_val',
                          'mse_val', 'mae_val',  'r2_train', 'mse_train',
                          'mae_train']
regression_metrics = ['acc_test', 'f1_test',
                      'acc_val', 'f1_val', 'acc_train', 'f1_train']
selected_metrics = st.sidebar.multiselect(
    label='Metrics:', options=classification_metrics+regression_metrics, default=['r2_test', 'acc_test'])

color_variable = st.sidebar.selectbox(
    label='Color', options=['target', 'features', 'model'], index=0)
linestyle_variable = st.sidebar.selectbox(
    label='Linestyle', options=['target', 'features', 'model', 'metric'], index=1)
# shape_variable = st.sidebar.selectbox(
#     label='Shape not implememted', options=['target', 'features', 'model', 'metric'], index=2)
# alpha_variable = st.sidebar.selectbox(
#     label='Alpha not implememted', options=['target', 'features', 'model', 'metric'], index=3)

df_selected = df[(df['target'].isin(selected_targets)) &
                 (df['features'].isin(selected_features)) &
                 (df['model'].isin(selected_models)) &
                 (df['grid'] == grid)]


if len(df_selected) > 0:
    data = []
    for _, row in df_selected.iterrows():
        df_ = pd.read_csv(row.full_path, index_col=False)
        df_['model'] = row.model
        df_['features'] = row.features
        df_['target'] = row.target
        data.append(df_)
    data = pd.concat(data, axis=0, ignore_index=True)

    print(data)

    for m in ['acc', 'f1']:
        fig = plt.figure(figsize=(10, 4))
        flag = False
        for s in ['test', 'val', 'train']:
            metric = f'{m}_{s}'
            if metric in selected_metrics and metric in data.columns:
                sns.lineplot(x="n", y=metric, data=data,
                             hue=color_variable, style=linestyle_variable)
                flag = True
        if flag:
            st.pyplot(fig)
    for m in ['r2', 'mae', 'mse']:
        fig = plt.figure(figsize=(10, 4))
        flag = False
        for s in ['test', 'val', 'train']:
            metric = f'{m}_{s}'
            if metric in selected_metrics and metric in data.columns:
                sns.lineplot(x="n", y=metric, data=data, hue='model')
                flag = True
        if flag:
            st.pyplot(fig)

    # st.dataframe(data)
else:
    st.text('empty selection')


# def load_data(nrows):
#     data = pd.read_csv(DATA_URL, nrows=nrows)
#     def lowercase(x): return str(x).lower()
#     data.rename(lowercase, axis='columns', inplace=True)
#     data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
#     return data


# # Create a text element and let the reader know the data is loading.
# data_load_state = st.text('Loading data...')
# # Load 10,000 rows of data into the dataframe.
# data = load_data(10000)
# # Notify the reader that the data was successfully loaded.
# data_load_state.text('Loading data...done!')


# if st.checkbox('Show raw data'):
#     st.subheader('Raw data')
#     st.write(data)


# st.subheader('Number of pickups by hour')

# hist_values = np.histogram(
#     data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24))[0]

# st.bar_chart(hist_values)


# # min: 0h, max: 23h, default: 17h
# hour_to_filter = st.slider('hour', 0, 23, 17)
# filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
# st.subheader(f'Map of all pickups at {hour_to_filter}:00')
# st.map(filtered_data)
