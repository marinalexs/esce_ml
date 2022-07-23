from select import kevent
import streamlit as st
import pandas as pd
import numpy as np
import glob
import plotly.express as px
import yaml
import plotly.graph_objs as go


st.title("ESCE Viewer")


# @st.cache
def get_available_results():
    return glob.glob("results/*/statistics/*/*.stats.json")


available_results = get_available_results()
print(available_results)

df = pd.DataFrame(available_results, columns=["full_path"])
df[
    [
        "dataset",
        "model",
        "features",
        "features_trafo",
        "target",
        "targets_trafo",
        "matching",
        "grid",
    ]
] = (
    df["full_path"]
    .str.replace("results/", "")
    .str.replace("statistics/", "")
    .str.replace(".stats.json", "")
    .str.replace("/", "_")
    .str.split("_", expand=True)
)

print(df)

available_targets = df["target"].unique()
selected_targets = st.multiselect(label="Targets:", options=available_targets)

available_features = df[df["target"].isin(
    selected_targets)]["features"].unique()
selected_features = st.multiselect(
    label="Features:", options=available_features)

available_models = df[
    (df["target"].isin(selected_targets)) & (
        df["features"].isin(selected_features))
]["model"].unique()
selected_models = st.multiselect(label="Models:", options=available_models)

available_grids = df[
    (df["target"].isin(selected_targets)) & 
    (df["features"].isin(selected_features)) & 
    (df["model"].isin(selected_models))
]["grid"].unique()
grid = st.sidebar.selectbox(label="HP grid", options=available_grids, index=0)

max_x = st.sidebar.number_input(
    label="Extrapolation sample size", value=5, step=1)

# classification_metrics = [
#     "r2_test",
#     "mse_test",
#     "mae_test",
#     "r2_val",
#     "mse_val",
#     "mae_val",
#     "r2_train",
#     "mse_train",
#     "mae_train",
# ]
# regression_metrics = [
#     "acc_test",
#     "f1_test",
#     "acc_val",
#     "f1_val",
#     "acc_train",
#     "f1_train",
# ]
# selected_metrics = st.sidebar.multiselect(
#     label="Metrics:",
#     options=classification_metrics + regression_metrics,
#     default=["r2_test", "acc_test"],
# )


color_variable = st.sidebar.selectbox(
    label="Color", options=[None, "target", "features", "model"], index=1, key='color_variable')
linestyle_variable = st.sidebar.selectbox(
    label="Linestyle",options=[None, "target", "features", "model"], index=2, key='linestyle_variable')
shape_variable = st.sidebar.selectbox(
    label='Shape',options=[None, "target", "features", "model"], index=3, key='shape_variable')


df_selected = df[
    (df["target"].isin(selected_targets))
    & (df["features"].isin(selected_features))
    & (df["model"].isin(selected_models))
    & (df["grid"] == grid)
]


# st.dataframe(df_selected[df_selected.columns[1:]])

if len(df_selected) > 0:
    data = []
    for _, row in df_selected.iterrows():
        with open(row.full_path) as f:
            score = yaml.safe_load(f)
        df_ = pd.DataFrame({'n':score['x'],'y':score['y_mean'],'y_std':score['y_std']})
        df_["model"] = row.model
        df_["features"] = row.features
        df_["target"] = row.target
        # x_exp = np.linspace(128,2*max(score['x']))
        # y_exp = score['p_mean'][0] * np.power(x_exp,-score['p_mean'][1]) + score['p_mean'][2]
        data.append(df_)

    data = pd.concat(data, axis=0, ignore_index=True)
    # print(data)

    fig = px.line(data, x="n", y='y',error_y='y_std',color=color_variable,line_dash=linestyle_variable,symbol=shape_variable, log_x=True, template="simple_white")
    fig.update_layout( plot_bgcolor='white')

    for i,(_, row) in enumerate(df_selected.iterrows()):
        with open(row.full_path.replace('stats.json','bootstrap.json')) as f:
            p = yaml.safe_load(f)
        for p_ in p:
            x_exp = np.logspace(np.log10(128),max_x)
            y_exp = p_[0] * np.power(x_exp,-p_[1]) + p_[2]
            fig.add_trace(go.Scatter(x=x_exp, y=y_exp, line=dict(color=fig.data[i].line.color),opacity=5/len(p), showlegend=False))
    
    st.plotly_chart(fig)



# if len(df_selected) > 0:
#     data = []
#     for _, row in df_selected.iterrows():
#         df_ = pd.read_csv(row.full_path, index_col=False)
#         df_["model"] = row.model
#         df_["features"] = row.features
#         df_["target"] = row.target
#         data.append(df_)
#     data = pd.concat(data, axis=0, ignore_index=True)

#     result = data.groupby(['n', 'model', 'features', 'target'], as_index=False).agg(['mean', 'std'])
#     result =  result.reindex(columns=sorted(result.columns))
#     print(result)



#     fig = px.line(data, x="n", y='acc_test', title='Life expectancy in Canada')
#     st.plotly_chart(fig)
