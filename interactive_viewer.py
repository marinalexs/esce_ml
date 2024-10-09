import glob
import os
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import yaml
import re
import altair as alt
import streamlit as st
import logging

# Set page config at the very beginning
st.set_page_config(page_title="Interactive ESCE", layout="wide")

# Configure logging
log_level = os.environ.get('ESCE_LOG_LEVEL', 'WARNING').upper()
logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure YAML loader
loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(r'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.')
)

def get_available_results(directory: str = 'results') -> List[str]:
    return glob.glob(f"{directory}/**/*.stats.json", recursive=True)

def extract_metadata(path: str) -> pd.Series:
    parts = [Path(path).parts[1], Path(path).parts[3]] + Path(path).parts[4].split('_')
    columns = [
        "dataset", "model", "features", "target",
        "confound-correction-method", "confound-correction-cni",
        "balanced","quantile-transform", "grid",
    ]
    return pd.Series(parts + [None] * (len(columns) - len(parts)), index=columns)

def read_results_metadata(available_results: List[str]) -> pd.DataFrame:
    logger.info(f"Processing {len(available_results)} result files")
    df = pd.DataFrame(available_results, columns=["full_path"])
    df = df.join(df["full_path"].apply(extract_metadata))
    
    non_empty_files = [
        file_path for file_path in df["full_path"]
        if yaml.load(open(file_path), Loader=yaml.SafeLoader)
    ]
    df = df[df["full_path"].isin(non_empty_files)]
    df["cni"] = df.apply(lambda row: f"{row['confound-correction-method']}-{row['confound-correction-cni']}", axis=1)
    
    logger.info(f"Processed {len(df)} non-empty result files")
    return df

def load_data(results_metadata: pd.DataFrame) -> pd.DataFrame:
    data = []
    for _, row in results_metadata.iterrows():
        try:
            with open(row.full_path) as f:
                score = yaml.load(f, Loader=loader)
            if not score:
                logger.warning(f"{row.full_path} is empty - skipping")
                continue
            
            df_ = pd.DataFrame({"n": score["x"], "y": score["y_mean"], "y_std": score["y_std"]})
            for col in ['dataset', 'features', 'target', 'model', 'cni', 'confound-correction-method', 'confound-correction-cni', 'balanced','quantile-transform', 'grid']:
                if col in row.index:
                    df_[col] = row[col]
            data.append(df_)
        except Exception as e:
            logger.error(f"Error processing file {row.full_path}: {str(e)}")

    if not data:
        logger.warning("No data to plot.")
        return pd.DataFrame()

    data = pd.concat(data, ignore_index=True)
    data['y'] = data['y'].clip(lower=-1)
    data['y_std'] = pd.to_numeric(data['y_std'], errors='coerce').clip(upper=1).fillna(0)
    data.columns = data.columns.str.replace('_', '-')

    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].str.replace('_', '-')

    category_columns = ['dataset', 'features', 'target', 'model', 'cni', 'confound-correction-method', 'confound-correction-cni', 'balanced','quantile-transform', 'grid']
    data['id'] = data[category_columns].astype(str).agg('-'.join, axis=1)

    logger.info("Data processed successfully")
    return data

def create_chart(data: pd.DataFrame, selected_categories: Dict[str, List[str]]) -> alt.Chart:
    for col, values in selected_categories.items():
        if values:
            data = data[data[col].isin(values)]
    
    color_scale = alt.Scale(scheme='tableau20')

    # Add a selection for highlighting on hover
    highlight = alt.selection_single(on='mouseover', fields=['id'], nearest=True)

    base = alt.Chart(data).encode(
        x=alt.X('n:Q', 
                title='Sample Size', 
                scale=alt.Scale(type='log', 
                                domain=[data['n'].min() * 0.9, data['n'].max() * 1.1]),
                                axis=alt.Axis(format='~s')
                                ),
        y=alt.Y('y:Q', 
                title='Performance Metric'),
        color=alt.Color('id:N', scale=color_scale, legend=None),
        tooltip=list(selected_categories.keys()) + ['n', 'y', 'y-std'] 
    )

    # Create lines with highlighting
    lines = base.mark_line(strokeWidth=2).encode(
        opacity=alt.condition(highlight, alt.value(1), alt.value(0.5)),
        size=alt.condition(highlight, alt.value(3), alt.value(2))
    ).add_selection(highlight)

    # Create error bars
    error_bars = base.mark_errorbar(thickness=2, ticks=True).encode(
        y='y_min:Q',
        y2='y_max:Q'
    ).transform_calculate(
        y_min="datum.y - datum['y-std']",
        y_max="datum.y + datum['y-std']"
    )

    chart = (error_bars + lines).properties(width=700, height=400)

    legend_data = pd.DataFrame({'id': data['id'].unique()}).sort_values('id')
    legend = alt.Chart(legend_data).mark_square(size=100).encode(
        y=alt.Y('id:N', axis=None, sort=None),
        color=alt.Color('id:N', scale=color_scale, legend=None),
        x=alt.value(0)
    )

    legend_text = alt.Chart(legend_data).mark_text(align='left', dx=15).encode(
        y=alt.Y('id:N', axis=None, sort=None),
        text='id:N',
        x=alt.value(20)
    )

    legend_combined = (legend + legend_text).properties(
        width=700,
        height=len(legend_data) * 25
    )

    return alt.vconcat(chart, legend_combined).resolve_scale(
        color='shared'
    ).properties(
        # title="Interactive Line Plot with Legend"
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )

def main():
    st.title("Interactive ESCE Viewer")

    available_results = get_available_results()
    results_metadata = read_results_metadata(available_results)
    data = load_data(results_metadata)

    if data.empty:
        st.warning("No data available for plotting.")
        return

    st.sidebar.header("Filters")
    category_columns = ['dataset', 'features', 'target', 'model', 'confound-correction-method', 'confound-correction-cni', 'balanced','quantile-transform', 'grid']
    selected_categories = {
        col: st.sidebar.multiselect(f"Select [{col}]", data[col].unique().tolist(), default=data[col].unique().tolist())
        for col in category_columns
    }

    chart = create_chart(data, selected_categories)
    st.altair_chart(chart, use_container_width=True)

    # # Display the data DataFrame at the bottom
    # st.subheader("Data Table")
    # st.dataframe(data)

if __name__ == "__main__":
    main()