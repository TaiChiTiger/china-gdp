import os
os.environ['NUMEXPR_NUM_THREADS'] = '8'
import geopandas as gpd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from bokeh.models import GeoJSONDataSource
from bokeh.plotting import figure
from bokeh.models import CategoricalColorMapper, LinearColorMapper, ColorBar
from bokeh.palettes import brewer
from pysal.lib import weights
from scipy import stats
from pysal.explore import esda
from pysal.viz.splot.esda import plot_moran
import json

@st.echo()
def load_data():
    china_gdp = gpd.read_file("./datasets/China-GDP-2000-2016.shp")
    return china_gdp

def get_geodatasource(gdf):
    """Get getjsondatasource from geopandas object"""
    json_data = json.dumps(json.loads(gdf.to_json()))
    return GeoJSONDataSource(geojson = json_data)

def bokeh_plot_map(gdf, column=None, n_colors=8, title=''):
    """Plot bokeh map from GeoJSONDataSource """

    geosource = get_geodatasource(gdf)
    palette = brewer['Oranges'][n_colors]
    vals = gdf[column]
    color_mapper = LinearColorMapper(palette=palette[::-1], low=vals.min(), high=vals.max())
    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8, width=500, height=20,
                         location=(0,0), orientation='horizontal')
    tools = "tap,pan,box_zoom,reset"
    p = figure(title = title, toolbar_location='right', tools=tools)
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.visible = False
    p.yaxis.visible = False
    #Add patch renderer to figure
    p.patches('xs','ys', source=geosource, fill_alpha=1, line_width=0.5, line_color='black',
              fill_color={'field' :column , 'transform': color_mapper})
    p.add_layout(color_bar, 'below')

    return p

def plot_moran_map(gdf, column, palette, title=""):
    geosource = get_geodatasource(gdf)
    color_mapper = CategoricalColorMapper(palette=palette, factors=np.sort(gdf[column].unique()))
    tools = "tap,pan,box_zoom,reset"
    p = figure(title=title, toolbar_location='right', tools=tools)
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.visible = False
    p.yaxis.visible = False

    # Add patch renderer to figure
    p.patches('xs', 'ys', source=geosource, fill_alpha=1, line_width=0.5, line_color='black',
              fill_color={'field': column, 'transform': color_mapper})
    return p

def plot_bar_chart(x, y):
    trace = go.Bar(x=x, y=y)
    layout = go.Layout(
        autosize=True,
        hovermode='closest',
        xaxis=dict(tickangle=45),
        yaxis=dict(title="GDP"),
        showlegend=False,
        margin=dict(l=10, r=10, b=10, t=60)
    )
    fig = go.Figure(data=[trace], layout=layout)

    return fig

def plot_time_series(china_gdp, x, y_lst):
    traces = []
    for y in y_lst:
        gdps = china_gdp.loc[china_gdp["ID"] == y, x].values.ravel()
        traces.append(go.Scatter(x=x, y=gdps, name=y))
    layout = go.Layout(
        autosize=True,
        hovermode='closest',
        yaxis=dict(title="GDP"),
        showlegend=True,
        margin=dict(l=10, r=10, b=10, t=60)
    )
    fig = go.Figure(data=traces, layout=layout)
    fig.update_layout(coloraxis={'colorscale': 'viridis'})

    return fig

def plot_moranI(x, y):
    xn = np.linspace(x.min(), x.max(), 100)
    slope, intercept, _, _, _ = stats.linregress(x, y)
    yn = slope * xn + intercept
    trace1 = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(size=8,
                    color="blue"),
    )
    trace2 = go.Scatter(
        x=xn,
        y=yn,
        mode='lines',
        marker=dict(size=8,
                    color="red"),
    )
    layout = go.Layout(
        autosize=True,
        hovermode='closest',
        xaxis=dict(title="GDP_std"),
        yaxis=dict(title="Lag_std"),
        showlegend=False,
        margin=dict(l=10, r=10, b=10, t=60)
    )
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return fig

def main():
    st.title("Chinese GDP 2000 to 2016")
    st.text("Execept Hong Kong, Macau and Taiwanfrom")
    st.header("Spatial/Time Distribution")
    china_gdp = load_data()
    year = st.slider("Year: ", value=int(2000), min_value=int(2000), max_value=int(2016))
    if st.checkbox("Spatial Distribution by Year"):
        st.bokeh_chart(bokeh_plot_map(china_gdp, str(year)))
        x = china_gdp["ID"]
        y = china_gdp[str(year)]
        st.plotly_chart(plot_bar_chart(x, y))

    if st.checkbox("Time Trend by District"):
        district_selected = st.multiselect("District:", china_gdp["ID"])
        years = np.arange(2000, 2017).astype(str)
        # gdps = china_gdp.loc[china_gdp["ID"] == district_selected[0], years].values.ravel()
        st.plotly_chart(plot_time_series(china_gdp, years, district_selected))

    st.header("Global Spatial Autocorrelation")
    # create spatial weights
    w = weights.KNN.from_dataframe(china_gdp, k=8)
    w.transform = 'R'
    china_gdp[str(year) + "_lag"] = weights.spatial_lag.lag_spatial(w, china_gdp["2000"])
    china_gdp[str(year) + "_std"] = (china_gdp[str(year)] - china_gdp[str(year)].mean()) \
                                        / china_gdp[str(year)].std()
    china_gdp[str(year) + "_lag_std"] = (china_gdp[str(year) + "_lag"] - china_gdp[str(year) + "_lag"].mean()) \
                                        / china_gdp[str(year) + "_lag"].std()

    if st.checkbox("Spatial Lag"):
        st.bokeh_chart(bokeh_plot_map(china_gdp, str(year) + "_lag"))

    if st.checkbox("Binary Classification"):
        gdp_threshold = float(st.text_input("larger than:", 8000))
        gdp_min = china_gdp[str(year)].min()
        gdp_max = china_gdp[str(year)].max()
        if gdp_threshold > gdp_min and gdp_threshold < gdp_max:
            china_gdp["binary"] = np.where(china_gdp[str(year)] > gdp_threshold, 1, 0)
            st.bokeh_chart(bokeh_plot_map(china_gdp, "binary"))
        else:
            st.info("Please input a number between {}  and {}".format(gdp_min, gdp_max))

    # china_gdp[str(year) + "_std"] = (china_gdp[str(year)] - china_gdp[str(year)].mean()) \
    #                                 / china_gdp[str(year)].std()
    # china_gdp[str(year) + "_lag_std"] = (china_gdp[str(year) + "_lag"] - china_gdp[str(year) + "_lag"].mean()) \
    #                                     / china_gdp[str(year) + "_lag"].std()

    # Moran Plot"
    if st.checkbox("Moran Plot"):
        st.write(plot_moranI(china_gdp[str(year) + "_std"], china_gdp[str(year) + "_lag_std"]))

        # Moran's I"
        w.transform = 'R'
        moran = esda.moran.Moran(china_gdp[str(year)], w)
        st.text("Moran's I: {}".format(moran.I))
        plot_moran(moran)
        st.pyplot()
        # st.write(plot_moran(moran))

    # local spatial autocorrelation
    st.header("Local Spatial Autocorrelation")
    lisa = esda.moran.Moran_Local(china_gdp[str(year)], w)
    if st.checkbox("LISA"):
        china_gdp['Is'] = lisa.Is
        st.bokeh_chart(bokeh_plot_map(china_gdp, 'Is'))

    if st.checkbox("The Location of the LISA "):
        q_labels = ['Q1', 'Q2', 'Q3', 'Q4']
        labels = [q_labels[i - 1] for i in lisa.q]
        china_gdp["cl"] = labels
        palette = ['red', 'lightblue', 'blue', 'pink']
        st.bokeh_chart(plot_moran_map(china_gdp, "cl", palette))

    if st.checkbox("The significant observations "):
        sig = 1 * (lisa.p_sim < 0.05)
        labels = ['non-sig.', 'significant']
        labels = [labels[i] for i in sig]
        china_gdp["cl"] = labels
        palette = ['white', 'black']
        st.bokeh_chart(plot_moran_map(china_gdp, "cl", palette))

    if st.checkbox("LISA statistics"):
        sig = 1 * (lisa.p_sim < 0.05)
        hotspot = 1 * (sig * lisa.q == 1)
        coldspot = 3 * (sig * lisa.q == 3)
        doughnut = 2 * (sig * lisa.q == 2)
        diamond = 4 * (sig * lisa.q == 4)
        spots = hotspot + coldspot + doughnut + diamond
        spot_labels = ['0 ns', '1 hot spot', '2 doughnut', '3 cold spot', '4 diamond']
        labels = [spot_labels[i] for i in spots]
        china_gdp["cl"] = labels
        palette = ['grey', 'red', 'lightblue', 'blue', 'pink']
        st.bokeh_chart(plot_moran_map(china_gdp, "cl", palette))

if __name__ == '__main__':
    main()

