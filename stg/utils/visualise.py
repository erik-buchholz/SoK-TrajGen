#!/usr/bin/env python3
"""Contains visualisation methods."""
import logging
from typing import List, Union, Optional, Tuple

import folium
import folium.plugins
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt, axes
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch.utils.tensorboard import SummaryWriter
import io
import plotly.graph_objects as go

from stg.utils.colors import bar_colors

log = logging.getLogger()
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_bbox(df: pd.DataFrame, lat: str = 'lat', lon: str = 'lon'):
    """Return a bounding box."""
    southwest = [df[lat].min(), df[lon].min()]
    northeast = [df[lat].max(), df[lon].max()]
    bbox = [*southwest, *northeast]
    return bbox


def plot_trajectory(t: pd.DataFrame or np.ndarray,
                    labels: List[str] = None,
                    bbox: (float, float, float, float) = None,
                    title: str = "Trajectory",
                    xlabel: str = "Longitude",
                    ylabel: str = "Latitude",
                    fig: Figure = None,
                    ax: Axes = None,
                    ) -> (Figure, Axes):
    return plot_trajectories(
        (t,), labels=labels, bbox=bbox, title=title, xlabel=xlabel, ylabel=ylabel, fig=fig, ax=ax)


def plot_trajectories(ts: List[pd.DataFrame] or List[pd.DataFrame],
                      labels: List[str] = None,
                      bbox: (float, float, float, float) = None,
                      title: str = "Trajectory",
                      xlabel: str = "Longitude",
                      ylabel: str = "Latitude",
                      fig: Figure = None,
                      ax: Axes = None,
                      ) -> (Figure, Axes):
    """
    Display the given trajectories within a tight bounding box.
    :param ts: List of min. one trajectory OR one trajectory
    :param labels: Labels of the curves
    :param bbox: Bounding box
    :param title: Title of the plot
    :param xlabel: Label of the x-axis
    :param ylabel: Label of the y-axis
    :param fig: Figure to plot on
    :param ax: Axes to plot on
    :return: None
    """
    if type(ts[0]) is pd.DataFrame:
        ts = [t[['lon', 'lat']].to_numpy() for t in ts]
    if ax is None:
        fig, ax = plt.subplots()
    lines = []
    for i, t in enumerate(ts):
        line, = ax.plot(t[:, 0], t[:, 1], 'o-', linewidth=0.5)
        lines.append(line)
    if labels is not None:
        labels = [labels, ] if type(labels) is str else labels
        ax.legend(lines, labels)
    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def plot_traj_folium(df: List[pd.DataFrame] or pd.DataFrame) -> folium.Map:
    """Plot trajectories with folium"""
    if type(df) is list:
        ts = pd.concat(df)
    else:
        ts = df
        df = [df, ]
    bbox = get_bbox(ts)
    lat_mean = ts.lat.mean()
    lon_mean = ts.lon.mean()
    m = folium.Map(location=[lat_mean, lon_mean], control_scale=True, zoom_control=False)
    m.fit_bounds(bbox, padding=(0, 0))
    for i, t in enumerate(df):
        latlon = t[['lat', 'lon']].to_numpy()
        folium.PolyLine(latlon, color=bar_colors[i], weight=5, opacity=0.8).add_to(m)
        for j, p in enumerate(latlon):
            folium.Marker(p, popup=f"Stop {j + 1}").add_to(m)
    return m


def heatmap(df: pd.DataFrame, lon='lon', lat='lat', gridsize=500, bbox: (float, float, float, float) = None) -> (
        Figure, Axes):
    """
    Plot a heatmap of the entire dataset.
    :return: None
    """
    fig: Figure
    ax: axes.Axes
    fig, ax = plt.subplots()
    ax.set_title("Heatmap")

    cax = ax.hexbin(
        x=df[lon],
        y=df[lat],
        extent=bbox,
        gridsize=gridsize,
        cmap='hot',
        bins='log',
        zorder=1,
    )
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_facecolor('black')
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    fig.colorbar(cax, )
    return fig, ax


def folium_heatmap(df: pd.DataFrame, lon='lon', lat='lat', radius=5, blur=3) -> folium.Map:
    """
    Plot a heatmap of the entire dataset with folium.
    :return: None
    """
    m = folium.Map(location=[df[lat].mean(), df[lon].mean()], control_scale=True, zoom_control=False)
    bbox = get_bbox(df, lat, lon)
    m.fit_bounds(bbox, padding=(0, 0))
    folium.plugins.HeatMap(df[[lat, lon]],
                           radius=radius, blur=blur,
                           overlay=False
                           ).add_to(m)
    return m


def heatmap_to_tb(writer: SummaryWriter, prediction: pd.DataFrame, epoch: int, bbox=(-74.3, -73.7, 40.5, 41)) -> None:
    fig, ax = heatmap(prediction, bbox=bbox)
    ax.set_title(f"{epoch} Epochs")
    writer.add_figure('Heatmap', fig, epoch)


def example_to_tb(writer: SummaryWriter, original: pd.DataFrame, prediction: pd.DataFrame, epoch: int) -> None:
    fig, ax = plot_trajectories([original, prediction], [f"Original", 'Noise-only GAN'])
    ax.set_title(f"{epoch} Epochs")
    writer.add_figure('Example', fig, epoch)


def calc_zoom(min_x, max_x, min_y, max_y):
    """
    Calculate the plotly zoom level for a given bounding box.
    See https://stackoverflow.com/questions/46891914/control-mapbox-extent-in-plotly-python-api
    :param min_x: Minimum longitude
    :param max_x: Maximum longitude
    :param min_y: Minimum latitude
    :param max_y: Maximum latitude
    :return:
    """
    width_y = max_y - min_y
    width_x = max_x - min_x
    zoom_y = -1.446 * np.log(width_y) + 7.2753
    zoom_x = -1.415 * np.log(width_x) + 8.7068
    return min(round(zoom_y, 2), round(zoom_x, 2))


def plotly_fig_to_np(fig: go.Figure) -> np.ndarray:
    """
    Convert a plotly figure to a numpy array for tensorboard logging.
    :param fig: Plotly figure
    :return: np.ndarray
    """
    # Convert the Figure object to an image
    image_bytes = fig.to_image(format="png")
    image = Image.open(io.BytesIO(image_bytes))
    # Convert PIL image to NumPy array
    image_np = np.array(image)
    image_np = np.transpose(image_np, (2, 0, 1))
    return image_np


def plot_on_osm(trajectory: np.ndarray or pd.DataFrame,
                columns: List[str or int] = [0, 1],
                bbox: List[float] = None,
                reference_point: (float, float) = None,
                points: bool = False,
                as_numpy: bool = True,
                ):
    """
    Plot a trajectory on an OpenStreetMap background.
    :param trajectory: Trajectory to plot
    :param columns: Indices of trajectory containing longitude and latitude IN ORDER
    :param bbox: Bounding box of the map (min_lon, max_lon, min_lat, max_lat)
    :param reference_point: Reference point for the map center (lon, lat)
    :param points: Whether to plot points instead of a line
    :param as_numpy: Whether to return a numpy array or a plotly figure
    :return:
    """
    # If the input is a dataframe, extract the columns
    if isinstance(trajectory, pd.DataFrame):
        trajectory = trajectory[columns].values
        columns = [0, 1]
    # Extract lon and lat
    lon, lat = trajectory[:, columns[0]], trajectory[:, columns[1]]

    # Create the trace for the route
    trace = go.Scattermapbox(
        mode='markers+lines' if not points else 'markers',
        lon=lon,
        lat=lat,
        marker={'size': 4, 'color': 'black'},
        line=dict(width=3),
    )

    if reference_point is None:
        reference_point = (np.mean(lon), np.mean(lat))

    if bbox is None:
        zoom = 15
    else:
        zoom = calc_zoom(*bbox)

    # Create the layout with an OpenStreetMap background
    layout = go.Layout(
        mapbox=dict(
            center=dict(lon=reference_point[0], lat=reference_point[1]),
            style='open-street-map',
            zoom=zoom
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )

    # Create the figure
    fig = go.Figure(data=[trace], layout=layout)

    # Convert the figure to an image
    if as_numpy:
        fig = plotly_fig_to_np(fig)

    return fig


def plot_pointclouds(points: Union[List[np.ndarray], np.ndarray],
                     title: Optional[str] = None,
                     labels: Optional[List[str]] = None,
                     fig: Optional[Figure] = None,
                     ax: Optional[Axes] = None,
                     bbox: Optional[Tuple[float, float, float, float]] = (-1.0, 1.0, -1.0, 1.0),
                     xlabel: Optional[str] = 'Longitude',
                     ylabel: Optional[str] = 'Latitude',
                     use_grid: Optional[bool] = True,
                     color_id: Optional[int] = None,
                     color: Optional[str] = None,
                     ) -> Tuple[Figure, Axes]:
    """
    Plot multiple point clouds on a single plot with optional labels and bounding box.

    :param points: A single point cloud as a numpy array or a list of numpy arrays representing multiple point clouds.
        Expected shape per point cloud is (N, 2) where N is the number of points.
    :param title: The title of the plot. Default is None.
    :param labels: A list of strings representing the labels for each point cloud. Default is None.
    :param fig: An existing matplotlib Figure object to plot on. If None, a new figure is created. Default is None.
    :param ax: An existing matplotlib Axes object to plot on. If None, a new axes is created. Default is None.
    :param bbox: A tuple of four floats representing the bounding box limits in the format (xmin, xmax, ymin, ymax). Default is None.

    :return: A tuple containing the Figure and Axes objects for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(points, np.ndarray):
        points = [points]

    if labels is not None and len(labels) != len(points):
        raise ValueError("The number of labels must match the number of point clouds.")

    # Use matplotlib standard color palette
    bar_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Use rainbow color palette
    # bar_colors = plt.cm.rainbow(np.linspace(0, 1, max(len(points), 10)))

    # Determine a point size that is appropriate for the number of points in the point clouds
    point_size = 1 if len(points[0]) < 10000 else 0.1

    # Plot all point clouds with corresponding labels
    for i, p in enumerate(points):
        if p.shape[1] < 2:
            raise ValueError(f"Point cloud {i} does not have enough dimensions.")
        if color_id is not None:
            c = bar_colors[color_id]
        elif color is not None:
            # convert string to color
            c = color
        else:
            c = bar_colors[i]
        ax.scatter(p[:, 0], p[:, 1], s=point_size, alpha=1.0, c=[c])

    if labels is not None:
        # Use a large dot for the color displayed in the legend
        ax.legend(labels, scatterpoints=1, markerscale=8)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(use_grid)
    if title is not None:
        ax.set_title(title)

    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])

    return fig, ax
