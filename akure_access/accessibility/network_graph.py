"""
network_graph.py

Builds a routable NetworkX graph from a cleaned OSM roads layer (as
produced by the lga_extractor tool), suitable for network-distance /
travel-time analysis.
"""

import geopandas as gpd
import networkx as nx
import osmnx as ox

# Backward-compatible constant (walking speed), kept for any code that
# still imports WALKING_SPEED_KPH directly.
WALKING_SPEED_KPH = 5.0

# Mode configuration: OSM network type to query, plus an assumed
# average travel speed used to convert edge length into travel time.
#
# "okada" (commercial motorcycle taxis) is modeled on the OSM "drive"
# network (motorcycles use the same road network as cars in Nigeria,
# and OSM has no distinct motorcycle network type), but with a lower
# assumed average speed reflecting local traffic/road conditions and
# okada behavior (weaving through traffic, using narrower roads cars
# may avoid). These speed assumptions are approximations, see the
# methodology limitations section, and can be overridden by passing
# a custom `speed_kph` to graph_from_roads().
MODE_CONFIG = {
    "walk": {"network_type": "walk", "speed_kph": 5.0},
    "okada": {"network_type": "drive", "speed_kph": 25.0},
    "drive": {"network_type": "drive", "speed_kph": 35.0},
}


def graph_from_roads(
    roads_gdf: gpd.GeoDataFrame,
    boundary_polygon=None,
    mode: str = "walk",
    speed_kph: float = None,
) -> nx.MultiDiGraph:
    """
    Build a routable network graph for a given travel mode.

    Two construction paths are supported:

    1. If `boundary_polygon` is provided, the graph is built directly
       from OSM via osmnx.graph_from_polygon(), using the OSM
       network_type associated with `mode` (see MODE_CONFIG). This is
       the simplest and most robust option, and is recommended as the
       default, since OSMnx handles topology/connectivity correctly.
       `roads_gdf` is not required in this mode but may be used for
       validation/consistency checks against the extractor's own
       roads layer.

    2. If only `roads_gdf` is provided (no boundary polygon), a graph
       is constructed directly from the cleaned roads GeoDataFrame's
       geometries. This is a fallback path for cases where rebuilding
       directly from OSM is undesirable (e.g. to guarantee the graph
       matches exactly the same roads already extracted and versioned
       by the LGA extractor). Note: this fallback does not distinguish
       walk-only paths from vehicle roads, since that distinction lives
       in OSM's tag-based network typing, not in the extractor's plain
       roads layer, so `mode="walk"` and `mode="drive"` will yield the
       same underlying graph when built this way, differing only in
       assumed speed.

    Parameters
    ----------
    roads_gdf : geopandas.GeoDataFrame
        Cleaned roads layer, in EPSG:32631 (as produced by
        lga_extractor.clean.clean_layers()).
    boundary_polygon : shapely.geometry.Polygon, optional
        LGA boundary polygon in EPSG:4326, used to build the graph
        directly via OSMnx (recommended default path).
    mode : str
        One of "walk", "okada", "drive" (see MODE_CONFIG). Determines
        both the OSM network type queried (when using boundary_polygon)
        and the default assumed speed.
    speed_kph : float, optional
        Override the default speed assumption for the chosen mode.

    Returns
    -------
    networkx.MultiDiGraph
        A routable graph with 'length' (metres) and 'travel_time_min'
        (minutes) attributes on every edge, plus a graph-level
        'mode' attribute recording which mode was used to build it.
    """
    if mode not in MODE_CONFIG:
        raise ValueError(f"Unknown mode '{mode}'. Expected one of {list(MODE_CONFIG)}.")

    config = MODE_CONFIG[mode]
    effective_speed = speed_kph if speed_kph is not None else config["speed_kph"]

    if boundary_polygon is not None:
        G = ox.graph_from_polygon(boundary_polygon, network_type=config["network_type"])
        # NOTE: osmnx.graph_from_polygon() already computes edge lengths
        # internally in both 1.x and 2.x, so _has_lengths(G) normally
        # short-circuits this fallback. It's kept as a defensive fallback
        # in case that internal behavior ever changes. ox.add_edge_lengths
        # was removed from the top-level namespace in OSMnx 2.x, the
        # correct current location is ox.distance.add_edge_lengths.
        G = ox.distance.add_edge_lengths(G) if not _has_lengths(G) else G
    else:
        G = _graph_from_geometries(roads_gdf)

    _assign_travel_times(G, effective_speed)
    G.graph["mode"] = mode
    G.graph["speed_kph"] = effective_speed
    return G


def _has_lengths(G: nx.MultiDiGraph) -> bool:
    return all("length" in data for _, _, data in G.edges(data=True))


def _assign_travel_times(G: nx.MultiDiGraph, speed_kph: float) -> None:
    """Add a travel_time_min edge attribute based on the given speed."""
    speed_m_per_min = (speed_kph * 1000) / 60
    for _, _, data in G.edges(data=True):
        length_m = data.get("length", 0)
        data["travel_time_min"] = length_m / speed_m_per_min if speed_m_per_min else float("inf")


def _graph_from_geometries(roads_gdf: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    """
    Fallback: construct a simple undirected graph directly from line
    geometries in a cleaned roads GeoDataFrame, using coordinate tuples
    as node identifiers. Less robust than osmnx.graph_from_polygon (no
    topology cleanup), intended only as a consistency fallback.

    Each node is also given explicit 'x' and 'y' attributes matching its
    coordinate tuple. This is required for compatibility with
    osmnx.distance.nearest_nodes() (used by isochrones.nearest_graph_node)
    and with isochrones.compute_isochrone_polygon(), both of which read
    node['x'] / node['y'] rather than the node key itself, without this,
    nearest-node lookups against this fallback graph silently fail and
    every distance/time comes back as inf.
    """
    G = nx.MultiGraph()
    G.graph["crs"] = roads_gdf.crs.to_string() if roads_gdf.crs else "EPSG:32631"

    for _, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        for i in range(len(coords) - 1):
            u, v = coords[i], coords[i + 1]
            length = ((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2) ** 0.5
            G.add_node(u, x=u[0], y=u[1])
            G.add_node(v, x=v[0], y=v[1])
            G.add_edge(u, v, length=length, osmid=row.get("osmid"))

    return G
