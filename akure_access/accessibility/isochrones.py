"""
isochrones.py

Computes network-based travel-time isochrones from facility point
locations (health facilities, schools), and finds nearest-facility
travel times for arbitrary settlement/grid-cell points.
"""

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union


def nearest_graph_node(G: nx.MultiDiGraph, point: Point):
    """
    Find the nearest graph node to a given point, using each node's 'x'
    and 'y' attributes (in the graph's CRS).

    Uses a KD-tree (scipy.spatial.cKDTree) for an O(log n) lookup rather
    than a linear O(n) scan. The tree is built once per graph and cached
    on the graph object itself (G.graph["_kdtree"]), since this function
    is typically called many times per graph, once per origin point
    and once per candidate facility, for every settled grid cell and
    every mode, so rebuilding the tree on every call would dominate
    runtime on real road networks with thousands of nodes.

    This replaces osmnx.distance.nearest_nodes(), which assumes OSMnx's
    own internal graph conventions (e.g. integer node IDs) and does not
    work reliably against the geometry-fallback graph produced by
    network_graph.graph_from_roads() when no boundary polygon is given.
    A KD-tree over explicit 'x'/'y' node attributes works uniformly
    against both graph construction paths.
    """
    # Defensive backstop: a Polygon/MultiPolygon facility geometry
    # (e.g. a hospital mapped as a building outline rather than a
    # node) has no .x/.y and would otherwise raise here, this used to
    # fail silently further up the call chain in
    # batch_nearest_facility_distances()'s bare except, dropping the
    # facility from routing entirely with no visible error (see
    # clean.py's POINT_LAYERS collapse, which is the primary fix,
    # this is a second line of defense for anything that reaches this
    # function without having gone through that cleaning step).
    if not isinstance(point, Point):
        point = point.centroid

    cache_key = "_kdtree"
    ids_key = "_kdtree_node_ids"

    if cache_key not in G.graph or ids_key not in G.graph:
        node_ids = list(G.nodes())
        if not node_ids:
            raise ValueError("Graph has no nodes.")

        xs = np.array([G.nodes[n].get("x") for n in node_ids], dtype=float)
        ys = np.array([G.nodes[n].get("y") for n in node_ids], dtype=float)

        if np.isnan(xs).any() or np.isnan(ys).any():
            raise ValueError(
                "Graph contains nodes missing 'x'/'y' attributes; cannot "
                "perform nearest-node search."
            )

        G.graph[cache_key] = cKDTree(np.column_stack([xs, ys]))
        G.graph[ids_key] = node_ids

    tree = G.graph[cache_key]
    node_ids = G.graph[ids_key]

    _, nearest_idx = tree.query([point.x, point.y])
    return node_ids[int(nearest_idx)]


def compute_isochrone_polygon(
    G: nx.MultiDiGraph, origin_point: Point, trip_time_min: float, weight: str = "travel_time_min"
):
    """
    Compute an isochrone polygon: the area reachable from origin_point
    within trip_time_min, based on network travel time.

    Used to precompute facility catchment areas (see
    build_isochrones_for_facilities()) for the dashboard's optional
    "walking catchment" overlay, see notebooks/03_accessibility_analysis.ipynb,
    Section 5.3, which exports these to
    data/processed/{lga}/isochrones_health_walk.geojson for
    dashboard/app.py to load and display, so the dashboard itself never
    needs to build a live routable graph at request time.

    Note the approximation this makes: the returned polygon is a
    convex hull over reachable graph nodes, not the true reachable
    street-network footprint, convex hulls can overstate actual
    reachable area, since real street networks are rarely convex (e.g.
    a river or a gap in the road network can make an area within the
    hull genuinely unreachable). This is a deliberate, documented
    tradeoff for a fast, simple approximation suitable for an
    illustrative dashboard overlay, the project's actual
    access-deficit scoring does NOT use this approximation; it uses
    exact network shortest-path distances/times via
    nearest_facility_distance_and_time() / batch_nearest_facility_distances().

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Routable graph with a `weight` edge attribute representing
        travel time in minutes (see network_graph.graph_from_roads()).
    origin_point : shapely.geometry.Point
        Facility location, in the same CRS as the graph.
    trip_time_min : float
        Time budget in minutes.
    weight : str
        Edge attribute name to use as the travel-time weight.

    Returns
    -------
    shapely.geometry.Polygon or None
        Convex hull of nodes reachable within the time budget, as a
        simple, fast isochrone approximation. Returns None if the
        origin node cannot be matched to the graph.
    """
    try:
        center_node = nearest_graph_node(G, origin_point)
    except Exception:
        return None

    subgraph_nodes = nx.ego_graph(G, center_node, radius=trip_time_min, distance=weight)
    if len(subgraph_nodes) < 3:
        return None

    points = [Point((data["x"], data["y"])) for _, data in subgraph_nodes.nodes(data=True)]
    return MultiPoint(points).convex_hull


def build_isochrones_for_facilities(
    G: nx.MultiDiGraph, facilities_gdf: gpd.GeoDataFrame, trip_times_min=(15, 30, 45)
) -> gpd.GeoDataFrame:
    """
    Compute isochrone polygons for every facility in facilities_gdf, at
    each requested trip time.

    Used by notebooks/03_accessibility_analysis.ipynb (Section 5.3) to
    precompute health-facility walking catchments, exported to
    data/processed/{lga}/isochrones_health_walk.geojson for
    dashboard/app.py's optional catchment overlay. See the NOTE in
    compute_isochrone_polygon() above (which this function wraps) for
    the convex-hull approximation this makes and why it's acceptable
    for this illustrative overlay but not used for the project's actual
    access-deficit scoring.

    Parameters
    ----------
    G : networkx.MultiDiGraph
    facilities_gdf : geopandas.GeoDataFrame
        Point layer of facilities (health facilities or schools),
        cleaned/exported by lga_extractor, reprojected to match G's CRS.
    trip_times_min : tuple of float
        Trip time bands to compute (minutes).

    Returns
    -------
    geopandas.GeoDataFrame
        One row per (facility, trip_time) combination, with columns
        [facility_name, osmid, trip_time_min, geometry].
    """
    records = []
    for _, row in facilities_gdf.iterrows():
        for t in trip_times_min:
            poly = compute_isochrone_polygon(G, row.geometry, t)
            if poly is not None:
                records.append(
                    {
                        "facility_name": row.get("name"),
                        "osmid": row.get("osmid"),
                        "trip_time_min": t,
                        "geometry": poly,
                    }
                )

    crs = facilities_gdf.crs
    if not records:
        # gpd.GeoDataFrame([], crs=...) cannot infer a geometry column
        # from an empty list of records, and raises if a CRS is also
        # given, construct the empty result explicitly instead, so
        # this case (an empty facilities_gdf, or a facilities_gdf whose
        # points all failed to match the graph) behaves the same way
        # the rest of this codebase represents "empty but valid":
        # a GeoDataFrame with a real geometry column and the correct
        # CRS, just zero rows.
        return gpd.GeoDataFrame(
            columns=["facility_name", "osmid", "trip_time_min", "geometry"],
            geometry="geometry",
            crs=crs,
        )

    return gpd.GeoDataFrame(records, crs=crs)


def nearest_facility_travel_time(
    G: nx.MultiDiGraph, origin_point: Point, facilities_gdf: gpd.GeoDataFrame, weight: str = "travel_time_min"
) -> float:
    """
    Compute the network travel time (minutes) from origin_point to the
    single nearest facility in facilities_gdf.

    Kept as a thin wrapper around nearest_facility_distance_and_time()
    for backward compatibility with existing calling code.

    Parameters
    ----------
    G : networkx.MultiDiGraph
    origin_point : shapely.geometry.Point
        Settlement/grid-cell centroid, in the graph's CRS.
    facilities_gdf : geopandas.GeoDataFrame
        Point layer of facilities to route to.
    weight : str
        Edge attribute to use as travel-time weight.

    Returns
    -------
    float
        Travel time in minutes to the nearest reachable facility, or
        float('inf') if no facility is reachable from the origin.
    """
    _, time_min = nearest_facility_distance_and_time(G, origin_point, facilities_gdf, weight=weight)
    return time_min


def nearest_facility_distance_and_time(
    G: nx.MultiDiGraph,
    origin_point: Point,
    facilities_gdf: gpd.GeoDataFrame,
    weight: str = "travel_time_min",
    distance_attr: str = "length",
):
    """
    Compute both the network distance (km) and travel time (minutes)
    from origin_point to the single nearest facility in facilities_gdf,
    where "nearest" is determined by `weight` (typically travel time,
    so the result reflects the fastest facility to reach, not
    necessarily the closest in metres).

    NOTE: this runs one shortest-path search per facility, for a single
    origin. If you need this for MANY origins against the SAME set of
    facilities (e.g. scoring every grid cell in an LGA), use
    batch_nearest_facility_distances() + lookup_nearest_distance_time()
    instead, that computes the routing once via multi-source Dijkstra
    rather than once per origin, and is dramatically faster at scale
    (this is what add_access_times() in scoring.py uses). This function
    is kept for single-lookup use cases and backward compatibility.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Routable graph for a single mode, as built by
        network_graph.graph_from_roads(..., mode=...). The graph's
        'mode' and 'speed_kph' attributes (set by graph_from_roads)
        describe which mode this result applies to.
    origin_point : shapely.geometry.Point
        Settlement/grid-cell centroid, in the graph's CRS.
    facilities_gdf : geopandas.GeoDataFrame
        Point layer of facilities to route to.
    weight : str
        Edge attribute used to select the "nearest" facility and to
        report travel time.
    distance_attr : str
        Edge attribute used to report distance (metres), summed along
        the same shortest path used for the time calculation.

    Returns
    -------
    (float, float)
        (distance_km, travel_time_min). Both are float('inf') if no
        facility is reachable from the origin.
    """
    if facilities_gdf.empty:
        return float("inf"), float("inf")

    try:
        origin_node = nearest_graph_node(G, origin_point)
    except Exception:
        return float("inf"), float("inf")

    best_time = float("inf")
    best_distance_km = float("inf")

    for _, row in facilities_gdf.iterrows():
        try:
            dest_node = nearest_graph_node(G, row.geometry)
            path = nx.shortest_path(G, origin_node, dest_node, weight=weight)
            travel_time = nx.path_weight(G, path, weight=weight)
            distance_m = nx.path_weight(G, path, weight=distance_attr)
        except Exception:
            continue

        if travel_time < best_time:
            best_time = travel_time
            best_distance_km = distance_m / 1000

    return best_distance_km, best_time


def batch_nearest_facility_distances(
    G: nx.MultiDiGraph, facilities_gdf: gpd.GeoDataFrame, distance_attr: str = "length"
) -> dict:
    """
    Compute, for every node in G, the network distance (metres) to the
    nearest facility in facilities_gdf, in a single multi-source
    Dijkstra pass, rather than one shortest-path search per
    (origin, facility) pair.

    This is the key performance fix for scoring a full LGA: the naive
    approach (nearest_facility_distance_and_time, called once per grid
    cell) does `cells x facilities` separate shortest-path searches.
    For a real LGA this is `hundreds x dozens` = potentially tens of
    thousands of Dijkstra runs across a road network with thousands of
    nodes, per mode, per service, this is what made a full Notebook 03
    run take over an hour in practice.

    Multi-source Dijkstra instead finds the shortest distance from the
    NEAREST of several source nodes to every other node, in one pass
    (networkx.multi_source_dijkstra_path_length). Snapping each facility
    to its nearest graph node is still one KD-tree lookup per facility
    (fast, see nearest_graph_node), but the expensive routing computation
    itself now happens once per (mode, service) combination rather than
    once per grid cell.

    Distance (not travel time) is used as the Dijkstra weight
    deliberately: within a single mode's graph, every edge shares the
    same assumed speed (see network_graph.MODE_CONFIG), so travel time
    is simply distance divided by a constant, meaning the shortest
    path by distance and the shortest path by travel time are identical
    for this graph structure, and time can be derived from distance
    afterward (see lookup_nearest_distance_time) without needing a
    second, separately-weighted Dijkstra pass.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Routable graph for a single mode, as built by
        network_graph.graph_from_roads(...).
    facilities_gdf : geopandas.GeoDataFrame
        Point layer of facilities to route to (health facilities or
        schools).
    distance_attr : str
        Edge attribute to use as the Dijkstra weight (metres).

    Returns
    -------
    dict
        Mapping of graph_node -> distance in metres to the nearest
        facility. Nodes absent from this dict are unreachable from
        every facility (equivalent to an infinite distance).
    """
    if facilities_gdf.empty:
        return {}

    facility_nodes = set()
    skipped = 0
    for _, row in facilities_gdf.iterrows():
        try:
            facility_nodes.add(nearest_graph_node(G, row.geometry))
        except Exception:
            skipped += 1
            continue

    # Surface this loudly rather than letting it fail silently: if
    # every facility in a non-empty facilities_gdf fails to snap, the
    # caller ends up with every grid cell scored as unreachable
    # (inf) for this service, with no visible sign anything went
    # wrong. This is exactly what happened when a whole LGA's health
    # facilities were mapped as building-outline polygons upstream, a
    # bare `except: continue` here silently produced 0% health
    # access for an entire LGA. A partial skip (some facilities
    # genuinely un-snappable, e.g. disconnected from the graph) is
    # expected occasionally and only warns; a total skip is almost
    # always a real bug and should be investigated immediately.
    if skipped:
        import warnings
        warnings.warn(
            f"{skipped}/{len(facilities_gdf)} facilities in this layer could not "
            f"be matched to the routing graph and were excluded from nearest-"
            f"facility distances. If this is all or most of the layer, check "
            f"geometry type (should be Point after cleaning) and CRS before "
            f"trusting the resulting scores."
        )

    if not facility_nodes:
        return {}

    return nx.multi_source_dijkstra_path_length(G, sources=facility_nodes, weight=distance_attr)


def lookup_nearest_distance_time(
    G: nx.MultiDiGraph, origin_point: Point, distances_by_node: dict
):
    """
    Look up the precomputed nearest-facility distance/time for a single
    origin point, given the output of batch_nearest_facility_distances().

    This is an O(1) dictionary lookup (aside from snapping origin_point
    to its nearest graph node via KD-tree), intended to be called once
    per grid cell after batch_nearest_facility_distances() has already
    done the expensive routing work once for the whole graph.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        The same graph passed to batch_nearest_facility_distances(). Its
        'speed_kph' graph attribute (set by graph_from_roads()) is used
        to convert distance into travel time.
    origin_point : shapely.geometry.Point
        Settlement/grid-cell centroid, in the graph's CRS.
    distances_by_node : dict
        Output of batch_nearest_facility_distances(G, facilities_gdf).

    Returns
    -------
    (float, float)
        (distance_km, travel_time_min). Both are float('inf') if the
        origin cannot be matched to the graph, or is unreachable from
        every facility.
    """
    if not distances_by_node:
        return float("inf"), float("inf")

    try:
        origin_node = nearest_graph_node(G, origin_point)
    except Exception:
        return float("inf"), float("inf")

    distance_m = distances_by_node.get(origin_node, float("inf"))
    if distance_m == float("inf"):
        return float("inf"), float("inf")

    speed_kph = G.graph.get("speed_kph")
    if not speed_kph:
        return float("inf"), float("inf")

    speed_m_per_min = (speed_kph * 1000) / 60
    time_min = distance_m / speed_m_per_min if speed_m_per_min else float("inf")

    return distance_m / 1000, time_min
