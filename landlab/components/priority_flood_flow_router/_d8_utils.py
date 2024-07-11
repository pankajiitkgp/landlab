import numpy as np

from landlab.components.priority_flood_flow_router._d8_flow_routing import (
    _accumulate_at_receiver_node,
)
from landlab.components.priority_flood_flow_router._d8_flow_routing import (
    _calc_slope_to_d8,
)
from landlab.components.priority_flood_flow_router._d8_flow_routing import _find_d8_node
from landlab.components.priority_flood_flow_router._d8_flow_routing import (
    _find_receiver_d8_link,
)
from landlab.components.priority_flood_flow_router._d8_flow_routing import (
    _find_steepest_d8_neighbor_at_nodes,
)
from landlab.grid.nodestatus import NodeStatus


def calc_steepest_d8_slope(
    grid,
    z_at_node,
    nodes=None,
    is_active_node=None,
    with_neighbors=False,
):
    """Find the steepest slope to a d8 neighbor.

    Parameters
    ----------
    grid : RasterModelGrid
        A Landlab grid.
    z_at_node : array-like of float
        Array of values at grid nodes.
    nodes : array-like of int, optional
        Array of nodes for which to calculate gradients. The default
        is to operate on all *core* nodes.
    is_active_node : array-like of bool
        Array that indicates if a node is "active". Only slopes between
        *active* nodes are considered. If not provided, all nodes are
        considered *active*.
    with_neighbors : bool, optional
        If `True` return the corresponding d8 neighbor to which
        the steepest slope was found.

    Examples
    --------
    >>> from landlab import RasterModelGrid
    >>> from landlab.components.priority_flood_flow_router._d8_utils import (
    ...     calc_steepest_d8_slope,
    ... )

    >>> grid = RasterModelGrid((3, 4), xy_spacing=(4.0, 3.0))
    >>> z_at_node = [
    ...     [0.0, 1.0, 5.0, 13.0],
    ...     [0.0, 1.0, 5.0, 13.0],
    ...     [0.0, 1.0, 5.0, 13.0],
    ... ]
    >>> calc_steepest_d8_slope(grid, z_at_node).reshape(grid.shape)
    array([[0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.25, 1.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  ]])
    >>> _, d8_neighbors = calc_steepest_d8_slope(grid, z_at_node, with_neighbors=True)
    >>> d8_neighbors.reshape(grid.shape)
    array([[-1, -1, -1, -1],
           [-1,  2,  2, -1],
           [-1, -1, -1, -1]])

    >>> z_at_node = [
    ...     [0.0, 15.0, 30.0, 30.0],
    ...     [10.0, 30.0, 30.0, 30.0],
    ...     [20.0, 30.0, 30.0, 30.0],
    ... ]
    >>> calc_steepest_d8_slope(grid, z_at_node).reshape(grid.shape)
    array([[0., 0., 0., 0.],
           [0., 6., 3., 0.],
           [0., 0., 0., 0.]])
    >>> _, d8_neighbors = calc_steepest_d8_slope(grid, z_at_node, with_neighbors=True)
    >>> d8_neighbors.reshape(grid.shape)
    array([[-1, -1, -1, -1],
           [-1,  6,  6, -1],
           [-1, -1, -1, -1]])
    """
    if nodes is None:
        nodes = grid.core_nodes
    if is_active_node is None:
        is_active_node = grid.status_at_node != NodeStatus.CLOSED

    d8_receiver_at_node = np.full(grid.number_of_nodes, -1, dtype=int)
    steepest_slope = np.zeros(grid.number_of_nodes, dtype=float)

    _find_steepest_d8_neighbor_at_nodes(
        grid.shape,
        (grid.dx, grid.dy),
        np.asarray(nodes),
        is_active_node.view(np.int8),
        np.asarray(z_at_node).reshape(-1),
        d8_receiver_at_node,
        steepest_slope,
    )

    if with_neighbors:
        return steepest_slope, d8_receiver_at_node
    else:
        return steepest_slope


def calc_slope_to_d8(
    grid,
    z_at_node,
    d8_neighbor_at_node,
    nodes=None,
):
    """Calculate the slope between a node and one of its d8 neighbors.

    Examples
    --------
    >>> import numpy as np
    >>> from landlab import RasterModelGrid
    >>> from landlab.components.priority_flood_flow_router._d8_utils import (
    ...     calc_slope_to_d8,
    ... )

    >>> grid = RasterModelGrid((3, 4), xy_spacing=(4.0, 3.0))
    >>> z_at_node = [
    ...     [0.0, 1.0, 2.0, 3.0],
    ...     [0.0, 1.0, 2.0, 3.0],
    ...     [0.0, 1.0, 2.0, 3.0],
    ... ]
    >>> neighbor_at_node = np.full(grid.number_of_nodes, 1)
    >>> calc_slope_to_d8(grid, z_at_node, neighbor_at_node)
    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    >>> neighbor_at_node = np.full(grid.number_of_nodes, 0)
    >>> calc_slope_to_d8(grid, z_at_node, neighbor_at_node).reshape(grid.shape)
    array([[ 0.  ,  0.  ,  0.  ,  0.  ],
           [ 0.  , -0.25, -0.25,  0.  ],
           [ 0.  ,  0.  ,  0.  ,  0.  ]])

    >>> neighbor_at_node = np.full(grid.number_of_nodes, 4)
    >>> calc_slope_to_d8(grid, z_at_node, neighbor_at_node).reshape(grid.shape)
    array([[ 0. ,  0. ,  0. ,  0. ],
           [ 0. , -0.2, -0.2,  0. ],
           [ 0. ,  0. ,  0. ,  0. ]])

    >>> neighbor_at_node = np.full(grid.number_of_nodes, 6)
    >>> calc_slope_to_d8(grid, z_at_node, neighbor_at_node, nodes=[5]).reshape(
    ...     grid.shape
    ... )
    array([[0. , 0. , 0. , 0. ],
           [0. , 0.2, 0. , 0. ],
           [0. , 0. , 0. , 0. ]])
    """
    if nodes is None:
        nodes = grid.core_nodes

    slope_at_d8 = np.zeros(grid.number_of_nodes, dtype=float)

    _calc_slope_to_d8(
        grid.shape,
        (grid.dx, grid.dy),
        np.asarray(nodes),
        np.asarray(z_at_node).reshape(-1),
        np.asarray(d8_neighbor_at_node).reshape(-1),
        slope_at_d8,
    )

    return slope_at_d8


def find_d8_node(
    grid,
    d8_neighbor_at_node,
    nodes=None,
):
    """Find d8 neighbor nodes.

    Parameters
    ----------
    grid : RasterModelGrid
        A Landlab grid.
    d8_neighbor_at_node : array-like of int
        A d8 neighbor for each grid node.
    nodes : array-like of int, optional
        Array of nodes on which to operate. The default is to operate
        on all *core* nodes.

    Examples
    --------
    >>> from landlab import RasterModelGrid
    >>> from landlab.components.priority_flood_flow_router._d8_utils import (
    ...     find_d8_node,
    ... )

    >>> grid = RasterModelGrid((3, 4), xy_spacing=(4.0, 3.0))
    >>> d8_neighbors_at_node = [
    ...     [-1, -1, -1, -1],
    ...     [-1, 7, 3, -1],
    ...     [-1, -1, -1, -1],
    ... ]
    >>> find_d8_node(grid, d8_neighbors_at_node).reshape(grid.shape)
    array([[-1, -1, -1, -1],
           [-1,  2,  2, -1],
           [-1, -1, -1, -1]])
    """
    if nodes is None:
        nodes = grid.core_nodes

    d8_node_at_node = np.full(grid.number_of_nodes, -1, dtype=int)
    _find_d8_node(
        grid.shape,
        np.asarray(nodes),
        np.asarray(d8_neighbor_at_node).reshape(-1),
        d8_node_at_node,
    )

    return d8_node_at_node


def find_d8_link(
    grid,
    d8_neighbor_at_node,
    nodes=None,
):
    """Find links to d8 neighbors.

    Parameters
    ----------
    grid : RasterModelGrid
        A Landlab grid.
    d8_neighbor_at_node : array-like of int
        A d8 neighbor for each grid node.
    nodes : array-like of int, optional
        Array of nodes on which to operate. The default is to operate
        on all *core* nodes.

    Examples
    --------
    >>> from landlab import RasterModelGrid
    >>> from landlab.components.priority_flood_flow_router._d8_utils import (
    ...     find_d8_link,
    ... )

    >>> grid = RasterModelGrid((3, 4), xy_spacing=(4.0, 3.0))
    >>> d8_neighbors_at_node = [
    ...     [-1, -1, -1, -1],
    ...     [-1, 7, 3, -1],
    ...     [-1, -1, -1, -1],
    ... ]
    >>> find_d8_link(grid, d8_neighbors_at_node).reshape(grid.shape)
    array([[-1, -1, -1, -1],
           [-1, 20,  5, -1],
           [-1, -1, -1, -1]])
    """
    if nodes is None:
        nodes = grid.core_nodes

    d8_link_at_node = np.full(grid.number_of_nodes, -1, dtype=int)
    _find_receiver_d8_link(
        np.asarray(nodes),
        np.asarray(d8_neighbor_at_node).reshape(-1),
        grid.d8s_at_node,
        d8_link_at_node,
    )

    return d8_link_at_node


def accumulate_at_receiver_node(
    value_at_node,
    receiver_node_at_node,
    nodes=None,
    out=None,
):
    """Accumulate values on receiver nodes.

    Parameters
    ----------
    value_at_node : array-like of float
        Node values to accumulate.
    receiver_node_at_node : array-like of int
        Node receiving values from each node.
    node : array-like of int, optional
        Ordered nodes to accumulate. If not provided, the nodes
        are ordered as the grid's *core* nodes.

    Examples
    --------
    >>> from landlab import RasterModelGrid
    >>> from landlab.components.priority_flood_flow_router._d8_utils import (
    ...     accumulate_at_receiver_node,
    ... )

    >>> grid = RasterModelGrid((3, 4), xy_spacing=(4.0, 3.0))
    >>> receiver_node_at_node = [
    ...     [-1, -1, -1, -1],
    ...     [-1, 0, 5, -1],
    ...     [-1, -1, -1, 6],
    ... ]
    >>> value_at_node = grid.ones(at="node")
    >>> accumulate_at_receiver_node(
    ...     value_at_node,
    ...     receiver_node_at_node,
    ...     nodes=[11, 6, 5, 0],
    ... ).reshape(grid.shape)
    array([[4., 1., 1., 1.],
           [1., 3., 2., 1.],
           [1., 1., 1., 1.]])

    >>> accumulate_at_receiver_node(
    ...     value_at_node,
    ...     receiver_node_at_node,
    ...     nodes=range(12),
    ... ).reshape(grid.shape)
    array([[2., 1., 1., 1.],
           [1., 2., 2., 1.],
           [1., 1., 1., 1.]])

    >>> accumulate_at_receiver_node(
    ...     value_at_node,
    ...     receiver_node_at_node,
    ...     nodes=range(11, -1, -1),
    ... ).reshape(grid.shape)
    array([[4., 1., 1., 1.],
           [1., 3., 2., 1.],
           [1., 1., 1., 1.]])
    """
    if nodes is None:
        nodes = np.arange()
    if out is None:
        receiver_value_at_node = np.empty_like(value_at_node)
    else:
        receiver_value_at_node = out

    # receiver_value_at_node = np.empty_like(value_at_node)

    _accumulate_at_receiver_node(
        np.asarray(nodes),
        np.asarray(receiver_node_at_node).reshape(-1),
        value_at_node,
        receiver_value_at_node,
    )

    return receiver_value_at_node
