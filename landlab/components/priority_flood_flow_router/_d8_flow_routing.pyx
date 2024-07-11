cimport cython
from libc.math cimport sqrt
from libc.stdint cimport int8_t

from cython.parallel import prange

ctypedef fused id_t:
    cython.integral
    long long


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _find_steepest_d8_neighbor_at_nodes(
    shape,
    xy_spacing,
    const id_t [:] nodes,
    const int8_t [:] is_active_node,
    const cython.floating [:] z_at_node,
    id_t [:] d8_receiver_at_node,
    cython.floating [:] steepest_slope_at_node,
):
    cdef long i
    cdef long offset
    cdef long node
    cdef long n
    cdef long max_n
    cdef double slope
    cdef double max_slope
    cdef double dx = xy_spacing[0]
    cdef double dy = xy_spacing[1]
    cdef long n_cols = shape[1]
    cdef long n_nodes = nodes.shape[0]
    cdef long[8] neighbor = [
        1, n_cols, -1, -n_cols, n_cols + 1, n_cols - 1, -n_cols - 1, -n_cols + 1
    ]
    cdef double diagonal_distance = sqrt(dx*dx + dy*dy)
    cdef double[8] one_over_distance_to_neighbor = [
        1.0 / dx,
        1.0 / dy,
        1.0 / dx,
        1.0 / dy,
        1.0 / diagonal_distance,
        1.0 / diagonal_distance,
        1.0 / diagonal_distance,
        1.0 / diagonal_distance,
    ]
    cdef long n_neighbors = 8

    for i in prange(n_nodes, nogil=True, schedule="static"):
        node = nodes[i]

        max_slope = 0.0
        max_n = -1
        for n in range(n_neighbors):
            offset = neighbor[n]
            if is_active_node[node + offset]:
                slope = (
                    z_at_node[node] - z_at_node[node + offset]
                ) * one_over_distance_to_neighbor[n]

                if slope > max_slope:
                    max_slope = slope
                    max_n = n

        d8_receiver_at_node[node] = max_n
        steepest_slope_at_node[node] = max_slope


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _find_d8_node(
    shape,
    const id_t [:] nodes,
    const id_t [:] d8_neighbor_at_node,
    id_t [:] d8_node_at_node,
):
    cdef long i
    cdef long node
    cdef long d8
    # cdef long n_rows = shape[0]
    cdef long n_cols = shape[1]
    # cdef long n_nodes = n_rows * n_cols
    cdef long[8] neighbor = [
        1, n_cols, -1, -n_cols, n_cols + 1, n_cols - 1, -n_cols - 1, -n_cols + 1
    ]
    cdef long n_nodes = len(nodes)

    # for node in prange(n_nodes, nogil=True, schedule="static"):
    for i in prange(n_nodes, nogil=True, schedule="static"):
        node = nodes[i]
        d8 = d8_neighbor_at_node[node]

        if d8 == -1:
            d8_node_at_node[node] = node
        else:
            d8_node_at_node[node] = node + neighbor[d8]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _find_receiver_d8_link(
    const id_t [:] nodes,
    const id_t [:] d8_neighbor_at_node,
    const id_t [:, :] d8_links_at_node,
    id_t [:] d8_link_at_node,
):
    cdef long i
    cdef long node
    cdef long d8
    cdef long n_nodes = nodes.shape[0]

    for i in prange(n_nodes, nogil=True, schedule="static"):
        node = nodes[i]
        d8 = d8_neighbor_at_node[node]

        if d8 == -1:
            d8_link_at_node[node] = -1
        else:
            d8_link_at_node[node] = d8_links_at_node[node, d8]


@cython.boundscheck(False)
@cython.wraparound(False)
def _calc_distance_to_d8(
    xy_spacing,
    const id_t [:] nodes,
    const id_t [:] d8_neighbor_at_node,
    cython.floating [:] d8_distance_at_node,
):
    cdef long i
    cdef long node
    cdef long d8
    cdef long n_nodes = nodes.shape[0]
    cdef double dx = xy_spacing[0]
    cdef double dy = xy_spacing[1]
    cdef double diagonal_distance = sqrt(dx*dx + dy*dy)
    cdef double[8] distance_to_neighbor = [
        dx,
        dy,
        dx,
        dy,
        diagonal_distance,
        diagonal_distance,
        diagonal_distance,
        diagonal_distance,
    ]

    for i in prange(n_nodes, nogil=True, schedule="static"):
        node = nodes[i]
        d8 = d8_neighbor_at_node[node]

        if d8 == -1:
            d8_distance_at_node[node] = 0.0
        else:
            d8_distance_at_node[node] = distance_to_neighbor[d8]


@cython.boundscheck(False)
@cython.wraparound(False)
def _calc_slope_to_d8(
    shape,
    xy_spacing,
    const id_t [:] nodes,
    const cython.floating [:] z_at_node,
    const id_t [:] d8_neighbor_at_node,
    cython.floating [:] d8_slope_at_node,
):
    cdef long n_cols = shape[1]
    cdef long i
    cdef long d8
    cdef long node
    cdef long n_nodes = nodes.shape[0]
    cdef long[8] neighbor = [
        1, n_cols, -1, -n_cols, n_cols + 1, n_cols - 1, -n_cols - 1, -n_cols + 1
    ]
    cdef double dx = xy_spacing[0]
    cdef double dy = xy_spacing[1]
    cdef double diagonal_distance = sqrt(dx*dx + dy*dy)
    cdef double[8] one_over_distance_to_d8 = [
        1.0 / dx,
        1.0 / dy,
        1.0 / dx,
        1.0 / dy,
        1.0 / diagonal_distance,
        1.0 / diagonal_distance,
        1.0 / diagonal_distance,
        1.0 / diagonal_distance,
    ]

    for i in prange(n_nodes, nogil=True, schedule="static"):
        node = nodes[i]
        d8 = d8_neighbor_at_node[node]

        if d8 >= 0:
            d8_slope_at_node[node] = (
                z_at_node[node] - z_at_node[node + neighbor[d8]]
            ) * one_over_distance_to_d8[d8]


@cython.boundscheck(False)
@cython.wraparound(False)
def _accumulate_at_receiver_node(
    const id_t [:] nodes,
    const id_t [:] receiver_node_at_node,
    const cython.floating [:] value_at_node,
    cython.floating [:] receiver_value_at_node,
):
    cdef long i
    cdef long node
    cdef long donor_node
    cdef long receiver_node
    cdef long n_nodes = len(value_at_node)
    cdef long n_donor_nodes = len(nodes)

    for node in prange(n_nodes, nogil=True, schedule="static"):
        receiver_value_at_node[node] = value_at_node[node]

    with nogil:
        for i in range(n_donor_nodes):
            donor_node = nodes[i]
            receiver_node = receiver_node_at_node[donor_node]
            if receiver_node >= 0:
                receiver_value_at_node[receiver_node] += (
                    receiver_value_at_node[donor_node]
                )
