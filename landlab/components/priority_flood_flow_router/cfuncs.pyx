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
cpdef _route_flow_d8(
    shape,
    xy_spacing,
    id_t [:] receivers,
    cython.floating [:] distance_receiver,
    cython.floating [:] steepest_slope,
    const cython.floating [:] z_at_node,
    const cython.floating [:] z_original,
    const id_t [:] nodes,
    const int8_t [:] is_active_cell,
    const id_t [:, :] adj_link,
    id_t [:] receiver_link,
):
    """Calcualte D8 flow dirs"""
    cdef long n_cols = shape[1]
    cdef long n_nodes = len(nodes)
    cdef double dx = xy_spacing[0]
    cdef double dy = xy_spacing[1]
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
    cdef long i
    cdef long node
    cdef long offset
    cdef long n
    cdef long max_n
    cdef double dz
    cdef double max_dz

    for i in prange(n_nodes, nogil=True, schedule="static"):
        node = nodes[i]

        # Differences after filling can be very small, *1e3 to exaggerate those
        # Set to -1 at boundaries (active cell ==0)
        # If cells have equal slopes, the flow will be directed following
        # Landlab rotational ordening going first to cardial, then to diagonal cells
        max_dz = 0.0
        max_n = -1
        for n in range(8):
            offset = neighbor[n]
            if is_active_cell[node + offset]:
                dz = (
                    z_at_node[node] - z_at_node[node + offset]
                ) * one_over_distance_to_neighbor[n]

                if dz > max_dz:
                    max_dz = dz
                    max_n = n

        if max_n >= 0:
            receivers[node] = node + neighbor[max_n]
            distance_receiver[node] = 1.0 / one_over_distance_to_neighbor[max_n]
            receiver_link[node] = adj_link[node, max_n]
            steepest_slope[node] = max(
                0,
                (
                    z_original[node] - z_original[receivers[node]]
                ) * one_over_distance_to_neighbor[max_n]
            )
        else:
            receivers[node] = node
            receiver_link[node] = -1
            steepest_slope[node] = 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
def _accumulate_at_reciever_nodes(
    const id_t [:] nodes,
    const id_t [:] receiver_node_at_node,
    const cython.floating [:] value_at_node,
    cython.floating [:] receiver_value_at_node,
):
    cdef long i
    cdef long node

    with nogil:
        for i in range(len(nodes)):
            node = nodes[i]
            receiver_value_at_node[receiver_node_at_node[node]] += value_at_node[node]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _accumulate_flow_d8(
    cython.floating [:] a,
    cython.floating [:] q,
    const id_t [:] stack_flip,
    const id_t [:] receivers,
):
    """Accumulates drainage area and discharge, permitting transmission losses."""
    cdef long donor
    cdef long rcvr

    # Work from upstream to downstream.
    with nogil:
        for donor in stack_flip:
            rcvr = receivers[donor]
            a[rcvr] += a[donor]
            q[rcvr] += q[donor]
