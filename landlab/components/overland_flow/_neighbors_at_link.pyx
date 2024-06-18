import numpy as np

cimport cython
cimport numpy as np
from cython.parallel cimport prange
from libc.math cimport fabs
from libc.math cimport powf

ctypedef fused id_t:
    cython.integral
    long long


@cython.boundscheck(False)
def neighbors_at_link(
    id_t [:] links,
    shape,
    id_t [:, :] out,
):
    cdef int stride
    cdef int n_links
    cdef int link
    cdef int i
    cdef bint is_top, is_bottom, is_left, is_right

    stride = 2 * shape[1] - 1
    n_links = (shape[0] - 1) * shape[1] + shape[0] * (shape[1] - 1)

    for i in range(links.shape[0]):
        link = links[i]

        is_top = link > (n_links - stride)
        is_bottom = link < stride
        is_left = link % stride == 0 or (link + shape[1]) % stride == 0
        is_right = (link - (shape[1] - 2)) % stride == 0 or (link + 1) % stride == 0

        if not is_right:
            out[i, 0] = link + 1

        if not is_top:
            out[i, 1] = link + stride

        if not is_left:
            out[i, 2] = link - 1

        if not is_bottom:
            out[i, 3] = link - stride


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def calc_discharge_at_link(
    shape,
    cython.floating [:] q_at_link,
    const cython.floating [:] h_at_link,
    const cython.floating [:] water_slope_at_link,
    const cython.floating [:] mannings_at_link,
    const double theta,
    const double g,
    const double dt,
):
    cdef long n_rows = shape[0]
    cdef long n_cols = shape[1]
    cdef long horizontal_links_per_row = n_cols - 1
    cdef long vertical_links_per_row = n_cols
    cdef long links_per_row = horizontal_links_per_row + vertical_links_per_row
    cdef long n_links = horizontal_links_per_row * n_rows + vertical_links_per_row * (n_rows - 1)
    cdef long row
    cdef long first_link
    cdef long link
    cdef np.ndarray[cython.floating, ndim=1] q_mean_at_link = np.empty_like(n_links)

    weighted_mean_of_parallel_links(
        shape,
        theta,
        q_at_link,
        q_mean_at_link,
    )

    for row in prange(0, n_rows - 1, nogil=True, schedule="static"):
        first_link = links_per_row * row
        for link in range(first_link, first_link + links_per_row):
            q_at_link[link] = (
                q_mean_at_link[link] - g * dt * h_at_link[link] * water_slope_at_link[link]
            ) * powf(h_at_link[link], 7.0 / 3.0) / (
                g * dt * mannings_at_link[link] ** 2.0 * fabs(q_at_link[link])
            )


@cython.boundscheck(False)
@cython.wraparound(False)
def sum_parallel_links(
    cython.numeric[:] out,
    const cython.numeric[:] value_at_link,
    shape,
):
    cdef int n_rows = shape[0]
    cdef int n_cols = shape[1]
    cdef int links_per_row = 2 * shape[1] - 1
    cdef int row, col
    cdef int link

    for row in prange(n_rows, nogil=True, schedule="static"):
        link = row * links_per_row + 1
        for col in range(1, n_cols - 2):
            out[link] = value_at_link[link - 1] + value_at_link[link + 1]
            link = link + 1

    for row in prange(1, n_rows - 2, nogil=True, schedule="static"):
        link = row * links_per_row + n_cols - 1
        for col in range(n_cols):
            out[link] = value_at_link[link - links_per_row] + value_at_link[link + links_per_row]
            link = link + 1


@cython.boundscheck(False)
@cython.wraparound(False)
def weighted_mean_of_parallel_links(
    shape,
    const double weight,
    const cython.floating [:] value_at_link,
    cython.floating [:] out,
):
    cdef long n_rows = shape[0]
    cdef long n_cols = shape[1]
    cdef long horizontal_links_per_row = n_cols - 1
    cdef long vertical_links_per_row = n_cols
    cdef long links_per_row = horizontal_links_per_row + vertical_links_per_row
    cdef long row
    cdef long first_link
    cdef long link

    for row in prange(0, n_rows, nogil=True, schedule="static"):
        first_link = links_per_row * row

        link = first_link
        out[link] = _calc_weighted_mean(
            0.0,
            value_at_link[link],
            value_at_link[link + 1],
            weight,
        )
        for link in range(first_link + 1, first_link + horizontal_links_per_row - 1):
            out[link] = _calc_weighted_mean(
                value_at_link[link - 1],
                value_at_link[link],
                value_at_link[link + 1],
                weight,
            )
        link = first_link + horizontal_links_per_row - 1
        out[link] = _calc_weighted_mean(
            value_at_link[link - 1],
            value_at_link[link],
            0.0,
            weight,
        )

    with nogil:
        first_link = horizontal_links_per_row
        for link in range(first_link, first_link + vertical_links_per_row):
            out[link] = _calc_weighted_mean(
                0.0,
                value_at_link[link],
                value_at_link[link + links_per_row],
                weight,
            )
    for row in prange(1, n_rows - 1, nogil=True, schedule="static"):
        first_link = links_per_row * row + horizontal_links_per_row
        for link in range(first_link, first_link + vertical_links_per_row):
            out[link] = _calc_weighted_mean(
                value_at_link[link - links_per_row],
                value_at_link[link],
                value_at_link[link + links_per_row],
                weight,
            )
    with nogil:
        first_link = links_per_row * (n_rows - 2) + horizontal_links_per_row
        for link in range(first_link, first_link + vertical_links_per_row):
            out[link] = _calc_weighted_mean(
                value_at_link[link - links_per_row],
                value_at_link[link],
                0.0,
                weight,
            )


cdef cython.floating _calc_weighted_mean(
    cython.floating value_at_left,
    cython.floating value_at_center,
    cython.floating value_at_right,
    cython.floating weight,
) noexcept nogil:
    return weight * value_at_center + (1.0 - weight) * 0.5 * (value_at_left + value_at_right)
