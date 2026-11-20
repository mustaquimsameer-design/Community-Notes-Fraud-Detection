"""
fraudar.py
----------
Numba-accelerated implementation of the FRAUDAR algorithm (Hooi et al., KDD 2016).
Uses an indexed binary min-heap to perform camouflage-resistant greedy peeling
on bipartite graphs (raters x notes).
"""

import time
import numpy as np
import scipy.sparse as sp
from numba import njit


@njit(fastmath=True)
def sift_up(i, heap, pos, keys):
    while i > 0:
        parent = (i - 1) >> 1
        if keys[heap[i]] < keys[heap[parent]]:
            u = heap[i]
            v = heap[parent]
            heap[i] = v
            heap[parent] = u
            pos[u] = parent
            pos[v] = i
            i = parent
        else:
            break


@njit(fastmath=True)
def sift_down(i, size, heap, pos, keys):
    while True:
        left = (i << 1) + 1
        right = left + 1
        smallest = i
        if left < size and keys[heap[left]] < keys[heap[smallest]]:
            smallest = left
        if right < size and keys[heap[right]] < keys[heap[smallest]]:
            smallest = right
        if smallest != i:
            u = heap[i]
            v = heap[smallest]
            heap[i] = v
            heap[smallest] = u
            pos[u] = smallest
            pos[v] = i
            i = smallest
        else:
            break


@njit(fastmath=True)
def fraudar_peel_core(N, M, indptr_r, indices_r, indptr_c, indices_c, col_weights):
    """
    Core greedy peeling loop compiled with Numba.
    
    Parameters:
    - N: number of raters (rows)
    - M: number of notes (columns)
    - indptr_r, indices_r: CSR representation of rater -> notes
    - indptr_c, indices_c: CSC representation of note -> raters
    - col_weights: 1D array of length M containing 1.0 / log(deg_j + c)
    """
    V = N + M
    keys = np.zeros(V, dtype=np.float64)

    # Initialize priority for raters: sum of weights of their incident notes
    for i in range(N):
        w_sum = 0.0
        for ptr in range(indptr_r[i], indptr_r[i + 1]):
            j = indices_r[ptr]
            w_sum += col_weights[j]
        keys[i] = w_sum

    # Initialize priority for notes: degree * col_weight
    for j in range(M):
        deg = indptr_c[j + 1] - indptr_c[j]
        keys[N + j] = deg * col_weights[j]

    total_w = 0.0
    for i in range(N):
        total_w += keys[i]

    heap = np.arange(V, dtype=np.int32)
    pos = np.arange(V, dtype=np.int32)

    # Build heap in O(V) time
    for i in range((V >> 1) - 1, -1, -1):
        sift_down(i, V, heap, pos, keys)

    size = V
    active_r = N
    active_c = M

    best_density = total_w / (active_r + active_c) if (active_r + active_c) > 0 else 0.0
    best_step = -1

    peeled_order = np.empty(V, dtype=np.int32)

    for step in range(V - 1):
        # Pop minimum priority node
        u = heap[0]
        last = heap[size - 1]
        heap[0] = last
        pos[last] = 0
        pos[u] = -1
        size -= 1
        sift_down(0, size, heap, pos, keys)

        peeled_order[step] = u

        if u < N:
            # Popped node is a rater
            i = u
            active_r -= 1
            total_w -= keys[i]
            for ptr in range(indptr_r[i], indptr_r[i + 1]):
                j = indices_r[ptr]
                v = N + j
                if pos[v] != -1:
                    keys[v] -= col_weights[j]
                    sift_up(pos[v], heap, pos, keys)
        else:
            # Popped node is a note
            j = u - N
            active_c -= 1
            total_w -= keys[u]
            for ptr in range(indptr_c[j], indptr_c[j + 1]):
                i = indices_c[ptr]
                v = i
                if pos[v] != -1:
                    keys[v] -= col_weights[j]
                    sift_up(pos[v], heap, pos, keys)

        if active_r > 0 and active_c > 0:
            cur_density = total_w / (active_r + active_c)
            if cur_density > best_density:
                best_density = cur_density
                best_step = step

    peeled_order[V - 1] = heap[0]
    return best_density, best_step, peeled_order


def fraudar_extract(csr_mat, col_weight_c=5.0):
    """
    Extracts the highest-density suspicious bipartite subgraph using FRAUDAR.
    
    Returns:
    - raters: np.ndarray of rater indices
    - notes: np.ndarray of note indices
    - weighted_density: float (camouflage-weighted score)
    - unweighted_density: float (edge_count / (len(raters) + len(notes)))
    - edge_count: int
    """
    N, M = csr_mat.shape
    if csr_mat.nnz == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), 0.0, 0.0, 0

    csc_mat = csr_mat.tocsc()
    col_deg = np.diff(csc_mat.indptr).astype(np.float64)

    # FRAUDAR column weights: 1 / log(deg + c)
    col_weights = np.zeros(M, dtype=np.float64)
    nonzero_mask = col_deg > 0
    col_weights[nonzero_mask] = 1.0 / np.log(col_deg[nonzero_mask] + col_weight_c)

    best_density, best_step, peeled_order = fraudar_peel_core(
        N, M,
        csr_mat.indptr.astype(np.int32),
        csr_mat.indices.astype(np.int32),
        csc_mat.indptr.astype(np.int32),
        csc_mat.indices.astype(np.int32),
        col_weights,
    )

    if best_step == -1:
        surviving = peeled_order
    else:
        surviving = peeled_order[best_step + 1:]

    subgraph_raters = np.array([x for x in surviving if x < N], dtype=np.int32)
    subgraph_notes = np.array([x - N for x in surviving if x >= N], dtype=np.int32)

    subgraph_raters.sort()
    subgraph_notes.sort()

    # Calculate actual unweighted subgraph metrics
    if len(subgraph_raters) > 0 and len(subgraph_notes) > 0:
        sub_mat = csr_mat[subgraph_raters, :][:, subgraph_notes]
        edge_count = sub_mat.nnz
        unweighted_density = edge_count / (len(subgraph_raters) + len(subgraph_notes))
    else:
        edge_count = 0
        unweighted_density = 0.0

    return subgraph_raters, subgraph_notes, float(best_density), float(unweighted_density), int(edge_count)


def remove_subgraph_edges(csr_mat, raters, notes):
    """
    In-place zeroes out edges between `raters` and `notes` and drops them from the CSR matrix.
    """
    if len(raters) == 0 or len(notes) == 0:
        return csr_mat

    in_notes = np.zeros(csr_mat.shape[1], dtype=bool)
    in_notes[notes] = True

    for i in raters:
        start = csr_mat.indptr[i]
        end = csr_mat.indptr[i + 1]
        for ptr in range(start, end):
            j = csr_mat.indices[ptr]
            if in_notes[j]:
                csr_mat.data[ptr] = 0

    csr_mat.eliminate_zeros()
    return csr_mat
