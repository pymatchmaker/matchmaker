import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t

def viterbi_step_cy(
    np.ndarray[DTYPE_t, ndim=1] prev_probs,
    np.ndarray[DTYPE_t, ndim=2] alpha,
    np.ndarray[DTYPE_t, ndim=1] S,
    np.ndarray[DTYPE_t, ndim=1] r,
    np.ndarray[DTYPE_t, ndim=1] b,
    int D1,
    int D2
):
    """
    This function performs a fast outer-product Viterbi update.
    Parameters
    ----------
    prev_probs : ndarray (N,)
        Previous state probabilities.
    alpha : ndarray (N, N)
        Transition probabilities between states.
    S : ndarray (N,)
        Skip probabilities for each state.
    r : ndarray (N,)
        Resumption probabilities for each state.
    b : ndarray (N,)
        Emission probabilities for each state.
    D1 : int
        Maximum forward transition distance.
    D2 : int
        Maximum backward transition distance.
    Returns
    -------
    new_probs : ndarray (N,)
        Updated state probabilities after the Viterbi step.
    """
    
    cdef int n_states = prev_probs.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] new_probs = np.zeros(n_states, dtype=np.float64)
    cdef int i, j, j_start, j_end
    cdef double local_max, val, skip_contrib, global_skip_max

    # Compute skip values and global max
    cdef np.ndarray[DTYPE_t, ndim=1] skip_values = prev_probs * S
    global_skip_max = np.max(skip_values)

    for i in range(n_states):
        j_start = max(0, i - D2)
        j_end = min(n_states, i + D1 + 1)
        local_max = 0.0
        for j in range(j_start, j_end):
            val = prev_probs[j] * alpha[j, i]
            if val > local_max:
                local_max = val
        skip_contrib = r[i] * global_skip_max
        new_probs[i] = b[i] * (skip_contrib if skip_contrib >= local_max else local_max)

    if np.sum(new_probs) > 0:
        new_probs /= np.sum(new_probs)
    else:
        new_probs = np.ones(n_states) / n_states
    return new_probs