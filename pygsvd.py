import numpy as np
import numpy.linalg as la
import _gsvd


def gsvd(A, B, full_matrices=False, extras='uv', X1=False):
    '''Compute the generalized singular value decomposition of
    a pair of matrices ``A`` of shape ``(m, n)`` and ``B`` of
    shape ``(p, n)``

    The GSVD is defined as a joint decomposition, as follows.

        A = U*C*X.T
        B = V*S*X.T

        or

        A = U*C*inv(X1)
        B = V*S*inv(X1)

    where

        C.T*C + S.T*S = I

    where ``U`` and ``V`` are unitary matrices.

    Parameters
    ----------
    A, B : ndarray
        Input matrices on which to perform the decomposition. Must
        be no more than 2D (and will be promoted if only 1D). The
        matrices must also have the same number of columns.
    full_matrices : bool, optional
        If ``True``, the returned matrices ``U`` and ``V`` have
        at most ``p`` columns and ``C`` and ``S`` are of length ``p``.
    extras : str, optional
        A string indicating which of the orthogonal transformation
        matrices should be computed. By default, this only computes
        the generalized singular values in ``C`` and ``S``, and the
        right generalized singular vectors in ``X``. The string may
        contain either 'u' or 'v' to indicate that the corresponding
        matrix is to be computed.
    X1 : bool, optional
        If ``True``, X inverse transpose is returned in place of the
        default X matrix.  This may be useful for regularization
        routines.  This matrix satisfies U.T@A@X = C, V.T@B@X = S.

    Returns
    -------
    C : ndarray
        The generalized singular values of ``A``. These are returned
        in decreasing order.
    S : ndarray
        The generalized singular values of ``B``. These are returned
        in increasing order.
    X : ndarray
        The right generalized singular vectors of ``A`` and ``B``.
    U : ndarray
        The left generalized singular vectors of ``A``, with
        shape ``(m, m)``. This is only returned if
        ``'u' in extras`` is True.
    V : ndarray
        The left generalized singular vectors of ``B``, with
        shape ``(p, p)``. This is only returned if
        ``'v' in extras`` is True.

    Raises
    ------
    A ValueError is raised if ``A`` and ``B`` do not have the same
    number of columns, or if they are not both 2D (1D input arrays
    will be promoted).

    A ValueError is raised if (A.T, B.T).T does not have full column rank.
    This is due to complexities of sorting and may be removed later.

    A RuntimeError is raised if the underlying LAPACK routine fails.

    Notes
    -----
    This routine is intended to be as similar as possible to the
    decomposition provided by Matlab and Octave. Note that this is slightly
    different from the decomposition as put forth in Golub and Van Loan [1],
    and that this routine is thus not directly a wrapper for the underlying
    LAPACK routine.

    One important difference between this routine and that provided by
    Matlab is that this routine returns the singular values in decreasing
    order, for consistency with NumPy's ``svd`` routine.

    References
    ----------
    [1] Golub, G., and C.F. Van Loan, 2013, Matrix Computations, 4th Ed.
    '''
    # The LAPACK routine stores R inside A and/or B, so we copy to
    # avoid modifying the caller's arrays.
    dtype = np.complex128 if any(map(np.iscomplexobj, (A, B))) else np.double
    Ac = np.array(A, copy=True, dtype=dtype, order='C', ndmin=2)
    Bc = np.array(B, copy=True, dtype=dtype, order='C', ndmin=2)
    m, n = Ac.shape
    p = Bc.shape[0]
    if (n != Bc.shape[1]):
        raise ValueError('A and B must have the same number of columns')

    # Allocate input arrays to LAPACK routine
    compute_uv = tuple(each in extras for each in 'uv')
    sizes = (m, p)
    U, V = (np.zeros((size, size), dtype=dtype) if compute
            else np.zeros((1, 1), dtype=dtype)
            for size, compute in zip(sizes, compute_uv))
    Q = np.zeros((n, n), dtype=dtype)
    C = np.zeros((n,), dtype=np.double)
    S = np.zeros((n,), dtype=np.double)
    iwork = np.zeros((n,), dtype=np.int32)

    # Compute GSVD via LAPACK wrapper, returning the effective rank
    k, l = _gsvd.gsvd(Ac, Bc, U, V, Q, C, S, iwork,
                      compute_uv[0], compute_uv[1])
    # r is the rank of (A.T, B.T)
    # l is the rank of B
    r = k + l
    if r < n:
        print('''Warning: This code currently assumes rank(A.T, B.T)=n''')
    if l != min(p, n):
        print('''Warning: This code currently assumes the rank of B to be
                 the minimum of its rows and columns''')
    if la.matrix_rank(A) != min(m, n):
        print('''Warning: This code currently assumes the rank of A to be
                 the minimum of its rows and columns''')
    R = _extract_R(Ac, Bc, k, l)
    tmp = np.eye(n, dtype=R.dtype)
    if X1:
        # Compute X so that U'AX = C and V'BX = S
        tmp[n-r:, n-r:] = la.inv(R)
    else:
        # Compute X so that A = UCX' and B = VCX'
        tmp[n-r:, n-r:] = R.conj().T \
            if R.dtype == np.complex128 else R.T
    X = Q.dot(tmp)

    # Sort and reduce if needed
    if m >= n and p >= n:
        ix = np.argsort(C[:n])[::-1]
        C[:n] = C[:n][ix]
        S[:n] = S[:n][ix]
        X[:, :n] = X[:, ix]
        if not full_matrices:
            X = X[:, :n]
        if compute_uv[0]:
            U[:, :n] = U[:, ix]
            if not full_matrices:
                U = U[:, :n]
        if compute_uv[1]:
            V[:, :n] = V[:, ix]
            if not full_matrices:
                V = V[:, :n]
    elif m >= n:
        # Since p < n, we can't consistently sort more
        # than p columns.  The first n-p columns will correspond
        # to 1s in C or 0s in S -- they don't need to be sorted
        # For C, S, X, V we can specify last p cols, but for U,
        # we need to ensure the correct start column: n-p: n
        ix = np.argsort(C[-p:])[::-1]
        C[-p:] = C[-p:][ix]
        S[-p:] = S[-p:][ix]
        X[:, -p:] = X[:, -p:][:, ix]
        if not full_matrices:
            X = X[:, :n]
        if compute_uv[0]:
            U[:, n-p:n] = U[:, n-p:n][:, ix]
            if not full_matrices:
                U = U[:, :n]
        if compute_uv[1]:
            V = V[:, ix]
    else:
        # Since m < n, we can't consistently sort more
        # than m columns.  The last n-m columns will correspond
        # to 1s in C or 0s in S -- they don't need to be sorted
        ix = np.argsort(C[:m])[::-1]
        C[:m] = C[:m][ix]
        S[:m] = S[:m][ix]
        X[:, :m] = X[:, :m][:, ix]
        if not full_matrices:
            X = X[:, :n]
        if compute_uv[0]:
            U = U[:, ix]
        if compute_uv[1]:
            V[:, :m] = V[:, :m][:, ix]
            if not full_matrices:
                V = V[:, :n]

    outputs = (C, S, X) + tuple(arr for arr, compute in
                                zip((U, V), compute_uv) if compute)
    return outputs


def _extract_R(A, B, k, l):
    '''Extract the diagonalized matrix R from A and/or B.

    The indexing performed here is taken from the LAPACK routine
    help, which can be found here:

    ``http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_gab6c743f531c1b87922eb811cbc3ef645.html#gab6c743f531c1b87922eb811cbc3ef645``
    '''
    m, n = A.shape
    r = k + l
    # R should always have dimensions rxr
    R = np.zeros((r, r), dtype=A.dtype)
    if (m - r) >= 0:
        R = A[:r, n-r:]
    else:
        R[:m, :] = A[:, n-r:]
        R[m:, m:] = B[(m-k):l, (n+m-r):]
    return R
