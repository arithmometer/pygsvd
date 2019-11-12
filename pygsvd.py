import numpy as np
import numpy.linalg as la
from scipy.linalg.lapack import dtrtri, ztrtri
import _gsvd


def gsvd(A, B, full_matrices=False, extras='uv', X1=False):
    '''Compute the generalized singular value decomposition of
    a pair of matrices ``A`` of shape ``(m, n)`` and ``B`` of
    shape ``(p, n)``

    The GSVD is defined as a joint decomposition, as follows.

        A = U*C*X.T   C = U.T*A*inv(X.T)
        B = V*S*X.T   S = V.T*B*inv(X.T)

        or letting X1 = inv(X.T)

        A = U*C*inv(X1)  C = U.T*A*X1
        B = V*S*inv(X1)  S = V.T*B*X1

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
        default X matrix.  This may be convenient for regularization
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
    # r is the rank of the matrix (A.T | B.T).T denoted A|B
    # l is the rank of B
    r = k + l
    R = _extract_R(Ac, Bc, k, l)
    tmp = np.eye(n, dtype=R.dtype)
    if X1:
        # Compute X so that U'AX = C and V'BX = S
        # invert R by back substitution
        tmp[n-r:, n-r:] = ztrtri(R, overwrite_c=1)[0] \
            if R.dtype == np.complex128 else dtrtri(R, overwrite_c=1)[0]
    else:
        # Compute X so that A = UCX' and B = VCX'
        tmp[n-r:, n-r:] = R.conj().T \
            if R.dtype == np.complex128 else R.T
    X = Q.dot(tmp)

    # Sort columns of X, U and V to achieve the correct ordering of
    # the singular values.
    if m - r >= 0:
        ix = np.argsort(C[k:r])[::-1]  # sort l values
        X[:, -l:] = X[:, -l:][:, ix]
        if compute_uv[0]:
            U[:, k:k+l] = U[:, k:k+l][:, ix]
        if compute_uv[1]:
            V[:, :l] = V[:, :l][:, ix]
        C[k:r] = C[k:r][ix]
        S[k:r] = S[k:r][ix]
    else:  # m - r < 0
        ix = np.argsort(C[k:m])[::-1]  # sort m-k values
        X[:, n-l:n+m-r] = X[:, n-l:n+m-r][:, ix]
        if compute_uv[0]:
            U[:, k:] = U[:, k:][:, ix]
        if compute_uv[1]:
            V[:, :m-k] = V[:, :m-k][:, ix]
        C[k:m] = C[k:m][ix]
        S[k:m] = S[k:m][ix]

    # For convenience in reconstructing A and B from their decompositions,
    # try to move SV's to the diagonal in cases when rank(A|B) < n.
    # This is not possible if rank(A|B) > rank(B) and
    # the number of rows of B is less than rank(A|B).
    if n-r > 0:
        X = np.roll(X, r-n, axis=1)
    if k > 0 and p >= r:
        V = np.roll(V, k, axis=1)
    # If full matrices are not required, limit X, U, and V to at most r
    # columns.
    if not full_matrices:
        X = X[:, :r]
        if compute_uv[0] and m > r:
            U = U[:, :r]
        if compute_uv[1] and p > r:
            V = V[:, :r]

    C = C[:r]
    S = S[:r]

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
