import itertools

import pygsvd
import numpy as np
import pytest
import numpy.linalg as la


def _load_matrices(mattype):
    return tuple(map(np.loadtxt,
                     ('{}/{}.txt'.format(mattype, x) for x in
                      ('a', 'b', 'c', 's', 'x', 'u', 'v'))))


def test_square_matrices():
    '''Test that the correctness of the routine on square matrices.

    This verifies that the returned matrices have the expected
    values, and that the original matrices can be reconstructed
    from them.
    '''
    matrices = _load_matrices('square')
    outputs = pygsvd.gsvd(matrices[0], matrices[1],
                          full_matrices=True, extras='uv')
    for xin, xout in zip(matrices[2:], outputs):
        assert np.allclose(np.abs(xin), np.abs(xout))
    c, s, x, u, v = outputs
    assert np.allclose(u.dot(np.diag(c)).dot(x.T), matrices[0])
    assert np.allclose(v.dot(np.diag(s)).dot(x.T), matrices[1])


def test_nonsquare_matrices():
    '''Test that the correctness of the routine on non-square matrices.

    This verifies that the returned matrices have the expected
    values, but only up to the number of columns in the inputs.
    The remaining columns are neither constrained nor relevant.
    It also verifies the reconstruction
    '''
    matrices = _load_matrices('nonsquare')
    outputs = pygsvd.gsvd(matrices[0], matrices[1],
                          full_matrices=True, extras='uv')
    max_size = matrices[0].shape[1]
    c, s, x, u, v = outputs
    for xin, xout in zip(matrices[2:4], (c, s)):
        assert np.allclose(np.abs(xin), np.abs(xout))
    for xin, xout in zip(matrices[4:], (x, u, v)):
        assert np.allclose(np.abs(xin[:, :max_size]),
                           np.abs(xout[:, :max_size]))
    assert np.allclose(u[:, :max_size].dot(np.diag(c)).dot(x.T), matrices[0])
    assert np.allclose(v[:, :max_size].dot(np.diag(s)).dot(x.T), matrices[1])


def test_nonsquare_matrices_p_lt_n_lt_m():
    '''Test that the correctness of the routine on non-square matrices
       a (mxn) and b (pxn), with p < n < m, where a has full column rank
       b has full row rank and m + p > n.

    This verifies that the returned matrices have the expected
    values, but only up to the number of columns in the inputs.
    The remaining columns are neither constrained nor relevant.
    It also verifies the reconstruction
    '''
    matrices = _load_matrices('nonsquare2')
    outputs = pygsvd.gsvd(matrices[0], matrices[1],
                          full_matrices=True, extras='uv')
    m, n = matrices[0].shape
    p = matrices[1].shape[0]
    c, s, x, u, v = outputs
    for xin, xout in zip(matrices[2:4], (c, s)):
        assert np.allclose(np.abs(xin), np.abs(xout))
    for xin, xout in zip(matrices[4:], (x, u, v)):
        assert np.allclose(np.abs(xin[:, :n]),
                           np.abs(xout[:, :n]))
    # Form matrices for c and s so we can reconstruct a and b
    cc = np.zeros((m, n))
    cc[:n, :n] = np.diag(c)
    ss = np.zeros((p, n))
    ss[:, n-p:] = np.diag(s[n-p:])
    assert np.allclose(u.dot(cc).dot(x.T), matrices[0])
    assert np.allclose(v.dot(ss).dot(x.T), matrices[1])


def test_nonsquare_matrices_p_lt_n_m():
    '''Test that the correctness of the routine on non-square matrices
       a (mxn) and b (pxn), with p < n < m, where a has full column rank
       b has full row rank and m + p > n.

    This verifies that the returned matrices have the expected
    values, but only up to the number of columns in the inputs.
    The remaining columns are neither constrained nor relevant.
    It also verifies the reconstruction
    '''
    matrices = _load_matrices('nonsquare3')
    outputs = pygsvd.gsvd(matrices[0], matrices[1],
                          full_matrices=True, extras='uv')
    m, n = matrices[0].shape
    p = matrices[1].shape[0]
    c, s, x, u, v = outputs
    for xin, xout in zip(matrices[2:4], (c, s)):
        assert np.allclose(np.abs(xin), np.abs(xout))
    for xin, xout in zip(matrices[4:], (x, u, v)):
        assert np.allclose(np.abs(xin[:, :n]),
                           np.abs(xout[:, :n]))
    # Form matrices for c and s so we can reconstruct a and b
    cc = np.diag(c)
    ss = np.zeros((p, n))
    ss[:, n-p:] = np.diag(s[n-p:])
    assert np.allclose(u.dot(cc).dot(x.T), matrices[0])
    assert np.allclose(v.dot(ss).dot(x.T), matrices[1])


def test_nonsquare_matrices_p_lt_n_eq_m():
    '''Test that the correctness of the routine on non-square matrices
       a (mxn) and b (pxn), with p < n = m, where a has full column rank
       b has full row rank.

    This verifies that the returned matrices have the expected
    values, but only up to the number of columns in the inputs.
    The remaining columns are neither constrained nor relevant.
    It also verifies the reconstruction
    '''
    n = 22
    L = .45
    power = 10
    c = np.zeros(n)
    c[:2] = [1-2*L, L]
    d = np.ones(n)
    a = np.diag(d*(1-2*L)) + np.diag(d[:-1]*L, 1) + np.diag(d[:-1]*L, -1)
    a = la.matrix_power(a, power)  # n x n blur array
    o = np.ones(n)
    b = (np.zeros((n, n)) - np.diag(o) + np.diag(o[:n-1], 1))[:n-1, :]
    outputs = pygsvd.gsvd(a, b,
                          full_matrices=True, extras='uv')
    m, n = a.shape
    p = b.shape[0]
    c, s, x, u, v = outputs
    # Form matrices for c and s so we can reconstruct a and b
    cc = np.ones(n)
    cc[n-p:] = c[1:]
    cc = np.diag(cc)
    ss = np.zeros((p, n))
    ss[:, n-p:] = np.diag(s[1:])
    assert np.allclose(u.dot(cc).dot(x.T), a)
    assert np.allclose(v.dot(ss).dot(x.T), b)


def test_A_short_B_square():
    '''Test the case: bnaeker/pygsvd - issue #5
       opened on Nov 13, 2018
       where ``a`` is a short matrix and ``b`` is square
    '''
    m, n, p = 4, 15, 15
    a = np.random.poisson(1, (m, n))
    b = np.random.poisson(1, (p, n))
    C, S, X, U, V = pygsvd.gsvd(a, b, extras='uv')
    CC = np.zeros((m, n))
    CC[:, :m] = np.diag(C[:m])

    assert np.allclose(U.dot(CC).dot(X.T), a)
    assert np.allclose(V.dot(np.diag(S)).dot(X.T), b)


def test_rank_deficient_matrices():
    '''Test the correctness of the routine when using rank-deficient matrices.

    In this case, the return matrix ``V`` is treated differently. As some
    of the first columns are irrelevant, the sorting of ``V`` applies only
    to the trailing columns.
    '''
    matrices = _load_matrices('rank-deficient')
    outputs = pygsvd.gsvd(matrices[0], matrices[1],
                          full_matrices=True, extras='uv')
    for xin, xout in zip(matrices[2:], outputs):
        assert np.allclose(np.abs(xin), np.abs(xout))
    c, s, x, u, v = outputs
    assert np.allclose(u.dot(np.diag(c)).dot(x.T), matrices[0])
    assert np.allclose(v.dot(np.diag(s)).dot(x.T), matrices[1])


def test_large_matrices():
    '''Test the correctness of the routine on larger matrices.'''
    matrices = _load_matrices('large')
    c, s, x, u, v = pygsvd.gsvd(matrices[0], matrices[1],
                                full_matrices=True, extras='uv')
    assert np.allclose(u.dot(np.diag(c)).dot(x.T), matrices[0])
    assert np.allclose(v.dot(np.diag(s)).dot(x.T), matrices[1])


def test_same_columns():
    '''Verify that the method raises a ValueError when the inputs
    have a different number of columns.
    '''
    with pytest.raises(ValueError):
        pygsvd.gsvd(np.arange(10).reshape(5, 2), np.arange(10).reshape(2, 5))


def test_dimensions():
    '''Verify that the input succeeds with 1D arrays (which are promoted
    to 2D), and raises a ValueError for 3+D arrays.
    '''
    x = np.arange(10)
    pygsvd.gsvd(x, x)
    with pytest.raises(ValueError):
        pygsvd.gsvd(x.reshape(5, 2, 1), x.reshape(5, 2, 1))


def test_non_full_matrices():
    '''Verify that the kwarg to return the "economy" decomposition works.

    This option allows returning the matrices of singular vectors only
    up to the actual numerical rank of the inputs, rather than the
    full square matrices.
    '''
    a, b = _load_matrices('nonsquare')[:2]
    c, s, x, u, v = pygsvd.gsvd(a, b, full_matrices=True, extras='uv')
    assert x.shape == (a.shape[1], a.shape[1])
    assert u.shape == (a.shape[0], a.shape[0])
    assert v.shape == (b.shape[0], b.shape[0])

    c, s, x, u, v = pygsvd.gsvd(a, b, extras='uv')
    assert x.shape == (a.shape[1], a.shape[1])
    assert u.shape == (a.shape[0], a.shape[1])
    assert v.shape == (b.shape[0], b.shape[1])


def test_return_extras():
    '''Verify that the extra arrays are returned as expected.'''
    names = ('a', 'b', 'c', 's', 'x', 'u', 'v')
    matrices = dict(zip(names,
                        map(np.loadtxt, ('square/{}.txt'.format(name)
                                         for name in names))))
    extras = 'uv'

    # Make all combinations of 'uv' of length 0 through 3, inclusive
    for combo_length in range(len(extras) + 1):
        combinations = list(itertools.combinations(extras, combo_length))

        for combo in combinations:
            # Join the combination letters to make the `extra` kwarg
            ex = ''.join(combo)

            # Compute GSVD including the extras, assigned to a dict
            out = dict(zip(('c', 's', 'r') + combo,
                           pygsvd.gsvd(matrices['a'], matrices['b'], extras=ex)))

            # Compare each extra output to the expected
            for each in combo:
                assert np.allclose(np.abs(out[each]), np.abs(matrices[each]))


def test_complex():
    '''Verify that the routine handles complex inputs correctly.

    If either one of the arrays is complex, the complex version of the
    LAPACK routine should be called (zggsvd3). The returns singular
    values are always real, but the returned matrices should be complex
    as well.
    '''
    a = np.loadtxt('square/a.txt')
    b = np.loadtxt('square/b.txt').astype(np.complex)
    c, s, x, u, v = pygsvd.gsvd(a, b, full_matrices=True, extras='uv')
    assert c.dtype == np.double  # Singular values are always real
    assert s.dtype == np.double  # Singular values are always real
    for matrix in (x, u, v):
        assert np.iscomplexobj(matrix)
    assert np.allclose(u.dot(np.diag(c)).dot(x.T.conj()), a)
    assert np.allclose(v.dot(np.diag(s)).dot(x.T.conj()), b)


def test_dtype_promotion():
    '''Verify that the routine handles inputs of non-double type correctly.

    Any real-valued non-double inputs (integer, float32) should be promoted
    to doubles.
    '''
    a, b = _load_matrices('square')[:2]
    dtypes = (np.int16, np.int32, np.int64, np.float32)
    for dtype in dtypes:
        outputs = pygsvd.gsvd(a.astype(dtype), b.astype(dtype))
        for out in outputs:
            assert out.dtype == np.double
