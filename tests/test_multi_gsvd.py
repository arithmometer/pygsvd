from types import SimpleNamespace
from numpy.random import rand
from numpy.linalg import matrix_rank
from numpy import allclose, vstack, diag, zeros
from pygsvd import gsvd
import numpy as np
np.set_printoptions(suppress=True)


def test_many():
    """
    Reads a list of test cases from text file and tests
    the GSVD result for both the full and reduced matrices cases
    """
    def check_case(full_matrices=False):
        """
        Checks a single case
        """
        C, S, X, U, V = gsvd(A, B, X1=True, full_matrices=True)
        assert matrix_rank(B) == s.l, msg
        assert matrix_rank(vstack((A, B))) == s.r, msg
        assert s.q == s.m + s.p, msg
        CC = U.T@A@X
        if s.m < s.r:
            CCd = zeros(s.r)
            CCd[:s.m] = diag(CC)
        else:
            CCd = diag(CC)
            CCd = CCd[:s.r]
        assert CCd.shape == C.shape, msg
        assert allclose(CCd, C), msg
        SS = V.T@B@X
        k = s.r-s.l
        if k > 0 and s.p < s.r:
            SSd1 = diag(SS, k)
            SSd = zeros(s.r)
            SSd[k:] = SSd1[:s.r-k]
        else:
            SSd = diag(SS)
            SSd = SSd[:s.r]
        assert SSd.shape == S.shape, msg
        assert allclose(SSd, S), msg

    with open('cases.txt', 'r') as f:
        s = f.read()
        lines = s.strip().split('\n')
        cases = [l.strip().split('#') for l in lines]
        labels, assignments = zip(*cases)
        assignments = [a.strip().split(',') for a in assignments]
        assign_dicts = []
        for lst in assignments:
            d = {}
            for itm in lst:
                k, v = itm.strip().split('=')
                d[k] = int(v)
            assign_dicts.append(d)
    test_items = zip(labels, assign_dicts)

    for label, d in test_items:
        s = SimpleNamespace(**d)
        msg = label + ' m={}, n={}, p={}, l={}, r={}' \
            .format(s.m, s.n, s.p, s.l, s.r)
        A = rand(s.m, s.n)
        B = rand(s.p, s.n)
        # Ensure that the rank of AB = s.r by zeroing columns as needed
        r = matrix_rank(vstack((A, B)))
        i = 0
        while r > s.r:
            A[:, i] = 0
            B[:, i] = 0
            i += 1
            r = matrix_rank(vstack((A, B)))
        # Ensure that the rank of B = s.l by zeroing rows of B as needed
        l = matrix_rank(B)
        diff = l - s.l
        if diff > 0:
            B[:diff, :] = 0

        check_case(full_matrices=True)
        check_case(full_matrices=False)
