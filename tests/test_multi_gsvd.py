from types import SimpleNamespace
from numpy.random import rand
from numpy.linalg import matrix_rank
from numpy import allclose, vstack, diag
from pygsvd import gsvd
import numpy as np
np.set_printoptions(suppress=True)

def test_many():
    with open('cases1.txt','r') as f:
        s = f.read()
        lines = s.strip().split('\n')
        cases = [l.strip().split('#') for l in lines]
        # each case should have a label and some assignments
        labels, assignments = zip(*cases)
        assignments = [a.strip().split(',') for a in assignments]
        assign_dicts = []
        for lst in assignments:
            d = {}
            for itm in lst:
                k, v = itm.strip().split('=')
                d[k]=int(v)
            assign_dicts.append(d)

    test_items = zip(labels, assign_dicts)

    for label, d in test_items:
        s = SimpleNamespace(**d)
        A = rand(s.m, s.n)
        B = rand(s.p, s.n)
        diff = min([s.n, s.p]) - s.l
        if diff > 0:
            B[:diff,:] = 0
        if s.r < s.n:
            A[:, :s.n-s.r] = 0
            B[:, :s.n-s.r] = 0

        C,S,X,U,V = gsvd(A,B,X1=True, full_matrices=True)
        # TODO: remove most or all of these once tests pass
        assert matrix_rank(B)==s.l
        assert matrix_rank(vstack((A,B))) == s.r
        assert s.q == s.m + s.p
        assert s.m + s.p >= s.n
        CC = U.T@A@X
        CCd = diag(CC)
        lCCd = CCd.shape[0]
        assert CC.shape == A.shape
        assert allclose(CCd, C[:lCCd]), \
            label + ' m={}, n={}, p={}, l={}, r={}' \
                    .format(s.m, s.n, s.p, s.l, s.r)
        SS = V.T@B@X
        SSd = diag(SS)
        lSSd = SSd.shape[0]
        assert CC.shape == A.shape
        assert allclose(SSd, S[:lSSd]), \
            label + ' m={}, n={}, p={}, l={}, r={}' \
                    .format(s.m, s.n, s.p, s.l, s.r)
