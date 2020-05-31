"""
This is the code of a main function (signatureQ), which given a symetric matrix
outputs its signature. It's a helpful information when it comes to compute
extrema in Rn functions by its Hessian matrix.

Algorithm used is inspired in the one proposed in:
https://math.stackexchange.com/questions/1388421/reference-for-linear-algebra-books-that-teach-reverse-hermite-method-for-symmetr
"""

import numpy as np
import sympy as sp


def signatureQ(matrix):
    """Given a sympy matrix Returns a triplet (signature of the matrix) and a
    numpy array P, the transformation matrix to obtain the diagonalized and
    normalized matrix of the one given."""

    A = np.array(matrix).astype(np.float)
    A = insert_Id(A)
    n = len(A)
    for i in range(n-1):
        if A[i][i] == 0:
            j = i+1
            while j < n and A[j][j] == 0:
                j += 1
            if j == n:
                A = case_3(A, i)
            else:
                A = case_2(A, i, j)
        else:
            A = case_1(A, i)

    A = normalize(A)
    P = get_p(A)
    n_mes, n_menys, n_zero = get_signature(A)

    return (n_mes, n_menys, n_zero), P


def insert_Id(A):
    n = len(A)
    Id = np.eye(n)
    A = np.c_[A, Id]

    return A


def case_1(A, i):
    n = len(A)
    for j in range(i+1, n):
        if j != i:
            A = gauss(A, i, j)
            A = gauss(A.T, i, j).T
    return A


def gauss(A, i, j):
    A[j] = -A[j][i]*A[i] + A[i][i]*A[j]
    return A


def case_2(A, i, j=None):
    n = len(A)
    if j == None:
        j = i+1
        while j < n:
            if A[j][j] == 0:
                j += 1

    A = interchange(A, i, j)
    A = interchange(A.T, i, j).T

    return case_1(A, i)


def interchange(A, i, j):
    A[[i, j]] = A[[j, i]]
    return A


def case_3(A, i):
    n = len(A)
    k, l = find_non_zero(A, i, n)
    if k == None:
        return A
    A = combine(A, k, l)
    A = combine(A.T, k, l).T

    return case_2(A, i)


def find_non_zero(A, i, n):
    for k in range(i+1, n):
        for l in range(k, n):
            if A[k][l] != 0:
                return k, l
    return None, None


def combine(A, k, l):
    A[k] += A[l]
    return A


def normalize(A):
    n = len(A)
    for i in range(n):
        if A[i][i] != 0 and np.abs(A[i][i]) != 1:
            x = np.abs(A[i][i])
            A = divide(A, i, x)
            A = divide(A.T, i, x).T
    return A


def divide(A, i, x):
    n = len(A[0])
    for j in range(n):
        A[i][j] /= np.sqrt(x)
    return A


def get_p(A):
    n = len(A)
    P = np.zeros((n, n))
    for i in range(n):
        P[i] = A[i][n:]
    return P


def get_signature(A):
    n_mes = n_menys = n_zero = 0
    n = len(A)
    for i in range(n):
        d = round(A[i][i], 0)
        if d == 1:
            n_mes += 1
        elif d == 0:
            n_zero += 1
        elif d == -1:
            n_menys += 1
    return n_mes, n_menys, n_zero
