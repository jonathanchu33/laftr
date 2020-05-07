import numpy as np

eps = 1e-12

def pos(Y):
    return np.sum(np.round(Y)).astype(np.float32)

def neg(Y):
    return np.sum(np.logical_not(np.round(Y))).astype(np.float32)

def PR(Y):
    return pos(Y) / (pos(Y) + neg(Y))

def NR(Y):
    return neg(Y) / (pos(Y) + neg(Y))

def TP(Y, Ypred):
    return np.sum(np.multiply(Y, np.round(Ypred))).astype(np.float32)

def FP(Y, Ypred):
    return np.sum(np.multiply(np.logical_not(Y), np.round(Ypred))).astype(np.float32)

def TN(Y, Ypred):
    return np.sum(np.multiply(np.logical_not(Y), np.logical_not(np.round(Ypred)))).astype(np.float32)

def FN(Y, Ypred):
    return np.sum(np.multiply(Y, np.logical_not(np.round(Ypred)))).astype(np.float32)

def FP_soft(Y, Ypred):
    return np.sum(np.multiply(np.logical_not(Y), Ypred)).astype(np.float32)

def FN_soft(Y, Ypred):
    return np.sum(np.multiply(Y, 1 - Ypred)).astype(np.float32)

#note: TPR + FNR = 1; TNR + FPR = 1
def TPR(Y, Ypred):
    return TP(Y, Ypred) / pos(Y)

def FPR(Y, Ypred):
    return FP(Y, Ypred) / neg(Y)

def TNR(Y, Ypred):
    return TN(Y, Ypred) / neg(Y)

def FNR(Y, Ypred):
    return FN(Y, Ypred) / pos(Y)

def FPR_soft(Y, Ypred):
    return FP_soft(Y, Ypred) / neg(Y)

def FNR_soft(Y, Ypred):
    return FN_soft(Y, Ypred) / pos(Y)

def calibPosRate(Y, Ypred):
    return TP(Y, Ypred) / pos(Ypred)

def calibNegRate(Y, Ypred):
    return TN(Y, Ypred) / neg(Ypred)

def errRate(Y, Ypred):
    return (FP(Y, Ypred) + FN(Y, Ypred)) / float(Y.shape[0])

def accuracy(Y, Ypred):
    return 1 - errRate(Y, Ypred)

def subgroup(fn, mask, Y, Ypred=None):
    m = np.greater(mask, 0.5).flatten()
    Yf = Y.flatten()
    if not Ypred is None: #two-argument functions
        Ypredf = Ypred.flatten()
        return fn(Yf[m], Ypredf[m])
    else: #one-argument functions
        return fn(Yf[m])

def DI_FP(Y, Ypred, A):
    fpr1 = subgroup(FPR, A, Y, Ypred)
    fpr0 = subgroup(FPR, 1 - A, Y, Ypred)
    return abs(fpr1 - fpr0)

def DI_FN(Y, Ypred, A):
    fnr1 = subgroup(FNR, A, Y, Ypred)
    fnr0 = subgroup(FNR, 1 - A, Y, Ypred)
    return abs(fnr1 - fnr0)


def DI_FP_soft(Y, Ypred, A):
    fpr1 = subgroup(FPR_soft, A, Y, Ypred)
    fpr0 = subgroup(FPR_soft, 1 - A, Y, Ypred)
    return abs(fpr1 - fpr0)

def DI_FN_soft(Y, Ypred, A):
    fnr1 = subgroup(FNR_soft, A, Y, Ypred)
    fnr0 = subgroup(FNR_soft, 1 - A, Y, Ypred)
    return abs(fnr1 - fnr0)

def DI(Y, Ypred, A):
    return (DI_FN(Y, Ypred, A) + DI_FP(Y, Ypred, A)) * 0.5

def DI_soft(Y, Ypred, A):
    return (DI_FN_soft(Y, Ypred, A) + DI_FP_soft(Y, Ypred, A)) * 0.5

def DP(Ypred, A): #demographic disparity
    return abs(subgroup(PR, A, Ypred) - subgroup(PR, 1 - A, Ypred))

def NLL(Y, Ypred, eps=eps):
    return -np.mean(np.multiply(Y, np.log(Ypred + eps)) + np.multiply(1. - Y, np.log(1 - Ypred + eps)))

# Individual Fairness metrics
def compute_distances(X, Z):
    row_norms_X = np.sum(np.square(X), axis=1)
    row_norms_X.reshape(-1, 1)
    dX = row_norms_X - 2 * np.matmul(X, np.transpose(X)) + np.transpose(row_norms_X)

    row_norms_Z = np.sum(np.square(Z), axis=1)
    row_norms_Z.reshape(-1, 1)
    dZ = row_norms_Z - 2 * np.matmul(Z, np.transpose(Z)) + np.transpose(row_norms_Z)

    dX = dX[np.triu_indices(dX.shape[0], 1)]
    dZ = dZ[np.triu_indices(dZ.shape[0], 1)]

    nonzero_mask = np.logical_not(np.abs(dX) < 1e-3)

    return dZ[nonzero_mask] / dX[nonzero_mask]

def max_ratio(X, Z):
    return np.max(compute_distances(X, Z))

def avg_ratio(X, Z):
    return np.mean(compute_distances(X, Z))

if __name__ == '__main__':
    Y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    Ypred = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.2, 0.3, 0.8, 0.9])
    A = np.array([1, 1, 0, 1, 0, 0, 0, 1, 0, 1])

    assert pos(Y) == 5
    assert neg(Y) == 5
    assert pos(Ypred) == 5
    assert neg(Ypred) == 5

    assert TP(Y, Ypred) == 3
    assert FP(Y, Ypred) == 2
    assert TN(Y, Ypred) == 3
    assert FN(Y, Ypred) == 2
    assert np.isclose(TPR(Y, Ypred), 0.6)
    assert np.isclose(FPR(Y, Ypred) , 0.4)
    assert np.isclose(TNR(Y, Ypred) , 0.6)
    assert np.isclose(FNR(Y, Ypred) , 0.4)
    assert np.isclose(calibPosRate(Y, Ypred) , 0.6)
    assert np.isclose(calibNegRate(Y, Ypred) , 0.6)
    assert np.isclose(errRate(Y, Ypred) , 0.4)
    assert np.isclose(accuracy(Y, Ypred) , 0.6)
    assert np.isclose(subgroup(TNR, A, Y, Ypred) , 0.5)
    assert np.isclose(subgroup(pos, 1 - A, Ypred) , 2)
    assert np.isclose(subgroup(neg, 1 - A, Y) , 3)
    assert np.isclose(DI_FP(Y, Ypred, A) , abs(1.0 / 6))
    assert np.isclose(DI_FP(Y, Ypred, 1 - A) , abs(1.0 / 6))
    assert np.isclose(DI_FN(Y, Ypred, A) , abs(1.0 / 6))
    assert np.isclose(DI_FN(Y, Ypred, 1 - A) , abs(1.0 / 6))
    assert np.isclose(subgroup(accuracy, A, Y, Ypred), 0.6)
    assert np.isclose(subgroup(errRate, 1 - A, Y, Ypred), 0.4)


