import numpy as np
from scipy.stats import kde
from scipy.linalg import sqrtm


def kl(p, q, eps=1e-10):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    e = p * np.log2(p)
    e[p == 0] = 0
    ce = p * np.log2(q)
    ce[p == 0] = 0
    return np.sum(e - ce)


def js(p, q, eps=1e-10):
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m, eps) + 0.5 * kl(q, m, eps)


def apply_gaussian_kernel(X, xi, yi):
    k = kde.gaussian_kde([X[:, 0], X[:, 1]])
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    return zi / zi.sum()


def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid