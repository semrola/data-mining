__author__ = 'Sandi'

import numpy as np
from scipy import optimize as opt
import time
import Orange
from sklearn.cross_validation import train_test_split
import scipy as sp
from sklearn.metrics import log_loss
from sklearn import preprocessing

def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = np.sum(act*sp.log(pred) + sp.subtract(1, act)*sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0/len(act)
    return ll

def write(X):
    with open('uniform.csv', 'r') as f:
        for line in f:
            cols = line
            break
    with open('results.1.csv', 'wt') as out:
        out.write(cols)
        id = 1
        for line in X:
            tmp = str(id)
            id += 1
            for col in line:
                tmp += ',' + str(col)
            out.write('%s\n' % tmp)
    print('Writing done.')

def readTrain():
    x = []
    file = 'train.csv'
    i = 0
    with open(file, 'r') as f:
        for _ in range(1):
            next(f)
        for line in f:
            splitted = line.split(',')
            splitted[-1] = splitted[-1].split('_')[1].strip()
            if i % vsak == 0:
                x.append(splitted)
            i += 1
    return x

def readTest():
    x = []
    file = 'test.csv'
    with open(file, 'r') as f:
        for _ in range(1):
            next(f)
        for line in f:
            splitted = line.split(',')
            x.append(splitted)
    return x

def h(X, theta):
    return g(X.dot(theta))

def g(z):
    return 1 / (1 + np.power(np.e, -z))

def gradLog(theta, X, y, lambda_=0.1):
    m = X.shape[0]
    n = len(theta)
    grad = np.zeros(len(theta))

    for j in range(0, n):
        vsota = 0
        for i in range(m):
            vsota += (h(X[i], theta) - y[i])*X[i][j]
        if j == 0:
            vsota *= (1/m)
        else:
            vsota = (1/m) * (vsota + (lambda_ / m) * theta[j])
        grad[j] = vsota
    return grad

def costLog(theta, X, y, lambda_=0.1):
    eps = 1e-15
    m = X.shape[0]
    vsota = 0
    for i in range(m):
        vsota += y[i]*np.log(h(X[i], theta)+eps) + (1 - y[i]) * np.log(1 - h(X[i], theta) + eps)
    return -(1/m) * (vsota + (lambda_ / (2*m)) * np.sum(theta[1:]**2))

def count_class(x):
    if x.size == 0:
        return 0
    st = 0
    diff = []
    for item in x:
        if item not in diff:
            diff.append(item)
            st += 1
    return st, diff

def numerical_grad(f, params, epsilon):
    num_grad = np.zeros_like(theta1)
    perturb = np.zeros_like(params)
    for i in range(params.size):
        perturb[i] = epsilon
        j1 = f(params + perturb)
        j2 = f(params - perturb)
        num_grad[i] = (j1 - j2) / (2. * epsilon)
        perturb[i] = 0
    return num_grad

def gradient(theta, X, Y, lam=0.1):
    Theta = theta.reshape(Y.shape[1], X.shape[1])
    M = np.dot(X, Theta.T)
    P = np.exp(M - np.max(M, axis=1)[:, None])
    P /= np.sum(P, axis=1)[:, None]
    m = X.shape[0]
    grad = np.dot(X.T, P - Y).T / m + (lam/m)*Theta
    return grad.ravel()

def cost(theta, X, Y, lam=0.1, eps=1e-15):
    Theta = theta.reshape(Y.shape[1], X.shape[1])
    M = np.dot(X, Theta.T)
    P = np.exp(M - np.max(M, axis=1)[:, None])
    P /= np.sum(P, axis=1)[:, None]
    m = X.shape[0]
    cost = -np.sum(np.log(P+eps) * Y) / m + (lam / (2*m)) * np.sum(theta[1:]**2)
    return cost

def cost_grad(Theta_flat, X, Y, lambda_=0.1):
    eps = 1e-15
    Theta = Theta_flat.reshape((classes, X.shape[1]))
    M = X.dot(Theta.T)
    P = np.exp(M - np.max(M, axis=1)[:, None])
    P /= np.sum(P, axis=1)[:, None]
    cost = -np.sum(np.log(P+eps) * Y)
    cost += lambda_ * Theta_flat.dot(Theta_flat) / 2.0
    cost /= X.shape[0]
    grad = X.T.dot(P - Y).T
    grad += lambda_ * Theta
    grad /= X.shape[0]
    return cost, grad.ravel()

def fit(X, Y, cost, grad=None, lambda_=0.1):
    theta, f, d = opt.fmin_l_bfgs_b(cost, theta1, fprime=grad, args=(X, Y, lambda_))
    if not grad:
        theta = theta.reshape((X.shape[1], classes))
    return theta

def OvA(X, y, cost, grad=None, lambda_=0.1):
    classifiers = []
    for i in range(classes):
        #print(i+1)
        y1 = y.copy()
        tmp = (y1 == i).astype(int)
        theta = fit(X, tmp, cost, grad, lambda_)
        classifiers.append(theta)
    thetas = np.array(classifiers)  # shape = (9,94) - treba.T
    return thetas.T

def predict(X, theta):
    M = X.dot(theta)
    P = np.exp(M - np.max(M, axis=1)[:, None])
    P /= np.sum(P, axis=1)[:, None]
    return P

def CV(X, Y, lambda_, fitter, predictor, cost, grad=None, k=5):
    scores = []
    scores1 = []
    for i in range(k):
        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=1/k, random_state=i)
        theta = fitter(Xtr, Ytr, cost, grad, lambda_)
        preds = predictor(Xte, theta)
        if Yte.ndim == 1:
            Yte = np.eye(classes)[Yte.astype(int)]
        sc = llfun(Yte, preds)
        sc1 = log_loss(Yte, preds)
        scores.append(sc)
        scores1.append(sc1)
        print("I=", i+1, sc, sc1)
    scs = np.mean(scores)
    scs1 = np.mean(scores1)
    print('score:', scs, scs1)
    return scs

theta1 = None
classes = 0
vsak = 2

def main():
    global theta1
    global classes


    X = readTrain()
    X = np.array(X).astype(float)
    X[:, 0] = 1 #na zacetek damo samo enke
    y = X[:, -1].copy() #zadnji stolpec je y
    y -= 1
    print(np.bincount(y.astype(int)))
    X = np.delete(X, -1, axis=1)    #izbrisemo zadnji stolpec
    Xtest = readTest()
    Xtest = np.array(Xtest).astype(float)
    Xtest[:, 0] = 1
    print('Reading done.')

    '''
    data = Orange.data.Table('iris')
    X = data.X
    X = X.T
    y = data.Y
    X = np.vstack((np.ones(X.shape[0]), X.T))
    '''

    print(X.shape, y.shape)
    lambda_ = 1

    classes = count_class(y)[0]
    Y = np.eye(classes)[y.astype(int)]
    theta1 = np.zeros(X.shape[1] * Y.shape[1])

    grad_check = False
    #gradient check Softmax & LogReg
    if grad_check:
        ag = gradient(theta1, X, Y, lambda_)
        ng = numerical_grad(lambda params: cost(params, X, Y, lambda_), theta1, 1e-4)
        print('gradient check #1', np.sum((ag - ng)**2))

        costng = lambda a: costLog(a, X, Y, lambda_)
        gradng = lambda a: gradLog(a, X, Y, lambda_)
        ng = opt.check_grad(costng, gradng, theta1)
        print('gradient check #2', ng)

    #softmax
    '''
    CV(X, Y, lambda_, fit, predict, cost_grad)
    if False:
        theta = fit(X, Y, cost_grad, lambda_=lambda_)
        res = predict(Xtest, theta)
        write(res)
        suma = np.sum(res, axis=1)
        print('sumtest', [i for i in suma if np.abs(i - 1) > 1e-15])

    '''

    # logreg
    theta1 = np.zeros(X.shape[1])
    if True:
        theta = OvA(X, y, costLog, gradLog, lambda_=lambda_)
        res = predict(Xtest, theta)
        write(res)
        suma = np.sum(res, axis=1)
        print('sumtest', [i for i in suma if np.abs(i - 1) > 1e-15])

    CV(X, y, lambda_, OvA, predict, costLog, gradLog)


    #lams = [1e6, 1e7, 1e8]
    #for l in lams:
    #    print('lambda:', l)
    #    CV(X, Y, l)

    #normalizer = preprocessing.Normalizer().fit(X)
    #X = normalizer.transform(X)


main()