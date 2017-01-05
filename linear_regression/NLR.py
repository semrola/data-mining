import numpy as np
import re
from matplotlib import pyplot as plt
from scipy import linalg, optimize
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def numerical_grad(f, params, epsilon):
    num_grad = np.zeros_like(theta)
    perturb = np.zeros_like(params)
    for i in range(params.size):
        perturb[i] = epsilon
        j1 = f(params + perturb)
        j2 = f(params - perturb)
        num_grad[i] = (j1 - j2) / (2. * epsilon)
        perturb[i] = 0
    return num_grad


def cost(X, Y, theta):
    Theta = theta.reshape(Y.shape[1], X.shape[1])
    M = np.dot(X, Theta.T)
    P = np.exp(M - np.max(M, axis=1)[:, None])
    P /= np.sum(P, axis=1)[:, None]
    cost = -np.sum(np.log(P) * Y) / X.shape[0]
    return cost


def avg_cost(params, *args):
    X = args[0]
    y = args[1]
    m = X.shape[0]
    n = X.shape[1]
    print(m,n,y.shape)

    vsota = 0
    for i in range(1, m):   #suma
        xi = X[i]
        h = np.dot(params.T, xi)
        vsota = vsota + np.square(h - y[i])
    cost1 = (1/m)*vsota
    return cost1

def cost2(thetas, *args):
    X = args[0]
    y = args[1]
    m = X.shape[0]
    n = X.shape[1]
    vsota = 0
    for i in range(1, m):   #suma
        xi = X[i]
        h = np.dot(thetas.T, xi)
        vsota = vsota + np.square(h - y[i])
    return vsota


def grad(X, Y, theta):
    Theta = theta.reshape(Y.shape[1], X.shape[1])
    M = np.dot(X, Theta.T)
    P = np.exp(M - np.max(M, axis=1)[:, None])
    P /= np.sum(P, axis=1)[:, None]
    grad = np.dot(X.T, P - Y).T / X.shape[0]
    return grad.ravel()


def grad1(thetas, *args):
    X = args[0]
    y = args[1]
    return (X.dot(thetas) - y).dot(X)


def regularized(X, y, lambda_=0.1):
    L = np.eye(X.shape[1]) * lambda_
    L[0] = 0
    return np.linalg.pinv(X.T.dot(X) + L).dot(X.T.dot(y))


def gradient_descent(data, alpha=0.0000025, epochs=1000):
    thetas = []
    for subject in data:
        theta = np.zeros(subject.shape[1]).T
        y = subject[-2] #predzadnji stolpec (intenziteta)
        for i in range(epochs):
            theta = theta - alpha * (subject.dot(theta) - y).dot(subject)
        thetas.append(theta)
    return thetas


def stochastic_gradient(X, y, alpha=0.0000025, epochs=1000):
    #thetas = np.zeros(X.shape[1]).T #inicializiramo thete na 0 (dolzina je enaka stevilu znacilk)
    m = X.shape[0]
    thetas = np.random.randn(X.shape[1]).T
    for i in range(epochs):
        ii = 0
        for x in X: #gremo cez vse primere; x je vektor (posamezen primer) x^(0) je area
            jj = 0
            for j in x: #gremo cez vse znacilke/atribute
                h = np.dot(thetas.T, x)
                thetas[jj] = thetas[jj] + alpha * (y[ii] - h) * j
                jj += 1
            ii += 1
    return thetas


def batch_gradient(X, y, alpha=0.000001, epochs=10000):
    m = X.shape[0]
    n = X.shape[1]
    thetas = np.random.randn(n).T
    #thetas = np.zeros(n).T
    tmp = np.zeros(n).T
    resOld = avg_cost(thetas, X, y)
    epsilon = 0.00001
    diff = 1
    iterations = 0
    #for ii in range(epochs):    #za konvergenco
    while diff > epsilon:
        for j in range(n): #gremo cez vse znacilke/atribute
            vsota = 0
            for i in range(1, m):   #suma
                xi = X[i]
                h = np.dot(thetas.T, xi)
                vsota = vsota + (h - y[i]) * xi[j]
            tmp[j] = thetas[j] - alpha * (1/m) * vsota
        thetas = tmp

        res = cost1(thetas, X, y)
        diff = np.average(np.abs(res - resOld))
        resOld = res

        iterations += 1

    print('iters:', iterations)
    return thetas


def main():
    train = True
    i = 0

    if train:
        # NALOGA
        stolpecNapovedi = -2    #predzadnji
        trainingSet = open('train.tab')
        testSet = open('train.tab')

    else:
        # PORTLAND
        stolpecNapovedi = -1    #zadnji
        trainingSet = open('portland.tab')
        testSet = open('portland.tab')

    Data = []
    X = []
    y = []
    for line in trainingSet:
        i = i + 1
        if i < 4:
            continue    #prve 3 vrstice preskocimo
        else:
            lineSplitted = re.split('\t', line.strip())
            X.append(lineSplitted[:stolpecNapovedi])
            y.append(lineSplitted[stolpecNapovedi])

    X = np.vstack(X)
    X = np.column_stack((np.ones(len(X)), X))   # dodamo 1 na zacetek (x0 je vedno 1)
    y = np.hstack(y)
    X = X.astype(float)
    y = y.astype(float)

    print('Prebrano')

    #Theta = batch_gradient(X, y) #vektor thet
    #print(Theta)

    #theta = np.random.randn(3)
    #ng = numerical_grad(lambda params: cost1(X, y, params), theta, 1e-4)
    #print(ng)

    args = []
    #print(X,y)
    print(X.shape)
    X_new = X
    Theta = optimize.fmin_l_bfgs_b(func=cost2, x0=np.ones(X_new.shape[1]), args=(X_new, y), fprime=grad1)[0]
    print(Theta.shape)

    print('---------TESTING----------')
    Xtest = []
    i = 0
    for line in testSet:
        i = i + 1
        if i < 4:
            continue    #prve 3 vrstice preskocimo
        else:
            lineSplitted = re.split('\t', line.strip())
            Xtest.append(lineSplitted[:stolpecNapovedi])

    Xtest = np.vstack(Xtest)
    Xtest = np.column_stack((np.ones(len(Xtest)), Xtest))   # dodamo 1 na zacetek (x0 je vedno 1)
    Xtest = Xtest.astype(float)

    print('X shape:', X.shape)

    result = open('mean.txt', 'w')

    rows = []
    for x in Xtest:
        a = Theta.dot(x)
        rows.append(a)
        print(abs(a), file=result)
        #print(a)
    result.close()
    print(sum(abs(rows - y)))

main()