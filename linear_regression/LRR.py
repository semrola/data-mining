import numpy as np
import re
import time
from matplotlib import pyplot as plt
from scipy import linalg, optimize
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
#chi2,f_classif,f_regression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn import metrics
from sklearn import cross_validation
from scipy.stats.stats import pearsonr
from sklearn.decomposition import PCA

def numerical_grad(f, params, epsilon, theta):
    num_grad = np.zeros_like(theta)
    perturb = np.zeros_like(params)
    for i in range(params.size):
        perturb[i] = epsilon
        j1 = f(params + perturb)
        j2 = f(params - perturb)
        num_grad[i] = (j1 - j2) / (2. * epsilon)
        perturb[i] = 0
    return num_grad


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


def grad(theta, *args):
    X = args[0]
    Y = args[1]
    Theta = theta.reshape(Y.shape[1], X.shape[1])
    M = np.dot(X, Theta.T)
    P = np.exp(M - np.max(M, axis=1)[:, None])
    P /= np.sum(P, axis=1)[:, None]
    grad = np.dot(X.T, P - Y).T / X.shape[0]
    return grad.ravel()


def cost(thetas, *args):
    X = args[0]
    Y = args[1]
    Theta = thetas.reshape(X.shape[0], X.shape[1])
    M = np.dot(X, Theta.T)
    P = np.exp(M - np.max(M, axis=1)[:, None])
    P /= np.sum(P, axis=1)[:, None]
    cost = -np.sum(np.log(P) * Y) / X.shape[0]
    return cost


def cost1(thetas, *args):
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


def cost2(thetas, *args):
    X = args[0]
    y = args[1]
    lambda_ = args[2]
    m = X.shape[0]
    n = X.shape[1]
    vsota = 0
    for i in range(1, m):   #suma
        xi = X[i]
        h = thetas.dot(xi)
        vsota += (h - y[i])**2
    vsota2 = 0
    for i in range(1, n):
        vsota2 += thetas[i]**2
    return (1/(2*m)) * (vsota + lambda_ * vsota2)


def grad1(thetas, *args):
    X = args[0]
    y = args[1]
    return (X.dot(thetas) - y).dot(X)


def grad2(thetas, *args):
    X = args[0]
    y = args[1]
    lam = args[2]
    alpha = args[3]
    m = X.shape[0]
    n = X.shape[1]
    for j in range(1,n):
        suma = 0
        for i in range(1,m):
            xi = X[i]
            h = thetas.dot(xi)
            suma += (h - y[i]) * xi[j]
            suma += (lam / n) * thetas[j]
        suma *= 1/m
        thetas[j] -= alpha * suma
    return thetas


def grad3(thetas, *args):
    X = args[0]
    y = args[1]
    lam = args[2]

    m = X.shape[0]
    n = X.shape[1]
    for j in range(1,n):
        suma = 0
        for i in range(1,m):
            xi = X[i]
            h = thetas.dot(xi)
            suma += (h - y[i]) * xi[j]
        suma *= alpha * 1/m
        thetas[j] = thetas[j] * (1 - alpha * (lam/m)) - suma
    return thetas


def grad4(thetas, *args):
    X = args[0]
    y = args[1]
    lam = args[2]
    m = X.shape[0]
    n = X.shape[1]

    t0 = 0
    for i in range(1,m):
        xi = X[i]
        h = thetas.dot(xi)
        t0 += (h - y[i]) * xi[0]
    t0 *= 1/m * alpha
    thetas[0] -= t0
    '''
    for j in range(1,n):
        suma = 0
        for i in range(1,m):
            xi = X[i]
            h = thetas.dot(xi)
            suma += (h - y[i]) * xi[j]
        suma -= (lam/n) * thetas[j]
        suma *= 1/m
        suma *= alpha
        thetas[j] -= suma
    '''
    for j in range(1, n):
        suma = 0
        for i in range(1,m):
            xi = X[i]
            h = thetas.dot(xi)
            suma += (h - y[i]) * xi[j]
        suma *= 1/m
        suma *= alpha
        thetas[j] = thetas[j]*(1 - alpha*(lam/m)) - suma
    return thetas


def regularized(thetas, *args):
    X = args[0]
    y = args[1]
    lam = args[2]
    L = np.eye(X.shape[1])    #identiteta z 0 levo zgoraj (n+1) dimenzij
    L[0, 0] = 0
    inv = np.linalg.pinv(X.T.dot(X) + lam * L)
    return inv.dot(X.T.dot(y))


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


def preberi(file, skip_lines, stolpecY, test):
    stolpec_napovedi = stolpecY   # -2 = predzadnji
    training_set = open(file)

    x = []
    y = []
    i = 0
    for line in training_set:
        i += 1
        if i < skip_lines + 1:
            continue    #prvih n vrstic preskocimo
        else:
            line_splitted = re.split('\t', line.strip())
            x.append(line_splitted[:stolpec_napovedi])
            y.append(line_splitted[stolpec_napovedi])

    x = np.vstack(x)
    x = x.astype(float)

    print('Prebrano:', file)
    print('\tX=', x.shape)

    if not(test):
        y = np.hstack(y)
        y = y.astype(float)
        print('\ty=', y.shape)

    return x, y


def korelacija(x, y):
    xx = np.ones(x.shape[0])
    for i in range(x.shape[1]):
        c = pearsonr(x[:, i], y)
        if abs(c[0]) > 0.3:
            xx = np.column_stack((xx, x[:, i]))
    return xx


def skrci(x, y):
    #clf = ExtraTreesClassifier()
    #x_new = clf.fit(x, y).transform(x)
    #sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    #return sel.fit_transform(x)
    #return korelacija(x, y)

    return LinearSVC(C=0.01, penalty="l2", dual=True).fit_transform(x, y)


lam = 0.1
alpha = 1e-6
initials = []

def make_model(x, y, izpis=False):

    if izpis:
        print('Making model...')

    #initials = np.random.randn(x.shape[1])
    #initials = np.zeros(x.shape[1])
    #initials = np.ones(x.shape[1])
    if izpis:
        print('first')
    a, f, d = optimize.fmin_l_bfgs_b(func=cost2, x0=initials, args=(x, y, lam), fprime=grad4) #fprime=grad1,  approx_grad=True
    if izpis:
        print('second')
    a1, f1, d1 = optimize.fmin_l_bfgs_b(func=cost2, x0=initials, args=(x, y, lam), approx_grad=True) #fprime=grad1,  approx_grad=True

    b = np.count_nonzero(a)
    if izpis:
        print('f=', f)
        print('f1=', f1)
        print('funcalls', d['funcalls'])
        print('funcalls1', d1['funcalls'])
        print('iters', d['nit'])
        print('iters1', d1['nit'])
        print('warnflag', d['warnflag'])
        print('warnflag1', d1['warnflag'])
        print('gradAvg', np.mean(d['nit']))
        print('gradAvg1', np.mean(d1['nit']))
        print('NonZero:', b, 'razlika=', x.shape[1] - b)
        print('grads', d['grad'])
        print('grads1', d1['grad'])
    return a1


def cross_validate(x, y):
    print('------CV------')


    train = 6
    test = 4
    m = x.shape[0]

    #kf = KFold(m, n_folds=10)

    k = np.floor(m / train)
    #np.random.shuffle(x)
    test_set_x = x[0:k]   #testna mnozica vsebuje prvih k vrstic
    test_set_y = y[0:k]
    train_set_x = x[k:m]  #train set vsebuje vse ostale vrstice
    train_set_y = y[k:m]
    #clf = svm.SVC(kernel='linear', C=1, random_state=13)
    clf = svm.LinearSVC(loss='l2', random_state=13)
    scores = cross_validation.cross_val_score(clf, x, y, cv=3)
    #print('score', clf.score(test_set_x, test_set_y)*100)
    print(sum(scores))

    '''
    train = 9
    test = 1
    m = x.shape[0]
    k = np.floor(m / train)

    vsota = 0
    for i in range(train + test):
        np.random.shuffle(x)    #premesamo vrstice
        test_set_x = x[0:k]   #testna mnozica vsebuje prvih k vrstic
        test_set_y = y[0:k]
        train_set_x = x[k:m]  #train set vsebuje vse ostale vrstice
        train_set_y = y[k:m]
        thetas = make_model(train_set_x, train_set_y)
        napovedi = thetas.dot(test_set_x.T)
        vsota += (sum(napovedi - test_set_y)**2)

    vsota /= (train + test)
    print('CV: ', np.sqrt(vsota))
    '''


def evaluate(thetas, x, y):
    napovedi = thetas.dot(x.T)
    razlika = napovedi - y
    return sum(razlika**2)/len(razlika)


def precna_validacija(X, y, folds, alpha, lambda1):
    start = time.time()

    RMSE = 0
    for i in range(folds):
        x_train, x_test = cross_validation.train_test_split(X, test_size=1/folds, random_state=i)
        y_train, y_test = cross_validation.train_test_split(y, test_size=1/folds, random_state=i)
        theta = np.zeros(x_train.shape[1]).T
        #theta = gradient_descent_custom(x_train, y_train, theta, .0000025, 1, 12000, 0.0001, -0.0004, 0.0005)

        #theta = elastic_net_regression(x_train, y_train, alpha, lambda1)
        #theta = optimize.fmin_l_bfgs_b(cost2, approx_grad=True, x0=theta, args=(X, y))[0]
        theta = make_model(X, y)
        RMSE += evaluate(theta, x_test, y_test)
    RMSE = RMSE/folds
    print("Cross validation RMSE: ", RMSE)
    end = time.time()
    print("Čas potreben za prečno validacijo modela: ", end - start)
    print('RMSE deljen s testnimi primeri', RMSE / len(x_test))
    return RMSE


def main():
    global initials

    print('Lambda', lam)
    print('alpha', alpha)
    start = time.time()
    skip_lines = 3
    y_column = -2
    X, y = preberi('train.tab', skip_lines, y_column, False)

    #zmanjsamo seznam znacilk
    #svc = LinearSVC(C=0.01, penalty="l2", dual=True).fit(X, y)
    kbest = 500
    print('selectKBest=', kbest)
    svc = SelectKBest(f_regression, k=kbest).fit(X, y)
    X_new = svc.transform(X)
    X_new = np.column_stack((np.ones(len(X_new)), X_new))   # dodamo 1 na zacetek (x0 je vedno 1)
    initials = np.random.randn(X_new.shape[1])
    Theta = make_model(X_new, y, True)

    #gradient check
    ag = grad4(Theta, X_new, y, lam, alpha)
    ng = numerical_grad(lambda params: cost2(Theta, X_new, y, lam), Theta, 1e-4, Theta)
    print('gradient check', np.sum((ag - ng)**2))

    print('---------TESTING----------')
    Xtest = preberi('test.tab', skip_lines, y_column, True)[0]   #ne potrebujemo y, ker so ? oz. ga ni

    Xtest_new = svc.transform(Xtest)
    Xtest_new = np.column_stack((np.ones(len(Xtest_new)), Xtest_new))

    result = open('mean.txt', 'w')

    rows = []
    for x in Xtest_new:
        a = Theta.dot(x)
        if a < 0:
            a = 0
        if a > 100:
            a = 100
        rows.append(a)
        print(abs(a), file=result)
    result.close()

    #cross_validate(X_new, y)
    print('Precna...')
    precna_validacija(X_new, y, 10, 1, 1)
    end = time.time()
    print('Total time:', end - start)
    #print(sum((rows - y)**2))
    #print(sum(abs(rows - y)))

main()


