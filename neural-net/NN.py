__author__ = 'Sandi'

import numpy as np
from scipy import optimize as opt
import time
import Orange
from sklearn.cross_validation import train_test_split
import scipy as sp
import scipy.optimize
from sklearn.metrics import log_loss
from sklearn import preprocessing as pre
from Orange.data import ContinuousVariable
from NeuralNet import NeuralNet
from sklearn.cross_validation import ShuffleSplit, cross_val_score
from sklearn.linear_model import Ridge
import datetime


def add_ones(x):
    if len(x.shape) == 1:
        return np.hstack((1, x))
    else:
        return np.column_stack((np.ones(len(x)), x))

def g(z):
    return 1/(1+np.exp(-z))

def t(z):
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

class NN(Orange.classification.Learner):
    def __init__(self, arch, lambda_=1, soft=False, act=g, dropout=False):
        super().__init__()
        self.arch = arch
        self.lambda_ = lambda_
        self.name = "Neural Network"
        self.layers = len(self.arch)
        self.f = act

        self.softmax = soft
        if soft:
            self.cost = self.Jsoft
        else:
            self.cost = self.J

        self.dout = dropout
        if dropout:
            self.dropout = [0.2] * (len(arch) - 1)

        np.random.seed(42)

        self.theta_shape = np.array([(arch[i]+1, arch[i+1]) for i in range(len(arch)-1)])
        ind = np.array([s1*s2 for s1, s2 in self.theta_shape])
        self.theta_ind = np.cumsum(ind[:-1])
        self.theta_len = sum(ind)

    def init_thetas(self, epsilon=1):
        return np.random.rand(self.theta_len) * 2 * epsilon - epsilon

    def shape_thetas(self, thetas):
        t = np.split(thetas, self.theta_ind)
        return [t[i].reshape(shape) for i, shape in enumerate(self.theta_shape)]

    def h(self, a, thetas):
        """feed forward, prediction"""
        thetas = self.shape_thetas(thetas)
        for i, theta in enumerate(thetas):
            if i == len(thetas)-1 and self.softmax:  #zadnja iteracija
                a = self.softmaxFunc(add_ones(a), theta)
            else:
                a = self.f(add_ones(a).dot(theta))
        #a = self.softmax(add_ones(a), thetas[-1])
        return a

    def J(self, thetas):
        #tmp = self.h(self.X, thetas) + eps
        #a = -1/self.m * np.sum(self.y * np.log(tmp) + ((1-self.y) * np.log(1 - tmp)))
        #return a
        #return 0.5 * np.sum((self.h(self.X, thetas) - self.y)**2)/self.m # + self.lambda_/2 * np.sum(thetas**2)
        a = 0.5 * np.sum((self.h(self.X, thetas) - self.y)**2) + self.lambda_/2 * np.sum(thetas**2)
        return a/self.m

    def Jsoft1(self, thetas):
        eps = 1e-15
        a = -(np.sum(self.y * np.log(self.h(self.X, thetas) + eps))) + self.lambda_/2 * np.sum(thetas**2)
        return a/self.m

    def Jsoft(self, thetas):
        J = - np.sum(self.y * np.log(self.h(self.X, thetas) + 1e-15))
        return (J + (self.lambda_ /2.0) * np.sum(thetas**2)) / self.m

    def softmaxFunc(self, X, theta):
        M = X.dot(theta)
        P = np.exp(M - np.max(M, axis=1)[:, None])
        P /= np.sum(P, axis=1)[:, None]
        return P

    def grad_approx(self, thetas, e=1e-1):
        return np.array([(self.cost(thetas+eps) - self.cost(thetas-eps))/(2*e) for eps in np.identity(len(thetas)) * e])

    def backprop(self, thetas):
        thetas1 = self.shape_thetas(thetas)
        thetas = self.shape_thetas(thetas)

        #print(self.dropout)

        if self.dout:
            dropout_mask = []
            for i in range(len(thetas)):
                if self.dropout is None or self.dropout[0] < 1e-7:
                    dropout_mask.append(1)
                else:
                    dropout_mask.append(np.random.binomial(1, 1 - self.dropout[i], (self.m, self.arch[i])))

        #step 1 izracun aktivacij
        act = [self.X]
        a = self.X
        for i, theta in enumerate(thetas):
            a = self.f(add_ones(a).dot(theta))
            if self.dout and i != len(thetas)-1:
                a *= dropout_mask[i+1]
            act.append(a)
        if self.softmax:
            act[-1] = self.softmaxFunc(add_ones(act[-2]), thetas[-1])

        #step 2 zadnja napaka
        napake = [None] * len(self.arch)
        # napaka na zadnjem nivoju L
        if not self.softmax:
            napake[-1] = -(self.y - act[-1]) * (act[-1] * (1 - act[-1]))
        else:
            napake[-1] = -(self.y - act[-1])

        #step 3 ostale napake
        for i in range(len(self.arch)-2, 0, -1):
            napake[i] = napake[i+1].dot(thetas[i][1:,:].T) * (act[i]*(1 - act[i]))

        #step 4
        for i in range(len(self.arch)-1):
            if self.dout:
                tmp = napake[i+1].T.dot(act[i] * dropout_mask[i]).T + (self.lambda_ * thetas[i][1:, :])
            else:
                tmp = napake[i+1].T.dot(act[i]).T + (self.lambda_ * thetas[i][1:, :])
            err = np.sum(napake[i+1], axis=0)
            thetas1[i] = np.vstack((err, tmp)) / self.m

        x = np.hstack([i.ravel() for i in thetas1])
        return x

    def fit(self, X, y, W=None):
        self.X, self.y = X, y
        self.m = self.X.shape[0]
        thetas = self.init_thetas()
        thetas, fmin, info = opt.fmin_l_bfgs_b(self.cost, thetas, self.backprop, factr=1)
        #if info['warnflag']!=0:
        #    print(info)
        self.thetas = thetas
        return self

    def predict(self, X):
        return self.h(X, self.thetas)

    def test(self, a):
        thetas = np.array([-30, 10, 20, -20, 20, -20, -10, 20, 20])
        print(self.h(a, thetas))

    def get_params(self, deep = False):
        '''used for CV'''
        return {'lambda_': self.lambda_, 'arch': self.arch}
    def predict_proba(self, X):
        '''used for CV'''
        return self.h(X, self.thetas)


def main1():
    global vsak

    #first test
    X = np.array([[2,3,4],[1,1,2],[4,3,1],[5,6,2]])
    Y = np.array([0, 1, 0, 1])

    #real
    vsak = 1
    X = readTrain()
    X = np.array(X).astype(float)
    y = X[:, -1].copy() #zadnji stolpec je y
    y -= 1
    X = np.delete(X, -1, axis=1)    #izbrisemo zadnji stolpec
    X = np.delete(X, 0, axis=1)    #izbrisemo idje
    Xtest = readTest()
    Xtest = np.array(Xtest).astype(float)
    Xtest = np.delete(Xtest, 0, axis=1)    #izbrisemo idje
    print('Reading done.')
    Y=y

    #iris
    #data = Orange.data.Table('iris')
    #X = data.X
    #Y = data.Y

    classes = count_class(Y)[0]
    Y = np.eye(classes)[Y.astype(int)]

    print(X.shape, classes)

    #reduced
    ann = NN((X.shape[1], 10, 10, classes), lambda_=0.0001, soft=False, dropout=True)
    make(ann, X, Y, X)

    #real
    #ann = NN((X.shape[1], 20, 20, classes), lambda_=0.0003, soft=False, dropout=False)
    #make(ann, X, Y, Xtest)

    #iris
    #nn = NN((X.shape[1], 5, 5, classes), lambda_=0.0001, soft=False)
    #nn1 = NN((X.shape[1], 10, 10, classes), lambda_=0.0001, soft=False)
    #nn2 = NN((X.shape[1], 5, 5, classes), lambda_=0.0001, soft=True)
    #nn3 = NN((X.shape[1], 10, 10, classes), lambda_=0.0001, soft=True)
    #make(nn, X, Y, Xtest)
    #learners = [nn, nn1, nn2, nn3]

    #newY, meanY = stacking(learners, X, X, Y, k=5)
    #CV3(X, newY, nn)

    #CV3(X, meanY, nn)

    #sum test
    #print(np.sum(res, axis=1))
    #print(np.sum(res1, axis=1))

    #res = Orange.evaluation.CrossValidation(data, [ann], k=2)
    #print(Orange.evaluation.AUC(res))
    return

def make(nn, X, Y, Xte):
    print(nn.arch)
    start = time.time()
    #print(nn.theta_shape)
    print('# of thetas', nn.theta_len)
    print('Lambda', nn.lambda_)
    print('Softmax', nn.softmax)
    print('Dropout', nn.dout)

    model = nn.fit(X, Y)
    res1 = model.predict(Xte)
    #print(res1)

    #sum test
    print('sumtest', np.sum(res1, axis=1))

    cas = time.time() - start
    print('Time:', cas)

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%H-%M-%S_%d-%m-%Y')
    write(res1, 'NN_' + str(round(cas, 4)) + '_' +str(X.shape[0]) + '_' + str(nn.arch) + '_' + str(st))

    try:
        print('logloss', llfun(Y, res1))
    except BaseException:
        print('No logloss')

    print('Gradient check')
    test = nn.init_thetas()
    print(np.sum((nn.grad_approx(test)-nn.backprop(test))**2))

    print('CV')
    st = time.time()
    CV3(X, Y, nn)
    print('CV time:', time.time()-st)


def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = np.sum(act*sp.log(pred) + sp.subtract(1, act)*sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0/len(act)
    return ll

def write(X, name):
    with open('uniform.csv', 'r') as f:
        for line in f:
            cols = line
            break
    with open(name + '.csv', 'wt') as out:
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
    #file = 'train.csv'
    file = 'data4_reduced.csv'
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
    i= 0
    with open(file, 'r') as f:
        for _ in range(1):
            next(f)
        for line in f:
            splitted = line.split(',')
            x.append(splitted)
    return x

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

def CV2(X, Y, learner, k=5):
    scores = []
    for i in range(k):
        print("{}\rI=", i+1, end=' ')
        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=1/k, random_state=i)
        preds = []
        learner.fit(Xtr, Ytr)
        y_pred = learner.predict(Xte)
        preds.append(y_pred)
        preds = np.array(preds).T
        sc = llfun(Yte, preds)
        print(sc)
        scores.append(sc)
    print('score:', np.mean(scores))

def CV3(X, Y, nn):
    cv_split = ShuffleSplit(Y.shape[0], n_iter=3, test_size=0.2, random_state=42)
    sc = cross_val_score(nn, X, Y,cv=cv_split, scoring = 'log_loss', n_jobs=-1)
    print(-np.mean(sc))

def zgradi_domeno(X, classes):
    seznam_znacilk = [ContinuousVariable('feat'+str(x)) for x in range(1, X.shape[1]+1)]
    class_var = Orange.data.DiscreteVariable("class", values=[x for x in range(classes)])
    d = Orange.data.Domain(seznam_znacilk, class_var)
    return d

def stacking(learners, Xtrain, Xtest, Y, k=6):
    rows = Xtrain.shape[0]
    fold = int(rows / k)
    lastIndexOfTrain = fold*(k-1)

    learnerPredictions = [[] for _ in range(len(learners))] # seznam ki vsebuje 21 stolpicne arraye (kolikor je learnerjev)
    for i in range(k):
        Xtr = Xtrain[0:lastIndexOfTrain]
        Ytr = Y[0:lastIndexOfTrain]
        Xte = Xtrain[lastIndexOfTrain:]

        Xtrain = np.roll(Xtrain, fold, 0)
        Y = np.roll(Y, fold, 0)

        print("I =", i+1, end=' ')
        for idx, learner in enumerate(learners):
            foldpreds = []
            learner.fit(Xtr, Ytr)
            y_pred = learner.predict(Xte)
            foldpreds.append(y_pred)
            foldpreds = np.array(foldpreds).T
            learnerPredictions[idx].append(foldpreds)
    for id in range(len(learnerPredictions)):
        learnerPredictions[id] = learnerPredictions[id][::-1]

    trainpreds = []
    for learn in learners:
        preds = []
        learn.fit(Xtrain, Y)
        y_pred = learn.predict(Xtest)
        preds.append(y_pred)
        preds = np.array(preds).T
        trainpreds.append(preds)

    try:
        preds = []
        avgs = []
        #rig = Ridge(alpha=5, normalize=True)
        #Xtr = makeXtrain(learnerPredictions, i)
        Xte = np.array([r[:, i] for r in trainpreds]).T
        avgs.append(np.average(Xte, axis=1))
        #print('shapes:', y.shape, Xtr.shape, Xte.shape)
        #rig = learners[0].fit(Xtr, Y)
        #preds.append(rig.predict(Xte))
    except BaseException as e:
        print('EXC', e)
        print('y:', Y)
        print('Xtr', Xtr)
        print('Xte', Xte)
        print('i', i)
        print('preds', preds)
        raise
    meanY = np.array(avgs).T
    #newY = np.array(preds).T
    newY = None
    #print('SHAPE', newY.shape, meanY.shape)
    # CV(Xtrain, newY, rig)
    return newY, meanY

def makeXtrain(preds, i):
    a = np.array([])
    for x in range(len(preds)): #za vsak learner
        s = np.array([r[:, i] for r in preds[x]])   # za i-ti stolpec, x-ti learner
        s = s.flatten()
        if x == 0:
            a = np.hstack((a,s))
        else:
            a = np.vstack((a,s))
    return a.T

vsak = 100

main1()