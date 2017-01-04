# coding:utf-8
# print(__doc__)
import matplotlib

matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from matplotlib.contour import ContourSet
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import Tkinter as Tk
import tkFileDialog
import ttk
import sys
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import datetime

import matplotlib.image as mpimg


class Model(object):
    def __init__(self):
        self.train = []
        self.test = []
        self.CVsize = Tk.StringVar()
        self.clf = None


class Model_SVM(object):
    # CV == 0, fit with the whole training set and display scores of all kind
    # CV != 0, choose parameters
    def __init__(self, model, parameter={"kernel": "rbf", "C": 5, "gamma": 1, "poly degree": 3, "CV_size": 0}):
        self.train = model.train
        self.test = model.test
        self.CVsize = float(parameter["CV_size"].get())
        train = np.array(self.train)
        self.X_train = train[:, :-1]
        self.y_train = train[:, -1]
        self.X_train, self.X_CV, self.y_train, self.y_CV = train_test_split(self.X_train, self.y_train,
                                                                            test_size=self.CVsize)
        if self.CVsize == 0:  #
            self.clf = SVC(kernel=parameter["kernel"].get(), C=float(parameter["C"].get()),
                           gamma=float(parameter["gamma"].get()))
        self.model = model

    def fit(self):
        self.clf.fit(self.X_train, self.y_train)

    def score(self):
        pre = self.clf.predict(self.X_train)
        truth = self.y_train
        print ("score on training set: " + str(self.clf.score(self.X_train, truth)))
        print ("f1 on training set: " + str(f1_score(truth, pre, average=None)))
        print ("AUC score on training set: " + str(roc_auc_score(truth, pre)))

    def save_results(self):
        pre = self.model.clf.predict(self.model.test)
        df = pd.DataFrame({"predict": pre})
        fileName = tkFileDialog.asksaveasfilename()
        df.to_csv(fileName)

    def crossValidation(self):
        CList = [0.01, 0.1, 1, 2, 5, 10]
        gammaList = [0, 0.1, 0.5, 1, 2, 5, 10]
        degreeList = range(10)
        bestScore = [0, 0]  # score,C
        bestF1ScoreNeg = [0, 0]
        bestF1ScorePos = [0, 0]
        # bestAUCScore = [0,0]
        for C in CList:
            self.clf = SVC(kernel="linear", C=C)
            self.clf.fit(self.X_train, self.y_train)
            pre = self.clf.predict(self.X_CV)
            truth = self.y_CV
            score = self.clf.score(self.X_CV, truth)
            if score > bestScore[0]:
                bestScore[0] = score
                bestScore[1] = C

            f1pos = f1_score(truth, pre, average=None)[1]
            if f1pos > bestF1ScorePos[0]:
                bestF1ScorePos[0] = f1pos
                bestF1ScorePos[1] = C

            f1neg = f1_score(truth, pre, average=None)[0]
            if f1neg > bestF1ScoreNeg[0]:
                bestF1ScoreNeg[0] = f1neg
                bestF1ScoreNeg[1] = C

        print ("For linear kernel:")
        print ("Best [score,C] on Cross Validation set: " + str(bestScore))
        print ("Best [f1(pos),C] on Cross Validation set: " + str(bestF1ScorePos))
        print ("Best [f1(neg),C] on Cross Validation set" + str(bestF1ScoreNeg))

    def predict(self, features):
        return self.clf.predict(features)


class Model_Adaboost(object):
    def __init__(self, model, parameter={"n_estimators": 50, "CV_size": 0}):
        self.train = model.train
        self.test = model.test
        self.CVsize = float(parameter["CV_size"].get())
        train = np.array(self.train)
        self.X_train = train[:, :-1]
        self.y_train = train[:, -1]
        self.X_train, self.X_CV, self.y_train, self.y_CV = train_test_split(self.X_train, self.y_train,
                                                                            test_size=self.CVsize)
        if self.CVsize == 0:
            self.clf = AdaBoostClassifier(n_estimators=int(parameter["n_estimators"].get()))
        self.model = model

    def fit(self):
        self.clf.fit(self.X_train, self.y_train)

    def score(self):
        pre = self.clf.predict(self.X_train)
        truth = self.y_train
        print ("score: " + str(self.clf.score(self.X_train, truth)))
        print ("f1: " + str(f1_score(truth, pre, average=None)))
        print ("AUC score: " + str(roc_auc_score(truth, pre)))

    def save_results(self):
        pre = self.model.clf.predict(self.model.test)
        df = pd.DataFrame({"predict": pre})
        fileName = tkFileDialog.asksaveasfilename()
        df.to_csv(fileName)

    def crossValidation(self):
        estimatorList = [3, 5, 7, 10, 13, 15, 20, 25, 30, 50]
        bestScore = [0, 0]  # score,n_estimator
        bestF1ScoreNeg = [0, 0]
        bestF1ScorePos = [0, 0]
        # bestAUCScore = [0,0]
        for e in estimatorList:
            self.clf = AdaBoostClassifier(n_estimators=e)
            self.clf.fit(self.X_train, self.y_train)
            pre = self.clf.predict(self.X_CV)
            truth = self.y_CV
            score = self.clf.score(self.X_CV, truth)
            if score > bestScore[0]:
                bestScore[0] = score
                bestScore[1] = e

            f1pos = f1_score(truth, pre, average=None)[1]
            if f1pos > bestF1ScorePos[0]:
                bestF1ScorePos[0] = f1pos
                bestF1ScorePos[1] = e

            f1neg = f1_score(truth, pre, average=None)[0]
            if f1neg > bestF1ScoreNeg[0]:
                bestF1ScoreNeg[0] = f1neg
                bestF1ScoreNeg[1] = e

        print ("Adaboost:")
        print ("Best [score,n_estimators] on Cross Validation set: " + str(bestScore))
        print ("Best [f1(pos),n_estimators] on Cross Validation set: " + str(bestF1ScorePos))
        print ("Best [f1(neg),n_estimators] on Cross Validation set" + str(bestF1ScoreNeg))

    def predict(self, features):
        return self.clf.predict(features)


class Model_RF(object):
    def __init__(self, model, parameter={"n_estimators": 10, "max_depth": 5, "max_features": 10, "CV_size": 0}):
        self.train = model.train
        self.test = model.test
        self.CVsize = float(parameter["CV_size"].get())
        train = np.array(self.train)
        self.X_train = train[:, :-1]
        self.y_train = train[:, -1]
        self.X_train, self.X_CV, self.y_train, self.y_CV = train_test_split(self.X_train, self.y_train,
                                                                            test_size=self.CVsize)
        if self.CVsize == 0:
            self.clf = RandomForestClassifier(n_estimators=int(parameter["n_estimators"].get()),
                                              max_features=parameter["max_features"].get(),
                                              max_depth=int(parameter["max_depth"].get()))
        self.model = model

    def fit(self):
        self.clf.fit(self.X_train, self.y_train)

    def score(self):
        pre = self.clf.predict(self.X_train)
        truth = self.y_train
        print ("score: " + str(self.clf.score(self.X_train, truth)))
        print ("f1: " + str(f1_score(truth, pre, average=None)))
        print ("AUC score: " + str(roc_auc_score(truth, pre)))

    def save_results(self):
        pre = self.model.clf.predict(self.model.test)
        df = pd.DataFrame({"predict": pre})
        fileName = tkFileDialog.asksaveasfilename()
        df.to_csv(fileName)

    def crossValidation(self):
        estimatorList = [10, 50, 100, 200, 500]
        maxFeatList = ["sqrt", "log2", None]
        bestScore = [0, 0, None]
        bestF1ScoreNeg = [0, 0, None]
        bestF1ScorePos = [0, 0, None]
        for e in estimatorList:
            for maxFeat in maxFeatList:
                self.clf = RandomForestClassifier(n_estimators=e, max_features=maxFeat)
                self.clf.fit(self.X_train, self.y_train)
                pre = self.clf.predict(self.X_CV)
                truth = self.y_CV
                score = self.clf.score(self.X_CV, truth)
                if score > bestScore[0]:
                    bestScore[0] = score
                    bestScore[1] = e
                    bestScore[2] = maxFeat
                f1pos = f1_score(truth, pre, average=None)[1]
                if f1pos > bestF1ScorePos[0]:
                    bestF1ScorePos[0] = f1pos
                    bestF1ScorePos[1] = e
                    bestF1ScorePos[2] = maxFeat
                f1neg = f1_score(truth, pre, average=None)[0]
                if f1neg > bestF1ScoreNeg[0]:
                    bestF1ScoreNeg[0] = f1neg
                    bestF1ScoreNeg[1] = e
                    bestF1ScoreNeg[2] = maxFeat

        print ("Best [score,n_estimators,max_features] on Cross Validation set: " + str(bestScore))
        print ("Best [f1(pos),n_estimators,max_features] on Cross Validation set: " + str(bestF1ScorePos))
        print ("Best [f1(neg),n_estimators,max_features] on Cross Validation set" + str(bestF1ScoreNeg))

    def predict(self, features):
        return self.clf.predict(features)


class Model_KNN(object):
    def __init__(self, model, parameter={"K": 5}):
        self.train = model.train
        self.test = model.test

        self.CVsize = float(parameter["CV_size"].get())
        train = np.array(self.train)
        self.X_train = train[:, :-1]
        self.y_train = train[:, -1]

        self.X_train, self.X_CV, self.y_train, self.y_CV = train_test_split(self.X_train, self.y_train,
                                                                            test_size=self.CVsize)
        if self.CVsize == 0:
            self.clf = KNeighborsClassifier(int(parameter["K"].get()))
        self.model = model

    def fit(self):
        self.clf.fit(self.X_train, self.y_train)

    def score(self):
        pre = self.clf.predict(self.X_train)
        truth = self.y_train
        print ("score: " + str(self.clf.score(self.X_train, truth)))
        print ("f1: " + str(f1_score(truth, pre, average=None)))
        print ("AUC score: " + str(roc_auc_score(truth, pre)))

    def save_results(self):
        pre = self.model.clf.predict(self.model.test)
        df = pd.DataFrame({"predict": pre})
        fileName = tkFileDialog.asksaveasfilename()
        df.to_csv(fileName)

    def crossValidation(self):
        kList = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        bestScore = [0, 0]  # score,k
        bestF1ScoreNeg = [0, 0]
        bestF1ScorePos = [0, 0]
        # bestAUCScore = [0,0]
        for k in kList:
            if k > self.X_train.shape[0]:
                break
            self.clf = KNeighborsClassifier(k)
            self.clf.fit(self.X_train, self.y_train)
            pre = self.clf.predict(self.X_CV)
            truth = self.y_CV
            score = self.clf.score(self.X_CV, truth)
            if score > bestScore[0]:
                bestScore[0] = score
                bestScore[1] = k

            f1pos = f1_score(truth, pre, average=None)[1]
            if f1pos > bestF1ScorePos[0]:
                bestF1ScorePos[0] = f1pos
                bestF1ScorePos[1] = k

            f1neg = f1_score(truth, pre, average=None)[0]
            if f1neg > bestF1ScoreNeg[0]:
                bestF1ScoreNeg[0] = f1neg
                bestF1ScoreNeg[1] = k

        print ("KNN:")
        print ("Best [score,K] on Cross Validation set: " + str(bestScore))
        print ("Best [f1(pos),K] on Cross Validation set: " + str(bestF1ScorePos))
        print ("Best [f1(neg),K] on Cross Validation set" + str(bestF1ScoreNeg))

    def predict(self, features):
        return self.clf.predict(features)


class Model_LR(object):
    def __init__(self, model, parameter={"multi": "ovr", "C": 1}):
        self.train = model.train
        self.test = model.test
        self.CVsize = float(parameter["CV_size"].get())
        train = np.array(self.train)
        self.X_train = train[:, :-1]
        self.y_train = train[:, -1]
        self.multi = parameter["multi"].get()
        self.X_train, self.X_CV, self.y_train, self.y_CV = train_test_split(self.X_train, self.y_train,
                                                                            test_size=self.CVsize)
        if self.CVsize == 0:
            if parameter["multi"].get() == "multinomial":
                # works only for the 'lbfgs' solver
                self.clf = LogisticRegression(C=float(parameter["C"].get()), multi_class=parameter["multi"].get(),
                                              solver='lbfgs')
            else:
                self.clf = LogisticRegression(C=float(parameter["C"].get()), multi_class=parameter["multi"].get())
        self.model = model

    def fit(self):
        self.clf.fit(self.X_train, self.y_train)

    def score(self):
        pre = self.clf.predict(self.X_train)
        truth = self.y_train
        print ("score: " + str(self.clf.score(self.X_train, truth)))

    def save_results(self):
        pre = self.model.clf.predict(self.model.test)
        df = pd.DataFrame({"predict": pre})
        fileName = tkFileDialog.asksaveasfilename()
        df.to_csv(fileName)

    def crossValidation(self):
        CList = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        bestScore = [0, 0]  # score,C
        for C in CList:
            if self.multi == "multinomial":
                # works only for the 'lbfgs' solver
                self.clf = LogisticRegression(C=C, multi_class=self.multi, solver='lbfgs')
            else:
                self.clf = LogisticRegression(C=C, multi_class=self.multi)
            self.clf.fit(self.X_train, self.y_train)
            pre = self.clf.predict(self.X_CV)
            truth = self.y_CV
            score = self.clf.score(self.X_CV, truth)
            if score > bestScore[0]:
                bestScore[0] = score
                bestScore[1] = C
        print ("Best [score,C] on Cross Validation set: " + str(bestScore))

    def predict(self, features):
        return self.clf.predict(features)


class Model_xgb(object):
    def __init__(self, model,
                 parameter={"objective": "multi:softmax", "bst:max_depth": 5, "bst:eta": 1, "silent": 1, "nthread": 4}):
        self.train = model.train
        self.test = model.test
        self.CVsize = float(parameter["CV_size"].get())
        if self.CVsize == 0:
            self.CVsize = 0.2
        self.num_round = int(parameter["num_round"].get())
        train = np.array(self.train)

        self.X_train = train[:, :-1]
        self.y_train = train[:, -1]
        self.X_train, self.X_CV, self.y_train, self.y_CV = train_test_split(self.X_train, self.y_train,
                                                                            test_size=self.CVsize)

        self.model = model

        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dtest = xgb.DMatrix(np.array(self.test))
        self.dCV = xgb.DMatrix(self.X_CV, label=self.y_CV)

        self.evallist = [(self.dCV, 'eval'), (self.dtrain, 'train')]
        # for sparse matrix
        # csr = scipy.sparse.csr_matrix((dat, (row, col)))
        # dtrain = xgb.DMatrix(csr)
        # for missing value or weight, parameter in DMatrix()
        self.plst = []
        for param in parameter:
            self.plst.append((param, parameter[param].get()))

    def fit(self):
        self.bst = xgb.train(self.plst, self.dtrain, self.num_round, self.evallist)

    def score(self):
        pass

    def save_results(self):

        pre = self.bst.predict(self.dtest)
        df = pd.DataFrame({"predict": pre})
        fileName = tkFileDialog.asksaveasfilename()
        df.to_csv(fileName)

    def crossValidation(self):
        pass

    def predict(self, features):
        return self.clf.predict(features)


def plot_confusion_matrix(f, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    f.clf()
    a = f.add_subplot(111)

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    r = a.imshow(cm, interpolation='nearest', cmap=cmap)
    a.set_title(title)
    f.colorbar(mappable=r)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        # print('Confusion matrix, without normalization')
        pass

    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        a.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

   # f.tight_layout()
    a.set_ylabel('True label')
    a.set_xlabel('Predicted label')

    f.subplots_adjust(left=0.1,right=0.9,bottom=0.15,top=0.85)


class Controller(object):
    def __init__(self, model):
        self.model = model
        self.modelType = Tk.IntVar()
        self.parameter = {}
        self.isShown = False
        self.frame = Tk.Toplevel()
        self.frame.wm_title("Parameter")

    def showFrameHelper(self):
        if self.isShown == False:
            self.isShown = True
            self.showFrame()
        else:
            self.param_group.pack_forget()
            self.showFrame()

    def showFrame(self):

        self.parameter["CV_size"] = Tk.StringVar()
        self.param_group = Tk.Frame(self.frame)
        if self.modelType.get() == 0:
            self.parameter["kernel"] = Tk.StringVar()
            self.parameter["C"] = Tk.StringVar()
            self.parameter["gamma"] = Tk.StringVar()
            self.parameter["degree"] = Tk.StringVar()
            Tk.Radiobutton(self.param_group, text="linear", variable=self.parameter["kernel"],
                           value="linear").pack(anchor=Tk.W)
            Tk.Radiobutton(self.param_group, text="rbf", variable=self.parameter["kernel"],
                           value="rbf").pack(anchor=Tk.W)
            Tk.Radiobutton(self.param_group, text="poly", variable=self.parameter["kernel"],
                           value="poly").pack(anchor=Tk.W)
            Tk.Label(self.param_group, text="C").pack()
            Tk.Entry(self.param_group, textvariable=self.parameter["C"]).pack()
            Tk.Label(self.param_group, text="gamma").pack()
            Tk.Entry(self.param_group, textvariable=self.parameter["gamma"]).pack()
            Tk.Label(self.param_group, text="degree").pack()
            Tk.Entry(self.param_group, textvariable=self.parameter["degree"]).pack()

        if self.modelType.get() == 1:
            self.parameter["n_estimators"] = Tk.StringVar()
            Tk.Label(self.param_group, text="n_estimators").pack()
            Tk.Entry(self.param_group, textvariable=self.parameter["n_estimators"]).pack()

        if self.modelType.get() == 2:
            self.parameter["n_estimators"] = Tk.StringVar()
            self.parameter["max_depth"] = Tk.StringVar()
            self.parameter["max_features"] = Tk.StringVar()
            Tk.Label(self.param_group, text="n_estimators").pack()
            Tk.Entry(self.param_group, textvariable=self.parameter["n_estimators"]).pack()
            Tk.Label(self.param_group, text="max_depth").pack()
            Tk.Entry(self.param_group, textvariable=self.parameter["max_depth"]).pack()
            maxFeat_group = Tk.Frame(self.param_group)
            Tk.Label(maxFeat_group, text="max_features").pack()
            Tk.Radiobutton(maxFeat_group, text="sqrt", variable=self.parameter["max_features"],
                           value="sqrt").pack(side=Tk.LEFT)
            Tk.Radiobutton(maxFeat_group, text="log2", variable=self.parameter["max_features"],
                           value="log2").pack(side=Tk.LEFT)
            maxFeat_group.pack()

        if self.modelType.get() == 3:
            self.parameter["K"] = Tk.StringVar()
            Tk.Label(self.param_group, text="K").pack()
            Tk.Entry(self.param_group, textvariable=self.parameter["K"]).pack()

        if self.modelType.get() == 4:
            self.parameter["C"] = Tk.StringVar()
            self.parameter["multi"] = Tk.StringVar()
            Tk.Radiobutton(self.param_group, text="one vs all", variable=self.parameter["multi"],
                           value="ovr").pack(anchor=Tk.W)
            Tk.Radiobutton(self.param_group, text="multinomial", variable=self.parameter["multi"],
                           value="multinomial").pack(anchor=Tk.W)

            Tk.Label(self.param_group, text="C").pack()
            Tk.Entry(self.param_group, textvariable=self.parameter["C"]).pack()

        if self.modelType.get() == 5:
            self.parameter["max_depth"] = Tk.StringVar()
            self.parameter["eta"] = Tk.StringVar()
            self.parameter["silent"] = Tk.StringVar()
            self.parameter["objective"] = Tk.StringVar()
            self.parameter["num_round"] = Tk.StringVar()
            Tk.Label(self.param_group, text="objective").pack()
            Tk.Radiobutton(self.param_group, text="multi:softmax", variable=self.parameter["objective"],
                           value="multi:softmax").pack(anchor=Tk.W)
            Tk.Radiobutton(self.param_group, text="reg:logistic", variable=self.parameter["objective"],
                           value="reg:logistic").pack(anchor=Tk.W)
            Tk.Radiobutton(self.param_group, text="binary:logistic", variable=self.parameter["objective"],
                           value="binary:logistic").pack(anchor=Tk.W)
            Tk.Label(self.param_group, text="max_depth").pack()
            Tk.Entry(self.param_group, textvariable=self.parameter["max_depth"]).pack()
            Tk.Label(self.param_group, text="eta").pack()
            Tk.Entry(self.param_group, textvariable=self.parameter["eta"]).pack()
            Tk.Label(self.param_group, text="num_round").pack()
            Tk.Entry(self.param_group, textvariable=self.parameter["num_round"]).pack()
            Tk.Label(self.param_group, text="silent").pack()
            Tk.Entry(self.param_group, textvariable=self.parameter["silent"]).pack()

        Tk.Label(self.param_group, text="Cross Validation Size").pack()
        Tk.Label(self.param_group, text="Set it to 0 if no need").pack()
        cvSizeEntry = Tk.Entry(self.param_group, textvariable=self.parameter["CV_size"])
        cvSizeEntry.insert(0, 0)
        cvSizeEntry.pack()
        self.param_group.pack(side=Tk.LEFT)

    def fit(self):
        model_map = {0: "SVM", 1: "Adaboost", 2: "Random Forest", 3: "KNN", 4: "Logistic Regression", 5: "Xgboost"}

        if self.modelType.get() == 0:
            self.model = Model_SVM(self.model, self.parameter)

        elif self.modelType.get() == 1:
            self.model = Model_Adaboost(self.model, self.parameter)

        elif self.modelType.get() == 2:
            self.model = Model_RF(self.model, self.parameter)

        elif self.modelType.get() == 3:
            self.model = Model_KNN(self.model, self.parameter)

        elif self.modelType.get() == 4:
            self.model = Model_LR(self.model, self.parameter)

        elif self.modelType.get() == 5:
            self.model = Model_xgb(self.model, self.parameter)

        if float(self.parameter["CV_size"].get()) == 0:
            self.model.fit()
            self.model.score()

        else:
            self.model.crossValidation()

    def show(self):

    #    print(self.model.example['speed'])
    #    print(self.model.example['occupancy'])
    #    print(self.model.example.loc[0, 'incident_time'])
    #    print(int(self.model.example.loc[0, 'link']))

        m = self.model.example.as_matrix()

        times = m[:,0]
        seqs = np.linspace(1, len(times), len(times))
        speed = m[:,1]
        occupancy = m[:,2]

        self.center_graph1.cla();
        self.center_graph1.set_title('Speed')
        self.center_graph1.plot(seqs, speed)

        self.center_graph2.cla();
        self.center_graph2.set_title('Occupancy')
        self.center_graph2.set_ylim([0, 100])
        self.center_graph2.plot(seqs, occupancy)
        self.center_canvas.show()


    def calculate(self):

         class_names = ['incident', 'No incident']

         cmatrix = confusion_matrix(np.array(self.model.list['incident']), np.array(self.model.list['predict']))


         plot_confusion_matrix(self.right_figure, cmatrix, classes=class_names, normalize=True,
                               title='result normalized confusion matrix')
         rate = 100 * float(cmatrix[0, 0]) / (cmatrix[0, 0] + cmatrix[0, 1])
         print(u'准确率: ' + str(rate) + '%')

         self.model.list['detect_time'] = self.model.list['predict_time'].apply(lambda x: int(x[-2:])) \
                                          - self.model.list['Start_Time'].apply(lambda x: int(x[-2:]))
         detect_time = self.model.list['detect_time'].mean() * 60
         print(u'平均检测时间:' + str(detect_time) + 's')


         self.accuracy_label.config(text = u'准确率: ' + str(rate) + '%')
         self.avg_detection_time_label.config(text = u'平均检测时间:' + str(detect_time) + 's')


         self.right_canvas.show()


 #   def calculate(self):
 #       class_names = ['incident', 'No incident']
 #       cmatrix = confusion_matrix(np.array(self.model.list['incident']), np.array(self.model.list['predict']))
 #       plot_confusion_matrix(cmatrix, classes=class_names, normalize=True,
 #                             title='result normalized confusion matrix')
 #       plt.show()


    def save_results(self):
        self.model.save_results()
        # pre = self.model.clf.predict(self.model.test)
        # df = pd.DataFrame({"predict":pre})
        # fileName = tkFileDialog.asksaveasfilename()
        # df.to_csv(fileName)

    def loadTrainData(self):
        fileName = tkFileDialog.askopenfilename()
        self.model.train = pd.read_csv(str(fileName))
        print("Train data has been loaded")
        print("Shape: " + str(self.model.train.shape))

    def loadTestData(self):
        fileName = tkFileDialog.askopenfilename()
        self.model.test = pd.read_csv(str(fileName))
        print("Test data has been loaded")
        print("Shape: " + str(self.model.test.shape))

    def loadExample(self):
        fileName = tkFileDialog.askopenfilename()
        self.model.example = pd.read_csv(str(fileName))
        print("Example data has been loaded")
        print("Shape: " + str(self.model.example.shape))

        m = self.model.example.as_matrix()
        self.result_time = m[39][0]
        self.Start_Time = m[0][3]
        self.result_link = m[0][4]

        items = m[:,1:2]
        # print(len(items))

        for i in range(len(items)-40):
            temp = m[i:(i+40),1:2]
            temp2 = m[i:(i+40),2:3]

            feature_temp = [
                            np.mean(temp, axis=0)[0]
                           ,np.median(temp, axis=0)[0]
                           ,np.max(temp, axis=0)[0]
                           ,np.min(temp, axis=0)[0]
                           ,np.var(temp, axis=0)[0]
                           ,(np.max(temp, axis=0)[0]-np.min(temp, axis=0)[0])
                           ,pd.DataFrame(temp).skew(axis=0)[0]

                           ,np.mean(temp2, axis=0)[0]
                           ,np.median(temp2, axis=0)[0]
                           ,np.max(temp2, axis=0)[0]
                           ,np.min(temp2, axis=0)[0]
                           ,np.var(temp2, axis=0)[0]
                           ,(np.max(temp2, axis=0)[0]-np.min(temp, axis=0)[0])
                           ,pd.DataFrame(temp2).skew(axis=0)[0]
                            ]
            # print(i,feature_temp)

            if i==0:
                self.feature = [feature_temp]
            else:
                self.feature.append(feature_temp)

        pd.DataFrame(self.feature).to_csv('./temp/feature.csv',index=False)

    def predict(self):

        for i in range(len(self.feature)):

            result = self.model.predict(np.array([self.feature[i]]))
            print(i,np.array([self.feature[i]]),result[0])

            if int(result[0])==1:
                delta = datetime.timedelta(minutes=i)
                d = datetime.datetime.strptime(self.result_time, '%Y/%m/%d %H:%M')
                d += delta
                print(d)
                # self.result_time_label.config(text = 'Time: ' + self.result_time)
                self.result_time_label.config(text = 'Time: ' + str(d))
                self.result_link_label.config(text = 'Link: ' + str(int(self.result_link)))
                self.result_prediction_label.config(text = 'Prediction: ' + str(int(result[0])))

                temp = pd.read_csv('./listtemp.csv',usecols=[0])
                flag = 1
                if self.Start_Time=='0':
                    flag = 0
                line = '{0},{1},{2},{3},{4},I10-E,{5}'.\
                    format(len(temp)+1,flag,self.Start_Time,int(result[0]),self.result_time,int(self.result_link))
                incidentlist = open('./listtemp.csv','a')
                incidentlist.write('\n')
                incidentlist.write(line)
                break


    def loadList(self):
        fileName = tkFileDialog.askopenfilename()
        self.model.list = pd.read_csv(str(fileName))
        print("List data has been loaded")
        print("Shape: " + str(self.model.list.shape))


    def setCenterCanvas(self, canvas, figure):
        self.center_canvas = canvas
        self.center_figure = figure
        self.center_graph1 = self.center_figure.add_subplot(211)
        self.center_graph2 = self.center_figure.add_subplot(212)
        self.center_canvas.show()

    def setRightCanvas(self, canvas, figure, accuracy_label, avg_detection_time_label):
        self.right_canvas = canvas
        self.right_figure = figure
        self.right_canvas.show()
        self.accuracy_label = accuracy_label
        self.avg_detection_time_label = avg_detection_time_label

    def setResultLabels(self, result_prediction_label, result_time_label, result_link_label):
        self.result_prediction_label = result_prediction_label
        self.result_time_label = result_time_label
        self.result_link_label = result_link_label

class View(object):
    def __init__(self, root, controller):
        self.controllbar = ControllBar(root, controller)


class ControllBar(object):

    def __init__(self, root, controller):
        fm = Tk.Frame(root)

        top_panel = Tk.Frame(fm)
        top_panel.pack(side=Tk.TOP, fill=Tk.X)

        result_labels_container = Tk.Frame(top_panel)
        label1 = Tk.Label(result_labels_container, text='Prediction')
        label1.pack()
        label2 = Tk.Label(result_labels_container, text='Time')
        label2.pack()
        label3 = Tk.Label(result_labels_container, text='Link')
        label3.pack()
        label4 = Tk.Label(result_labels_container, text='Accuracy')
        label4.pack(fill=Tk.X)
        label5 = Tk.Label(result_labels_container, text='Avg detect time')
        label5.pack(fill=Tk.X)
        result_labels_container.pack(side=Tk.RIGHT, padx=200)
        controller.setResultLabels(label1, label2, label3)


        model_group = Tk.Frame(top_panel)
        # self.box = ttk.Combobox(model_group, textvariable = Tk.StringVar(), values = ["SVM","Adaboost"])
        # self.box.bind("SVM",controller.showFrameHelper)
        # self.box.pack()

        Tk.Radiobutton(model_group, text="SVM(0/1)", variable=controller.modelType,
                       value=0, command=controller.showFrameHelper).pack(anchor=Tk.W)
        Tk.Radiobutton(model_group, text="Adaboost(0/1)", variable=controller.modelType,
                       value=1, command=controller.showFrameHelper).pack(anchor=Tk.W)
        Tk.Radiobutton(model_group, text="Random Forest(0/1)", variable=controller.modelType,
                       value=2, command=controller.showFrameHelper).pack(anchor=Tk.W)
        Tk.Radiobutton(model_group, text="KNN(0/1)", variable=controller.modelType,
                       value=3, command=controller.showFrameHelper).pack(anchor=Tk.W)
        Tk.Radiobutton(model_group, text="Logistic Regression(Reg)", variable=controller.modelType,
                       value=4, command=controller.showFrameHelper).pack(anchor=Tk.W)
        Tk.Radiobutton(model_group, text="Xgboost", variable=controller.modelType,
                       value=5, command=controller.showFrameHelper).pack(anchor=Tk.W)

        model_group.pack(side=Tk.LEFT, padx=10)

        figure = Figure(figsize=(10,3), facecolor='white')
        image = figure.add_subplot(111)
        bottom_banner_container = FigureCanvasTkAgg(figure, master=fm)
        image.set_axis_off()
        image.imshow(mpimg.imread("./figure/road.png"))
        bottom_banner_container.get_tk_widget().pack(side=Tk.BOTTOM, fill=Tk.X)
        figure.subplots_adjust(hspace=0,wspace=0,left=0,top=1,right=1,bottom=0)


        file_group = Tk.Frame(fm)

        Tk.Button(file_group, text="example",
                  command=controller.loadExample).pack(anchor=Tk.W)
        Tk.Button(file_group, text="list",
                  command=controller.loadList).pack(anchor=Tk.W)
        Tk.Button(file_group, text="train",
                  command=controller.loadTrainData).pack(anchor=Tk.W)
        Tk.Button(file_group, text="test",
                  command=controller.loadTestData).pack(anchor=Tk.W)
        file_group.pack(side=Tk.LEFT, padx=10)


        right_chart_container = Tk.Frame(fm)
        figure = Figure(figsize=(6,6))
        canvas = FigureCanvasTkAgg(figure, master=right_chart_container)
        canvas.get_tk_widget().pack(fill=Tk.BOTH, padx=10, pady=10)
        right_chart_container.pack(side=Tk.RIGHT)
        controller.setRightCanvas(canvas, figure, label4, label5)

        output_group = Tk.Frame(fm)

        Tk.Button(output_group, text='Show', width=5, command=controller.show).pack()
        Tk.Button(output_group, text='Fit', width=5, command=controller.fit).pack()
        Tk.Button(output_group, text='Save Results', width=10, command=controller.save_results).pack()
        Tk.Button(output_group, text='Predict', width=10, command=controller.predict).pack()
        Tk.Button(output_group, text='Calculate', width=10, command=controller.calculate).pack()

        output_group.pack(side=Tk.RIGHT)

        center_chart_container = Tk.Frame(fm)
        figure = Figure(figsize=(8,6))
        canvas = FigureCanvasTkAgg(figure, master=center_chart_container)
        canvas.get_tk_widget().pack(fill=Tk.BOTH)
        center_chart_container.pack(fill=Tk.BOTH)
        controller.setCenterCanvas(canvas, figure)


        fm.pack(fill=Tk.BOTH)


def main(argv):
    root = Tk.Tk()
    model = Model()
    controller = Controller(model)
    root.wm_title("Incident Detect")
    view = View(root, controller)
    Tk.mainloop()


if __name__ == "__main__":
    main(sys.argv)
