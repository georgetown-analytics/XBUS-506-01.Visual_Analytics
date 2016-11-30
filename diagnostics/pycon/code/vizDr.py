#!/usr/bin/env python
# vizDr.py
#
#
# Title:        Visualization tools for machine learning
# Version:      1.0
# Author:       Rebecca Bilbro
# Date:         5/9/16
# Organization: District Data Labs

#############################################################################
# Imports
#############################################################################
from __future__ import print_function

import os
import zipfile
import requests
import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from pandas.tools.plotting import radviz
from pandas.tools.plotting import parallel_coordinates

from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize

from sklearn import cross_validation as cv

from sklearn.metrics import auc
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error as mse

from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import validation_curve

from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsClassifier


#############################################################################
# Get the Data
#############################################################################

sns.set_style("whitegrid")
ddl_heat = ["#DBDBDB","#DCD5CC","#DCCEBE","#DDC8AF","#DEC2A0","#DEBB91","#DFB583","#DFAE74",\
            "#E0A865","#E1A256","#E19B48","#E29539"]
ddlheatmap = colors.ListedColormap(ddl_heat)

ddl_ocean = ["#9BDAF9","#91D4F8","#87CDF6","#7CC7F5","#72C1F3","#68BAF2","#5EB4F0","#54ADEF",\
             "#4AA7ED","#3FA1EC","#359AEA","#2B94E9","#278BDD","#2382D1","#1F7AC5","#1B71B9",\
             "#1768AD","#145FA2","#105696","#0C4D8A","#08457E","#043C72","#003366"]
ddloceanmap = colors.ListedColormap(ddl_ocean)

OCCUPANCY = "http://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip"
CREDIT    = "http://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
CONCRETE  = "http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"

def download_data(url, path='data'):
    if not os.path.exists(path):
        os.mkdir(path)

    response = requests.get(url)
    name = os.path.basename(url)
    with open(os.path.join(path, name), 'w') as f:
        f.write(response.content)


#############################################################################
# Visual feature analysis tools
#############################################################################

def box_viz(df):
    ax = sns.boxplot(df)
    plt.xticks(rotation=60)
    plt.show()

def hist_viz(df,feature):
    ax = sns.distplot(df[feature])
    plt.xlabel(feature)
    plt.show()

def splom_viz(df,labels=None):
    ax = sns.pairplot(df, hue=labels, diag_kind="kde", size=2)
    plt.show()

def pcoord_viz(df, labels):
    fig = parallel_coordinates(df, labels, color=sns.color_palette())
    plt.show()

def rad_viz(df,labels):
    fig = radviz(df, labels, color=sns.color_palette())
    plt.show()

def joint_viz(feat1,feat2,df):
    ax = sns.jointplot(feat1, feat2, data=df, kind='reg', size=5)
    plt.xticks(rotation=60)
    plt.show()

#############################################################################
# Model exploration tools
#############################################################################

def classify(attributes, targets, model):
    """
    Executes classification using the specified model and returns
    a classification report.
    """
    # Split data into 'test' and 'train' for cross validation
    splits = cv.train_test_split(attributes, targets, test_size=0.2)
    X_train, X_test, y_train, y_test = splits

    model.fit(X_train, y_train)
    y_true = y_test
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_true, y_pred))

def regress(attributes, targets, model):
    # Split data into 'test' and 'train' for cross validation
    splits = cv.train_test_split(attributes, targets, test_size=0.2)
    X_train, X_test, y_train, y_test = splits

    model.fit(X_train, y_train)
    y_true = y_test
    y_pred = model.predict(X_test)
    print("Mean squared error = {:0.3f}".format(mse(y_true, y_pred)))
    print("R2 score = {:0.3f}".format(r2_score(y_true, y_pred)))

def get_preds(attributes, targets, model):
    """
    Executes classification or regression using the specified model
    and returns expected and predicted values.
    Useful for comparison plotting.
    """
    splits = cv.train_test_split(attributes, targets, test_size=0.2)
    X_train, X_test, y_train, y_test = splits

    model.fit(X_train, y_train)
    y_true = y_test
    y_pred = model.predict(X_test)
    return (y_true,y_pred)


#############################################################################
# Visual model evaluation tools
#############################################################################

def plot_classification_report(cr, title='Classification report', cmap=ddlheatmap):
    lines = cr.split('\n')
    classes = []
    matrix = []

    for line in lines[2:(len(lines)-3)]:
        s = line.split()
        classes.append(s[0])
        value = [float(x) for x in s[1: len(s) - 1]]
        matrix.append(value)

    fig, ax = plt.subplots(1)

    for column in range(len(matrix)+1):
        for row in range(len(classes)):
            txt = matrix[row][column]
            ax.text(column,row,matrix[row][column],va='center',ha='center')

    fig = plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(len(classes)+1)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    plt.show()

def roc_viz(y, yhat, model):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y,yhat)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic for %s' % model)
    plt.plot(false_positive_rate, true_positive_rate, 'blue', label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'m--')
    plt.xlim([0,1])
    plt.ylim([0,1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def roc_compare_two(ys, yhats, mods):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    for y, yhat, m, ax in ((ys[0], yhats[0], mods[0], ax1), (ys[1], yhats[1], mods[1], ax2)):
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y,yhat)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        ax.set_title('ROC for %s' % m)
        ax.plot(false_positive_rate, true_positive_rate, c="#2B94E9", label='AUC = %0.2f'% roc_auc)
        ax.legend(loc='lower right')
        ax.plot([0,1],[0,1],'m--',c="#666666")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()

def regr_error_viz(model,features,labels):
    predicted = cv.cross_val_predict(model, features, labels, cv=12)
    fig, ax = plt.subplots()
    ax.scatter(labels, predicted)
    ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

def error_compare_three(mods,X,y):
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    for mod, ax in ((mods[0], ax1),(mods[1], ax2),(mods[2], ax3)):
        predicted = cv.cross_val_predict(mod[0], X, y, cv=12)
        ax.scatter(y, predicted, c="#F2BE2C")
        ax.set_title('Prediction Error for %s' % mod[1])
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4, c="#2B94E9")
        ax.set_ylabel('Predicted')
    plt.xlabel('Measured')
    plt.show()

def plot_resids(model,features,labels):
    for feature in list(features):
        splits = cv.train_test_split(features[[feature]], labels, test_size=0.2)
        X_train, X_test, y_train, y_test = splits
        model.fit(X_train, y_train)
        plt.scatter(model.predict(X_train), model.predict(X_train) - y_train, c='#2B94E9', s=40, alpha=0.5)
        plt.scatter(model.predict(X_test), model.predict(X_test) - y_test, c='#94BA65', s=40)
    plt.hlines(y=0, xmin=0, xmax=100)
    plt.title('Plotting residuals using training (blue) and test (green) data')
    plt.ylabel('Residuals')
    plt.xlim([20,70])
    plt.ylim([-50,50])
    plt.show()

def resids_compare_three(mods,X,y):
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    plt.title('Plotting residuals using training (blue) and test (green) data')
    for mod, ax in ((mods[0], ax1),(mods[1], ax2),(mods[2], ax3)):
        for feature in list(X):
            splits = cv.train_test_split(X[[feature]], y, test_size=0.2)
            X_train, X_test, y_train, y_test = splits
            mod[0].fit(X_train, y_train)
            ax.scatter(mod[0].predict(X_train), mod[0].predict(X_train) - y_train, c='#2B94E9', s=40, alpha=0.5)
            ax.scatter(mod[0].predict(X_test), mod[0].predict(X_test) - y_test, c='#94BA65', s=40)
        ax.hlines(y=0, xmin=0, xmax=100)
        ax.set_title(mod[1])
        ax.set_ylabel('Residuals')
    plt.xlim([20,70])        # Adjust according to your dataset
    plt.ylim([-50,50])       # Adjust according to your dataset
    plt.show()


#############################################################################
# Visual parameter exploration
#############################################################################
def plot_val_curve(features, labels, model):
    p_range = np.logspace(-5, 5, 5)

    train_scores, test_scores = validation_curve(
        model, features, labels, param_name="gamma", param_range=p_range,
        cv=6, scoring="accuracy", n_jobs=1
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve")
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.semilogx(p_range, train_scores_mean, label="Training score", color="#E29539")
    plt.semilogx(p_range, test_scores_mean, label="Cross-validation score", color="#94BA65")
    plt.legend(loc="best")
    plt.show()

def blind_gridsearch(model, X, y):
    C_range = np.logspace(-2, 10, 5)
    gamma_range = np.logspace(-5, 5, 5)
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(SVC(), param_grid=param_grid)
    grid.fit(X, y)

    print(
        "The best parameters are {} with a score of {:0.2f}".format(
            grid.best_params_, grid.best_score_
        )
    )

def visual_gridsearch(model, X, y):
    C_range = np.logspace(-2, 10, 5)
    gamma_range = np.logspace(-5, 5, 5)
    param_grid = dict(gamma=gamma_range, C=C_range)
    grid = GridSearchCV(SVC(), param_grid=param_grid)
    grid.fit(X, y)

    scores = [x[1] for x in grid.grid_scores_]
    scores = np.array(scores).reshape(len(C_range), len(gamma_range))

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=ddloceanmap)
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title(
        "The best parameters are {} with a score of {:0.2f}.".format(
            grid.best_params_, grid.best_score_)
    )
    plt.show()


if __name__ == '__main__':
    ## You only need to do this once!!
    # download_data(OCCUPANCY)
    # download_data(CREDIT)
    # download_data(CONCRETE)
    # z = zipfile.ZipFile(os.path.join('data', 'occupancy_data.zip'))
    # z.extractall(os.path.join('data', 'occupancy_data'))

    occupancy   = pd.read_csv(os.path.join('data','occupancy_data','datatraining.txt'), sep=",")
    credit      = pd.read_excel(os.path.join('data','default%20of%20credit%20card%20clients.xls'), header=1)
    concrete    = pd.read_excel(os.path.join('data','Concrete_Data.xls'))
    occupancy.columns = ['date', 'temp', 'humid', 'light', 'co2', 'hratio', 'occupied']
    concrete.columns = ['cement', 'slag', 'ash', 'water', 'splast', 'coarse', 'fine','age','strength']
    credit.columns = ['id','limit','sex','edu','married','age','apr_delay','may_delay','jun_delay','jul_delay',\
                      'aug_delay','sep_delay','apr_bill','may_bill','jun_bill','jul_bill','aug_bill','sep_bill',\
                      'apr_pay','may_pay','jun_pay','jul_pay','aug_pay','sep_pay','default']

    credit = credit[0:10000]
    features = credit[[
        'limit', 'sex', 'edu', 'married', 'age', 'apr_delay'
    ]]
    labels   = credit['default']

    # blind_gridsearch(SVC(), features, labels)
    visual_gridsearch(SVC(), features, labels)
