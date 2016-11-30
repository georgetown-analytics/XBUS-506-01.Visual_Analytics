#!/usr/bin/env python
# vizDr.py


import os
import zipfile
import requests
import numpy as np
import pandas as pd
import seaborn as sns
from string import letters
from matplotlib import colors
import matplotlib.pyplot as plt
from pandas.tools.plotting import radviz
from matplotlib.colors import ListedColormap
from pandas.tools.plotting import parallel_coordinates


from sklearn import cross_validation as cv
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import validation_curve

from sklearn.metrics import auc
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error as mse

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.preprocessing import normalize, scale

from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, RANSACRegressor
from sklearn.neighbors import KNeighborsClassifier


sns.set_style("whitegrid")

DDL = ["#2B94E9", "#E29539", "#F2BE2C","#9BDAF9", "#666666", "#DBDBDB", "#4CB5F5", "#F2BE2C","#B7B8B6","#94BA65"]
CR =  ["#DBDBDB", "#CBD5DC", "#BBCEDE","#ABC8DF","#9BC1E0","#8BBBE1","#7BB4E3","#6BAEE4","#5BA7E5","#4BA1E6","#3B9AE8","#2B94E9"]

cmap = colors.ListedColormap(CR)

sns.set_style("whitegrid")
ddl_heat = ["#DBDBDB","#DCD5CC","#DCCEBE","#DDC8AF","#DEC2A0","#DEBB91","#DFB583","#DFAE74",\
            "#E0A865","#E1A256","#E19B48","#E29539"]
ddlheatmap = colors.ListedColormap(ddl_heat)

# sns.set_palette(cmap)

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

def box_viz(df):
    ax = sns.boxplot(data = df, palette=DDL)
    plt.xticks(rotation=60)
    fig = ax.get_figure()
    fig.savefig('figures/box_viz.png', transparent=True)

def hist_viz(df,feature):
    ax = sns.distplot(df[feature], color="#2B94E9")
    plt.xlabel(feature)
    fig = ax.get_figure()
    fig.savefig('figures/hist_viz.png', transparent=True)

def splom_viz(df,labels=None):
    ax = sns.pairplot(df, palette=DDL, hue=labels, diag_kind="kde", size=2)
    plt.savefig('figures/splom_viz.png', transparent=True)

def pcoord_viz(df, labels):
    fig = parallel_coordinates(df, labels, color=sns.color_palette())
    plt.savefig('figures/pcoord_viz.png', transparent=True)

def rad_viz(df,labels):
    fig = radviz(df, labels, color=sns.color_palette())
    plt.savefig('figures/rad_viz.png', transparent=True)

def joint_viz(feat1,feat2,df):
    ax = sns.jointplot(feat1, feat2, data=df, kind='reg', size=5, color="#E29539")
    plt.savefig('figures/joint_viz.png', transparent=True)

def plot_classification_report(cr, title='Classification report', cmap=cmap):
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
    plt.savefig('figures/classrpt_viz.png', transparent=True)

def roc_compare_two(ys, yhats, mods):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
    for y, yhat, m, ax in ((ys[0], yhats[0], mods[0], ax1), (ys[1], yhats[1], mods[1], ax2)):
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y,yhat)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        ax.set_title('ROC for %s' % m)
        ax.plot(false_positive_rate, true_positive_rate, c="#2B94E9", label='AUC = %0.2f'% roc_auc)
        ax.legend(loc='lower right')
        ax.plot([0,1],[0,1],'m--',c="#666666")
    plt.xlim([0,1])
    plt.ylim([0,1])

    plt.savefig('figures/rocauc_viz.png', transparent=True)


def regrErrorViz(model,features,labels):
    predicted = cv.cross_val_predict(model, features, labels, cv=12)
    fig, ax = plt.subplots()
    ax.scatter(labels, predicted)
    ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--', lw=4, color=sns.color_palette())
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.savefig('figures/regrerror_viz.png', transparent=True)


def error_compare_three(mods,X,y):
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    for mod, ax in ((mods[0], ax1),(mods[1], ax2),(mods[2], ax3)):
        predicted = cv.cross_val_predict(mod[0], X, y, cv=12)
        ax.scatter(y, predicted, c='#F2BE2C')
        ax.set_title('Prediction Error for %s' % mod[1])
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4, c='#2B94E9')
        ax.set_ylabel('Predicted')
    plt.xlabel('Measured')
    plt.savefig('figures/regrerror3_viz.png', transparent=True)




def plotResids(features,labels,model):
    for feature in list(features):
        splits = cv.train_test_split(features[[feature]], labels, test_size=0.2)
        X_train, X_test, y_train, y_test = splits
        model.fit(X_train, y_train)
        plt.scatter(model.predict(X_train), model.predict(X_train) - y_train, color=sns.color_palette(), s=40, alpha=0.5)
        plt.scatter(model.predict(X_test), model.predict(X_test) - y_test, c='g', s=40)
    plt.hlines(y=0, xmin=0, xmax=100)
    plt.title('Plotting residuals using training (blue) and test (green) data')
    plt.ylabel('Residuals')
    plt.xlim([20,70])
    plt.ylim([-50,50])
    plt.savefig('figures/resids_viz.png', transparent=True)


def resids_compare_three(mods,X,y):
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    plt.title('Plotting residuals using training (blue) and test (green) data')
    for m, ax in ((mods[0], ax1),(mods[1], ax2),(mods[2], ax3)):
        for feature in list(X):
            splits = cv.train_test_split(X[[feature]], y, test_size=0.2)
            X_tn, X_tt, y_tn, y_tt = splits
            m[0].fit(X_tn, y_tn)
            ax.scatter(m[0].predict(X_tn),m[0].predict(X_tn)-y_tn,c='#2B94E9',s=40,alpha=0.5)
            ax.scatter(m[0].predict(X_tt), m[0].predict(X_tt)-y_tt,c='#94BA65',s=40)
        ax.hlines(y=0, xmin=0, xmax=100)
        ax.set_title(m[1])
        ax.set_ylabel('Residuals')
    plt.xlim([20,70])        # Adjust according to your dataset
    plt.ylim([-50,50])       # Adjust according to your dataset
    plt.savefig('figures/resids3.png', transparent=True)

def plot_val_curve(features, labels, model):
    p_range = np.logspace(-5, 5, 5)

    train_scores, test_scores = validation_curve(
        model, features, labels, param_name='gamma', param_range=p_range,
        cv=6, scoring='accuracy', n_jobs=1
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title('Validation Curve')
    plt.xlabel('$\gamma$')
    plt.ylabel('Score')
    plt.semilogx(p_range, train_scores_mean, label='Training score', color='#E29539')
    plt.semilogx(p_range, test_scores_mean, label='Cross-validation score', color='#94BA65')
    plt.legend(loc='best')
    plt.savefig('figures/val_curve.png', transparent=True)

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
    plt.imshow(scores, interpolation='nearest', cmap=ddlheatmap)
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title(
        "The best parameters are {} with a score of {:0.2f}.".format(
        grid.best_params_, grid.best_score_)
    )
    plt.savefig('figures/viz_gridsearch.png', transparent=True)



if __name__ == '__main__':
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



    # box_viz(concrete)

    # hist_viz(credit,'age')

    # splom_viz(concrete[['cement', 'slag', 'ash', 'water', 'splast', 'coarse']])
    #
    # pcoord_viz(occupancy.ix[:,1:],'occupied')
    #
    # rad_viz(occupancy.ix[:,1:],'occupied')
    #
    # joint_viz('apr_bill','sep_bill',credit)



    # We'll divide our occupancy data into features (attributes) and labels (targets)
    occ_features = occupancy[['temp', 'humid', 'light', 'co2', 'hratio']]
    occ_labels   = occupancy['occupied']
    #
    # Let's scale our occupancy input vectors
    standardized_occ_features = scale(occ_features)

    # Then split the data into 'test' and 'train' for cross validation
    splits = cv.train_test_split(standardized_occ_features, occ_labels, test_size=0.2)
    X_train, X_test, y_train, y_test = splits

    # We'll use the suggested LinearSVC model first
    lin_clf = LinearSVC()
    lin_clf.fit(X_train, y_train)
    y_true = y_test
    y_pred = lin_clf.predict(X_test)
    # print confusion_matrix(y_true, y_pred)
    # print classification_report(y_true, y_pred, target_names=["not occupied","occupied"])
    # cr = classification_report(y_true, y_pred)
    # plot_classification_report(cr)
    #
    #
    # # Then try the k-nearest neighbor model next
    # knn_clf = KNeighborsClassifier()
    # knn_clf.fit(X_train, y_train)
    # y_true = y_test
    # y_pred = knn_clf.predict(X_test)
    # # print confusion_matrix(y_true, y_pred)
    # # print classification_report(y_true, y_pred, target_names=["not occupied","occupied"])
    #
    #
    #
    #
    # We'll divide our credit data into features (attributes) and labels (targets)
    cred_features = credit[['limit', 'sex', 'edu', 'married', 'age', 'apr_delay', 'may_delay', \
                            'jun_delay', 'jul_delay', 'aug_delay', 'sep_delay', 'apr_bill', 'may_bill', \
                            'jun_bill', 'jul_bill', 'aug_bill', 'sep_bill', 'apr_pay', 'may_pay', \
                            'jun_pay', 'jul_pay', 'aug_pay', 'sep_pay']]
    cred_labels   = credit['default']

    # Scale it
    standardized_cred_features = scale(cred_features)

    # Then split into 'test' and 'train' for cross validation
    splits = cv.train_test_split(standardized_cred_features, cred_labels, test_size=0.2)
    X_train, X_test, y_train, y_test = splits

    # We'll use the suggested LinearSVC model
    lin_clf = LinearSVC()
    lin_clf.fit(X_train, y_train)
    y_true_svc = y_test
    y_pred_svc = lin_clf.predict(X_test)

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)
    y_true_knn = y_test
    y_pred_knn = knn_clf.predict(X_test)

    actuals = np.array([y_true_svc,y_true_knn])
    predictions = np.array([y_pred_svc,y_pred_knn])
    models = ['LinearSVC','KNeighborsClassifier']

    credit=credit[0:1000]
    X = scale(credit[['limit','sex','edu','married','age','apr_delay']])
    y = credit['default']
    # plot_val_curve(X, y, SVC())
    # roc_compare_two(actuals, predictions, models)

    # visual_gridsearch(SVC(), X, y)
    # X = standardized_cred_features
    # y = cred_labels
    # p_range = np.logspace(-7, 3, 5)
    # train_scores, test_scores = validation_curve(SVC(), X, y, param_name="gamma",
    #                             param_range=p_range, cv=12, scoring="accuracy", n_jobs=1)
    # train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    # test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)
    #
    # plt.semilogx(p_range, train_scores_mean, label="Training score", color="r")
    # plt.semilogx(p_range, test_scores_mean, label="Cross-validation score", color="g")
    #
    # plt.show()

    # C_range = np.logspace(-2, 10, 13)
    # gamma_range = np.logspace(-9, 3, 13)
    # param_grid = dict(gamma=gamma_range, C=C_range)
    # grid = GridSearchCV(SVC(), param_grid=param_grid)
    # grid.fit(standardized_cred_features, cred_labels)
    #
    # print("The best parameters are %s with a score of %0.2f"
    #       % (grid.best_params_, grid.best_score_))
    #
    # scores = [x[1] for x in grid.grid_scores_]
    # scores = np.array(scores).reshape(len(C_range), len(gamma_range))
    #
    # plt.figure(figsize=(8, 6))
    # plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    # plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
    # plt.xlabel('gamma')
    # plt.ylabel('C')
    # plt.colorbar()
    # plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    # plt.yticks(np.arange(len(C_range)), C_range)
    # plt.title('Validation accuracy')
    # plt.show()









    # print confusion_matrix(y_true, y_pred)
    # print classification_report(y_true, y_pred, target_names=["did not default","defaulted"])
    # rocViz(y_true,y_pred,"Linear Support Vector Model")
    #
    #
    # # Then try k-nearest neighbor
    # knn_clf = KNeighborsClassifier()
    # knn_clf.fit(X_train, y_train)
    # y_true = y_test
    # y_pred = knn_clf.predict(X_test)
    # # print confusion_matrix(y_true, y_pred)
    # # print classification_report(y_true, y_pred, target_names=["did not default","defaulted"])
    #
    # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_pred)
    # roc_auc = auc(false_positive_rate, true_positive_rate)
    #
    # rocViz(false_positive_rate, true_positive_rate, roc_auc, model)
    #
    conc_features = concrete[['cement', 'slag', 'ash', 'water', 'splast', 'coarse', 'fine', 'age']]
    conc_labels   = concrete['strength']

    splits = cv.train_test_split(conc_features, conc_labels, test_size=0.2)
    X_train, X_test, y_train, y_test = splits
    ridge_reg = Ridge()
    ridge_reg.fit(X_train, y_train)
    y_true = y_test
    y_pred = ridge_reg.predict(X_test)
    # print "Mean squared error = %0.3f" % mse(y_true, y_pred)
    # print "R2 score = %0.3f" % r2_score(y_true, y_pred)
    # models = np.array([(Ridge(),'Ridge'), (SVR(),'SVR'), (RANSACRegressor(),'RANSAC')])
    # error_compare_three(models, conc_features, conc_labels)

    models = np.array([(Ridge(),'Ridge'), (LinearRegression(),'Linear Regression'), (SVR(),'SVR')])
    # resids_compare_three(models, conc_features, conc_labels)


    # Generate a large random dataset
    rs = np.random.RandomState(33)
    d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                     columns=list(letters[:26]))

    # Compute the correlation matrix
    corr = d.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(h_neg=60.9, h_pos=60.9, s=94.8, l=79.5, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,
                square=True, xticklabels=5, yticklabels=5,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.savefig('/Users/pepper/Desktop/Projects/DDL/pycon2016/vizMLslides/images/bricks.png', transparent=True)
