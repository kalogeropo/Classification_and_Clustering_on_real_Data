import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix


def show_graphs(df, old_df):
    for i in range(0, len(df.columns)):
        col_n = df.columns[i]  # column name
        col_v = df[col_n]  # column values
        print(str(col_n) + ":" + "max:" + str(col_v.max()) + "|min:" + str(col_v.min()))
        fig, axs = plt.subplots(2)
        axs[0].plot(col_v)
        axs[0].set_title('new:' + str(col_n))
        axs[1].plot(old_df[old_df.columns[i]])
        axs[1].set_title('old:' + str(col_n))
        plt.show()
    pass


def set_pandas_display_options() -> None:
    """Set pandas display options."""
    # Ref: https://stackoverflow.com/a/52432757/
    display = pd.options.display
    display.max_columns = 10000
    display.max_rows = 20000
    display.max_colwidth = 20000
    display.width = None
    # display.precision = 2  # set as needed
    pass


def create_bound(low_v, len_low, offset):
    low_arr = [low_v] * len_low
    for i in range(1, len_low):
        low_arr[i] = low_arr[i - 1] + offset
    return low_arr


def cov_f(data):
    for i in range(0, len(data.columns)):
        col_n = data.columns[i]  # column name
        col_v = data[col_n]      # column values
        if (col_v.dtype == 'object') or (col_v.dtype == 'bool'):
            data[col_n] = pd.factorize(col_v)[0]
            #  print(str(data.columns[i])+"|"+"max:"+str(df1[col_n].max())+"|min:"+str(df1[col_n].min()))
    data.mask((data == -1), inplace=True)  # NaN = -1 -> NaN = NaN
    return data


def remove_erroneous_values(dff, low=0.001, high=0.99, threshold=3, option=1):
    #a = dff['part_id']
    #dff = dff.drop(columns=['part_id'])
    #  print(dff.isnull().sum())
    """ using IQR:is a measure of statistical dispersion, being equal to the difference between 75th and 25th
     percentiles, or between upper and lower quartiles,[1][2] IQR = Q3 −  Q1. In other words, the IQR is the
     first quartile subtracted from the third quartile; these quartiles can be clearly seen on a box plot on the data.
     It is a trimmed estimator, defined as the 25% trimmed range, and is a commonly used robust measure of scale."""

    if option == 0:
        lb = dff.quantile(low)
        ub = dff.quantile(high)
        #  print(ub)
        #  print(lb)
        dff = dff[(dff <= ub) & (dff >= lb)]
        #dff.insert(0, 'part_id', a)

        """using z scores: are a way to compare results to a “normal” population."""
    elif option == 1:
        v = dff.values
        #  print(v)
        mask = np.abs((v - v.mean(0)) / v.std(0)) > threshold
        dff = pd.DataFrame(np.where(mask, np.nan, v), dff.index, dff.columns)
        #dff.insert(0, 'part_id', a)
    else:
        raise ValueError(" \n""option =0 :using IQR \n""option =1 :using z scores.\n")
    # print(dff)
    # print(dff.isnull().sum())
    return dff


def remove_erroneous_values_manual(dff):
    col = ['gait_get_up', 'raise_chair_time', 'social_phone', 'social_text', 'social_skype']
    df_t = dff.filter(col)
    lb = dff[col].quantile(0.001)
    ub = dff[col].quantile(0.8)
    df_t = df_t[(df_t <= ub) & (df_t >= lb)]
    dff[col] = df_t[col]
    return dff


def handle_missing_values(df, option=1, num_na=60):
    """
    option =0 :	Remove features which have many missing values (defined by num_na as threshold)
    option =1:	Fill missing values of each feature with the average value of the feature.
    option =2 :	Remove entries with missing values in some features.
    """
    if option == 0:
        df = df.dropna(thresh=len(df) - num_na, axis=1)
        df = df.fillna(df.mean())
    elif option == 1:
        df = df.fillna(df.mean())
    elif option == 2:
        df = df[df.isnull().sum(axis=1) < 5]
        df = df.fillna(df.mean())
        #  df = df.dropna(axis=0)
    elif option == 3:
        df = df.dropna(thresh=len(df) - num_na, axis=1)
        df = df[df.isnull().sum(axis=1) < 5]
        df = df.fillna(df.mean())
        #  df = df.dropna(axis=0)
    else:
        raise ValueError(" \n"
                         "option =0 :Remove features which have many missing values(defined by num_na as threshold) \n"
                         "option =1 :Fill missing values of each feature with the average value of the feature.\n"
                         "option =2 :Remove entries with missing values in some features.")
    return df


def select_classifier(option):
    if option == 1:
        classifier = KNeighborsClassifier(p=1, n_neighbors=3)
    elif option == 2:
        classifier = SVC(kernel="linear", C=0.001, random_state=42)
    elif option == 3:
        classifier = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=42)
    elif option == 4:
        classifier = RandomForestClassifier(n_estimators=30, criterion='entropy', random_state=42)
    elif option == 5:
        classifier = MLPClassifier(max_iter=10000, activation='logistic', random_state=42)
    elif option == 6:
        classifier = AdaBoostClassifier(n_estimators=200, random_state=42)
    elif option == 7:
        classifier = GaussianNB()
    elif option == 8:
        classifier = LogisticRegression(solver='liblinear', random_state=42, max_iter=500)
    else:
        raise ValueError("option values are : 1,2,3,4,5,6,7,8")
    return classifier


def cross_v(df, splits=5, i=5):
    x = df.drop("fried", axis=1)
    y = df["fried"]
    """ 
    kf = KFold(n_splits=splits)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    """
    classifier = select_classifier(option=i)
    accuracy = cross_val_score(classifier, x, y, scoring='accuracy', cv=splits)
    acc = accuracy.mean()
    return acc


def create_train_test_set(df, test_size=0.3):
    x = df.drop("fried", axis=1)
    y = df["fried"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=test_size, random_state=42)

    return x_train, x_test, y_train, y_test, x


def show_confusion_matrix(classifier, x, y, name):
    class_names =['Non-frail','Pre-frail','Frail']
    a =plot_confusion_matrix(classifier, x, y,normalize='true',display_labels=class_names)  # doctest: +SKIP
    a.ax_.set_title(name)
    print(name)
    #print(a.confusion_matrix)
    plt.show()  # doctest: +SKIP


def run_classification(col, data, classifier=1):
    #df = data.drop(labels=col, axis=1)
    # cv_acc = cross_v(df, 5, classifier)
    cv_acc = 0
    x_train, x_test, y_train, y_test, x = create_train_test_set(data)
    classifier = select_classifier(option=classifier)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    return y_test, y_pred, classifier,x_test


def part_1(df, j, high, low, thresh, x, na):
    df = remove_erroneous_values(df, option=j, high=high, low=low, threshold=thresh)
    df = remove_erroneous_values_manual(df)
    df = handle_missing_values(df, x, na)
    return df


"""
  feature_importances_df = pd.DataFrame(
         {"feature": list(x.columns), "importance": classifier.feature_importances_}).\
         sort_values("importance", ascending=False)
    print(feature_importances_df)
"""
