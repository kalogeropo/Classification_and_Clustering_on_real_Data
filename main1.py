import time

from sklearn.metrics import accuracy_score
from meros_A import *
from partC import *


high = 0.975
low = 0.034
names = ["Nearest Neighbors", "Linear SVM", "Decision Tree","Random Forest", "Neural Net", "AdaBoost", "Naive Bayes", "LogisticRegression"]
cl = ['weight_loss', 'exhaustion_score', 'gait_speed_slower', 'grip_strength_abnormal', 'low_physical_activity']
thresh = 3
method = ['IQR', 'z-score']
method2 = ['Remove features which have many missing values (defined by num_na as threshold)',
           'Fill missing values of each feature with the average value of the feature.',
           'remove rows with more than 4 NaN values & fill with mean() rest',
           'hybrid']
#set_pandas_display_options()
filename_1 = 'clinical_dataset.csv'
df = pd.read_csv(filename_1, sep=';')
df = cov_f(df)  # todo  1
old_df = df
result_list = []
#show_graphs(df,df)
for i in range(1, 9):
    for j in range(0, 1):
        for x in range(0, 4):
            start_time=time.time()
            #  print(str(names[i - 1]))  # todo  print alg name
            #  print(str(method[j]))  # todo  print alg name
            #  print(str(method2[x]))  # todo  print alg name
            df = part_1(old_df, j, high, low, thresh, x, 30)  # todo  1+2
            #show_graphs(old_df,df)
            y_test, y_pred, classifier, x_test = run_classification(cl, df, classifier=i)
            end_time=time.time()
            alg_exec_time = end_time-start_time
            acc = round((accuracy_score(y_test, y_pred)),3)
            #print(str(names[i - 1]) + ":" + str(acc))
            result_list.append([names[i-1],method[j],method2[x],acc,alg_exec_time])
            #show_confusion_matrix(classifier=classifier, x=x_test, y=y_test, name=(names[i - 1] + str(":") + str(acc)))
result_df =pd.DataFrame(result_list,columns=['Classifier','IQR/Z-score',"Err. Val. methods","Accuracy score","Time"])

print(result_df)
"""
optimized
KNN         [0,0.95]
Linear SVM  [0.06,0.95]
tree 1      [0.047,0.97]
forest      [0.034,0.975]
net         [0.003,0.959]
ada         [0.003,0.953]
naine       [0.038,0.985]
regression  [0.002,0.958]
"""

"""
Nearest Neighbors:0.506
Nearest Neighbors:0.438
Nearest Neighbors:0.532 ~0.623 (optimized)
Nearest Neighbors:0.491
Linear SVM:0.525
Linear SVM:0.586
Linear SVM:0.627
Linear SVM:0.547
Decision Tree:0.506
Decision Tree:0.531
Decision Tree:0.595
Decision Tree:0.547
Random Forest:0.617
Random Forest:0.599
Random Forest:0.698
Random Forest:0.621
Neural Net:0.617
Neural Net:0.568
Neural Net:0.611
Neural Net:0.615
AdaBoost:0.636
AdaBoost:0.654
AdaBoost:0.619
AdaBoost:0.634
Naive Bayes:0.543
Naive Bayes:0.537
Naive Bayes:0.603
Naive Bayes:0.528
LogisticRegression:0.574
LogisticRegression:0.543
LogisticRegression:0.627
LogisticRegression:0.54
Process finished with exit code 0
"""
plot_by_erroneous_methods(result_df)
plot_by_method(result_df)
plot_by_classifier_Max_acc_all_methods(result_df)
plot_by_classifier_Max_Time_all_methods(result_df)