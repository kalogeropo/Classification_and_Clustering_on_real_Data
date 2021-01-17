import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#visualization of results

#method = ['IQR', 'z-score']

def plot_by_erroneous_methods(df):
    temp1 = df.loc[
        df["Err. Val. methods"] == "Remove features which have many missing values (defined by num_na as threshold)"]
    temp2 = df.loc[
        df["Err. Val. methods"] == 'Fill missing values of each feature with the average value of the feature.']
    temp3 = df.loc[df["Err. Val. methods"] == 'remove rows with more than 4 NaN values & fill with mean() rest']
    temp4 = df.loc[df["Err. Val. methods"] == "hybrid"]

    for item in [temp1, temp2, temp3, temp4]:
        plt.plot(item["Classifier"], item["Accuracy score"])
    plt.title("Accuracy Scores on each classifier depending on erroneous values handling")
    plt.ylabel('Accuracy')
    plt.gca().legend(('Remove missing values',
                      'Fill with average',
                      'Remove rows with 4 or more NaN',
                      'hybrid'))
    plt.show()

    for item in [temp1, temp2, temp3, temp4]:
        plt.plot(item["Classifier"], item["Time"])
    plt.title("Time for each classifier depending on erroneous values handling")
    plt.ylabel('Time')
    plt.gca().legend(('Remove missing values',
                      'Fill with average',
                      'Remove rows with 4 or more NaN',
                      'hybrid'))
    plt.show()
    return 0

def plot_by_classifier_Max_acc_all_methods(df):
    temp1  = df.loc[df["Classifier"]=="Nearest Neighbors"]
    temp1in = temp1["Accuracy score"].idxmax()
    temp1 = temp1.loc[temp1in]

    temp2 = df.loc[df["Classifier"] == "Linear SVM"]
    temp2in = temp2["Accuracy score"].idxmax()
    temp2 = temp2.loc[temp2in]
    temp3 = df.loc[df["Classifier"] == "Decision Tree"]
    temp3in = temp3["Accuracy score"].idxmax()
    temp3 = temp3.loc[temp3in]

    temp4 = df.loc[df["Classifier"] == "Random Forest"]
    temp4in = temp4["Accuracy score"].idxmax()
    temp4 = temp4.loc[temp4in]

    temp5 = df.loc[df["Classifier"] == "Neural Net"]
    temp5in = temp5["Accuracy score"].idxmax()
    temp5 = temp5.loc[temp5in]

    temp6 = df.loc[df["Classifier"] == "AdaBoost"]
    temp6in = temp6["Accuracy score"].idxmax()
    temp6 = temp6.loc[temp6in]

    temp7 = df.loc[df["Classifier"] == "Naive Bayes"]
    temp7in = temp7["Accuracy score"].idxmax()
    temp7 = temp7.loc[temp7in]

    temp8 = df.loc[df["Classifier"] == "LogisticRegression"]
    temp8in = temp8["Accuracy score"].idxmax()
    temp8 = temp8.loc[temp8in]

    plot_list=[temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8]
    plot_df = pd.DataFrame(plot_list,
                             columns=['Classifier', 'IQR/Z-score', "Err. Val. methods", "Accuracy score", "Time"])
    print(plot_df)
    plot_df.plot.bar("Classifier","Accuracy score")
    plt.show()
    return 0

def plot_by_classifier_Max_Time_all_methods(df):
    temp1  = df.loc[df["Classifier"]=="Nearest Neighbors"]
    temp1in = temp1["Time"].idxmax()
    temp1 = temp1.loc[temp1in]

    temp2 = df.loc[df["Classifier"] == "Linear SVM"]
    temp2in = temp2["Time"].idxmax()
    temp2 = temp2.loc[temp2in]
    temp3 = df.loc[df["Classifier"] == "Decision Tree"]
    temp3in = temp3["Time"].idxmax()
    temp3 = temp3.loc[temp3in]

    temp4 = df.loc[df["Classifier"] == "Random Forest"]
    temp4in = temp4["Time"].idxmax()
    temp4 = temp4.loc[temp4in]

    temp5 = df.loc[df["Classifier"] == "Neural Net"]
    temp5in = temp5["Time"].idxmax()
    temp5 = temp5.loc[temp5in]

    temp6 = df.loc[df["Classifier"] == "AdaBoost"]
    temp6in = temp6["Time"].idxmax()
    temp6 = temp6.loc[temp6in]

    temp7 = df.loc[df["Classifier"] == "Naive Bayes"]
    temp7in = temp7["Time"].idxmax()
    temp7 = temp7.loc[temp7in]

    temp8 = df.loc[df["Classifier"] == "LogisticRegression"]
    temp8in = temp8["Time"].idxmax()
    temp8 = temp8.loc[temp8in]

    plot_list=[temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8]
    plot_df = pd.DataFrame(plot_list,
                             columns=['Classifier', 'IQR/Z-score', "Err. Val. methods", "Accuracy score", "Time"])
    print(plot_df)
    plot_df.plot.bar("Classifier","Time")
    plt.show()
    return 0

def plot_clusters_and_scores(list_of_scores,y_text,x_text,graphtitle):
    plt.plot(list_of_scores)
    plt.ylabel(y_text)
    plt.xlabel(x_text)
    plt.title(graphtitle)
    plt.show()
    return 0

def cluster_analysis(prediction,reality):
    no_of_predicted_clusters=set(prediction)
    no_of_real_clusters = set(reality)
    print(no_of_real_clusters,no_of_predicted_clusters)
    for i in no_of_real_clusters:
        x=np.count_nonzero(prediction == i)
        y=np.count_nonzero(reality == i)

        per = 100*(y-x)/y
        print("The real cluster %d has %d items"%(i,y))
        print("The predicted cluster %d has %d items" % (i, x))
        print("For Cluster %d the percentage of miss labeled item is %f" %(i,abs(per)))
    print("The real data consist of %d clusters and the predicted of %d"%(len(no_of_real_clusters),len(no_of_predicted_clusters)))
    return 0