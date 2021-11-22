##########################################################
#  Python script template for Question 1 (IAML Level 10)
#  Note that
#  - You should not change the name of this file, 'iaml212cw2_q1.py', which is the file name you should use when you submit your code for this question.
#  - You should write code for the functions defined below. Do not change their names.
#  - You can define function arguments (parameters) and returns (attributes) if necessary.
#  - In case you define additional functions, do not define them here, but put them in a separate Python module file, "iaml212cw2_my_helpers.py", and import it in this script.
#  - For those questions requiring you to show results in tables, your code does not need to present them in tables - just showing them with print() is fine.
#  - You do not need to include this header in your submission.
##########################################################

# --- Code for loading modules and the data set and pre-processing --->
# NB: You can edit the following and add code (e.g. code for loading sklearn) if necessary.

import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from iaml_cw2_helpers import *
from iaml212cw2_my_helpers import *

X, Y = load_Q1_dataset()
print('X: ', X.shape, 'Y: ', Y.shape)
Xtrn = X[100:, :];
Ytrn = Y[100:]  # traning dataset
Xtst = X[0: 100, :];
Ytst = Y[0: 100]  # test dataset

print_versions()


# <----

# Q1.1
def iaml212cw2_q1_1():
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(len(Xtrn[0])):
        Xa = []
        Xb = []
        for j in range(len(Xtrn)):
            if Ytrn[j] == 0:
                Xa.append(Xtrn[j][i])
            elif Ytrn[j] == 1:
                Xb.append(Xtrn[j][i])
            else:
                print("Unexpected class {0}".format(Ytrn[j]))
        ax_now = ax[int(i / 3)][i % 3]
        ax_now.hist([Xa, Xb], bins=15)
        ax_now.grid()
        ax_now.set_xlabel("A{0}".format(i))
        ax_now.set_ylabel('Frequency')
        ax_now.legend(["Class0", "Class1"], loc="upper right")
    fig.savefig('q1.1.jpg', bbox_inches='tight')


iaml212cw2_q1_1()  # comment this out when you run the function


# Q1.2
def iaml212cw2_q1_2():
    r = []
    for i in range(len(Xtrn[0])):
        x_ith_feature_array = [elem[i] for elem in Xtrn]
        r_value = calculate_r(x_ith_feature_array, Ytrn)
        # r_value = scipy.stats.pearsonr(x_ith_feature_array, Ytrn)[0]
        r.append(r_value)
        print("A{0} correlation coefficients is {1:.3f}".format(i, r_value))


iaml212cw2_q1_2()  # comment this out when you run the function


# Q1.4
def iaml212cw2_q1_4():
    variance_array = dict()
    for i in range(len(Xtrn[0])):
        x_ith_feature_array = [elem[i] for elem in Xtrn]
        variance_array["A{0}".format(i)] = np.var(x_ith_feature_array, ddof=1)
    variance_array = dict(reversed(sorted(variance_array.items(), key=lambda item: item[1])))
    sum_of_variances = np.sum(list(variance_array.values()))
    print('Sum of all biased sample variance variances: {0:.3f}'.format(sum_of_variances))
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    ax[0].bar(variance_array.keys(), variance_array.values())
    ax[0].grid()
    ax[0].set_xlabel("Attributes")
    ax[0].set_ylabel("The amount of variance explained")
    ax[0].set_title('The amount of variance explained by each of the (sorted) attributes')
    variance_cumulative_ratio_array = np.cumsum(list(variance_array.values())) / sum_of_variances
    variance_cumulative_ratio_array = np.insert(variance_cumulative_ratio_array, 0, 0)
    ax[1].plot(range(len(variance_cumulative_ratio_array)), variance_cumulative_ratio_array)
    ax[1].grid()
    ax[1].set_xlabel("The number of attributes")
    ax[1].set_ylabel("The Cumulative variance ratio")
    ax[1].set_title('The cumulative variance ratio against the number of attributes')
    ax[1].set_xticks(np.arange(len(variance_cumulative_ratio_array)))
    fig.savefig('q1.4.jpg',bbox_inches='tight')


iaml212cw2_q1_4()  # comment this out when you run the function


# Q1.5
def iaml212cw2_q1_5():
    variance_array = dict()
    for i in range(len(Xtrn[0])):
        x_ith_feature_array = [elem[i] for elem in Xtrn]
        variance_array["A{0}".format(i)] = np.var(x_ith_feature_array)
    variance_array = dict(reversed(sorted(variance_array.items(), key=lambda item: item[1])))
    sum_of_variances = np.sum(list(variance_array.values()))
    variance_cumulative_ratio_array = np.cumsum(list(variance_array.values())) / sum_of_variances
    variance_cumulative_ratio_array = np.insert(variance_cumulative_ratio_array, 0, 0)

    pca = PCA().fit(Xtrn)
    explained_amount = pca.explained_variance_
    explained_ratio = pca.explained_variance_ratio_
    print('Total amount of unbiased sample variance explained:{0:.3f}'.format(np.sum(explained_amount)))
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    explained_amount_labels = []
    for i in range(len(explained_amount)):
        explained_amount_labels.append('PC{0}'.format(i + 1))
    ax[0].bar(explained_amount_labels, explained_amount)
    ax[0].grid()
    ax[0].set_xlabel("Principal components")
    ax[0].set_ylabel("The amount of variance explained")
    ax[0].set_title('The amount of variance explained by each of the principal components')

    cumulative_explained_ratio = np.cumsum(explained_ratio) / np.sum(explained_ratio)
    cumulative_explained_ratio = np.insert(cumulative_explained_ratio, 0, 0)
    ax[1].plot(range(len(variance_cumulative_ratio_array)), variance_cumulative_ratio_array)
    ax[1].set_xlabel("The number of principal components")
    ax[1].set_ylabel("The Cumulative variance ratio")
    ax[1].set_title('The cumulative variance ratio against the number of principal components')
    ax[1].set_xticks(np.arange(len(variance_cumulative_ratio_array)))
    ax[1].grid()
    fig.savefig('q1.5b.jpg',bbox_inches='tight')

    pca_2d_class0 = []
    pca_2d_class1 = []
    pca_2d = []
    for i in range(len(Xtrn)):
        pca_2d.append([np.dot(pca.components_[0], Xtrn[i]), np.dot(pca.components_[1], Xtrn[i])])
        if Ytrn[i] == 0:
            pca_2d_class0.append([np.dot(pca.components_[0], Xtrn[i]), np.dot(pca.components_[1], Xtrn[i])])
        elif Ytrn[i] == 1:
            pca_2d_class1.append([np.dot(pca.components_[0], Xtrn[i]), np.dot(pca.components_[1], Xtrn[i])])
        else:
            print("Unexpected class {0}".format(Ytrn[i]))
    plt.figure(figsize=(8, 8))
    plt.scatter(x=[point[0] for point in pca_2d_class0], y=[point[1] for point in pca_2d_class0], c='blue', alpha=0.5)
    plt.scatter(x=[point[0] for point in pca_2d_class1], y=[point[1] for point in pca_2d_class1], c='red', alpha=0.5)
    plt.title('Labelled data in PCA space')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(["Class0", "Class1"], loc="upper right")
    plt.grid()
    plt.savefig('q1.5c.jpg',bbox_inches='tight')
    plt.show()

    pc1 = [point[0] for point in pca_2d]
    pc2 = [point[1] for point in pca_2d]
    pc1_r = []
    pc2_r = []
    labels = []
    for i in range(len(Xtrn[0])):
        x_ith_feature_array = [elem[i] for elem in Xtrn]
        pc1_r.append('{:.3f}'.format(scipy.stats.pearsonr(x_ith_feature_array, pc1)[0]))
        pc2_r.append('{:.3f}'.format(scipy.stats.pearsonr(x_ith_feature_array, pc2)[0]))
        labels.append('A{0}'.format(i))
    pca_correlation_coefficient = pd.DataFrame(data=[pc1_r, pc2_r], columns=labels, index=['PC1', 'PC2'])
    print(pca_correlation_coefficient)


iaml212cw2_q1_5()  # comment this out when you run the function


# Q1.6
def iaml212cw2_q1_6():
    variance_array = dict()
    for i in range(len(Xtrn[0])):
        x_ith_feature_array = [elem[i] for elem in Xtrn]
        variance_array["A{0}".format(i)] = np.var(x_ith_feature_array)
    variance_array = dict(reversed(sorted(variance_array.items(), key=lambda item: item[1])))
    sum_of_variances = np.sum(list(variance_array.values()))
    variance_cumulative_ratio_array = np.cumsum(list(variance_array.values())) / sum_of_variances
    variance_cumulative_ratio_array = np.insert(variance_cumulative_ratio_array, 0, 0)

    scaler = StandardScaler().fit(Xtrn)
    Xtrn_s = scaler.transform(Xtrn)
    Xtst_s = scaler.transform(Xtst)

    pca_std = PCA().fit(Xtrn_s)
    explained_amount = pca_std.explained_variance_
    explained_ratio = pca_std.explained_variance_ratio_
    print('Total amount of unbiased sample variance explained:{0:.3f}'.format(np.sum(explained_amount)))

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    explained_amount_labels = []
    for i in range(len(explained_amount)):
        explained_amount_labels.append('PC{0}'.format(i + 1))
    ax[0].bar(explained_amount_labels, explained_amount)
    ax[0].grid()
    ax[0].set_xlabel("Principal components")
    ax[0].set_ylabel("The amount of variance explained")
    ax[0].set_title('The amount of variance explained by each of the principal components')

    cumulative_explained_ratio = np.cumsum(explained_ratio) / np.sum(explained_ratio)
    cumulative_explained_ratio = np.insert(cumulative_explained_ratio, 0, 0)
    ax[1].plot(range(len(variance_cumulative_ratio_array)), variance_cumulative_ratio_array)
    ax[1].set_xlabel("The number of principal components")
    ax[1].set_ylabel("The Cumulative variance ratio")
    ax[1].set_title('The cumulative variance ratio against the number of principal components')
    ax[1].set_xticks(np.arange(len(variance_cumulative_ratio_array)))
    ax[1].grid()
    fig.savefig('q1.6b.jpg', bbox_inches='tight')

    pca_2d_class0 = []
    pca_2d_class1 = []
    pca_2d = []
    for i in range(len(Xtrn_s)):
        pca_2d.append([np.dot(pca_std.components_[0], Xtrn_s[i]), np.dot(pca_std.components_[1], Xtrn_s[i])])
        if Ytrn[i] == 0:
            pca_2d_class0.append([np.dot(pca_std.components_[0], Xtrn_s[i]), np.dot(pca_std.components_[1], Xtrn_s[i])])
        elif Ytrn[i] == 1:
            pca_2d_class1.append([np.dot(pca_std.components_[0], Xtrn_s[i]), np.dot(pca_std.components_[1], Xtrn_s[i])])
        else:
            print("Unexpected class {0}".format(Ytrn[i]))
    plt.figure(figsize=(8, 8))
    plt.scatter(x=[point[0] for point in pca_2d_class0], y=[point[1] for point in pca_2d_class0], c='blue', alpha=0.5)
    plt.scatter(x=[point[0] for point in pca_2d_class1], y=[point[1] for point in pca_2d_class1], c='red', alpha=0.5)
    plt.title('Labelled data in PCA space')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(["Class0", "Class1"], loc="upper right")
    plt.grid()
    plt.savefig('q1.6c.jpg',bbox_inches='tight')
    plt.show()

    pc1 = [point[0] for point in pca_2d]
    pc2 = [point[1] for point in pca_2d]
    pc1_r = []
    pc2_r = []
    labels = []
    for i in range(len(Xtrn[0])):
        x_ith_feature_array = [elem[i] for elem in Xtrn]
        pc1_r.append('{:.3f}'.format(scipy.stats.pearsonr(x_ith_feature_array, pc1)[0]))
        pc2_r.append('{:.3f}'.format(scipy.stats.pearsonr(x_ith_feature_array, pc2)[0]))
        labels.append('A{0}'.format(i))
    pca_correlation_coefficient = pd.DataFrame(data=[pc1_r, pc2_r], columns=labels, index=['PC1', 'PC2'])
    pca_correlation_coefficient


iaml212cw2_q1_6()  # comment this out when you run the function


# Q1.8
def iaml212cw2_q1_8():
    scaler = StandardScaler().fit(Xtrn)
    Xtrn_s = scaler.transform(Xtrn)
    Xtst_s = scaler.transform(Xtst)

    C_value_list = np.logspace(-2, 2, num=13)
    svc_train_result_dict = dict()
    svc_test_result_dict = dict()
    for C in C_value_list:
        skf = StratifiedKFold()
        train_score_list = []
        test_score_list = []
        for train_index, test_index in skf.split(Xtrn_s, Ytrn):
            X_train, y_train = Xtrn_s[train_index], Ytrn[train_index]
            X_test, y_test = Xtrn_s[test_index], Ytrn[test_index]
            clf = SVC(C=C).fit(X_train, y_train)
            train_score_list.append(clf.score(X_train, y_train))
            test_score_list.append(clf.score(X_test, y_test))
        svc_train_result_dict[C] = train_score_list
        svc_test_result_dict[C] = test_score_list
    svc_train_result_mean = [np.mean(value) for key, value in svc_train_result_dict.items()]
    svc_train_result_std = [np.std(value) for key, value in svc_train_result_dict.items()]
    svc_test_result_mean = [np.mean(value) for key, value in svc_test_result_dict.items()]
    svc_test_result_std = [np.std(value) for key, value in svc_test_result_dict.items()]
    C_value_list_log = [np.log10(C) for C in C_value_list]
    plt.figure(figsize=(8, 8))
    plt.errorbar(x=C_value_list_log, y=svc_train_result_mean, yerr=svc_train_result_std, color='blue')
    plt.errorbar(x=C_value_list_log, y=svc_test_result_mean, yerr=svc_test_result_std, color='red')
    plt.title('Accuracy against variances penalty parameter C')
    plt.xlabel('log(penalty parameter C)')
    plt.ylabel('Accuracy')
    plt.legend(["Training accuracy", "Test accuracy"], loc="upper right")
    plt.grid()
    plt.savefig('q1.8.jpg', bbox_inches='tight')

    print('Best C is {0:.4f}'.format(C_value_list[5]))
    print('Best C svc train mean is {0:.4f}'.format(svc_train_result_mean[np.argmax(svc_train_result_mean)]))
    print('Best C svc test mean is {0:.4f}'.format(svc_test_result_mean[np.argmax(svc_test_result_mean)]))
    clf = SVC(C=C_value_list[5]).fit(Xtrn_s, Ytrn)
    print('The number of instances correctly classified:{0} and classification accuracy: {1:.2f}'
          .format(len(Xtst_s) * clf.score(Xtst_s, Ytst), clf.score(Xtst_s, Ytst), ))


iaml212cw2_q1_8()  # comment this out when you run the function


# Q1.9
def iaml212cw2_q1_9():
    Ztrn = []
    for i in range(len(Xtrn)):
        if Ytrn[i] == 0 and Xtrn[i][4] > 1:
            Ztrn.append(np.array([Xtrn[i][4], Xtrn[i][7]]))
    Ztrn = np.array(Ztrn)
    print(Ztrn.shape)
    # Ztrn
    print('The mean vector is {0}'.format(np.mean(Ztrn, axis=0)))
    print('Covariance matrix is {0}'.format(np.cov(Ztrn, rowvar=False)))
    mean_vector = np.mean(Ztrn, axis=0)
    variance_cov_matrix = np.cov(Ztrn, rowvar=False)

    variance_cov_matrix_inv = np.linalg.inv(variance_cov_matrix)
    variance_cov_matrix_det = np.linalg.det(variance_cov_matrix)

    Ztrn_x = [point[0] for point in Ztrn]
    Ztrn_y = [point[1] for point in Ztrn]

    Ztrn_X, Ztrn_Y = np.meshgrid(np.linspace(0, 55), np.linspace(0, 55))
    coe = 1.0 / ((2 * np.pi) ** 2 * variance_cov_matrix_det) ** 0.5
    Z = coe * np.e ** (-0.5 * (variance_cov_matrix_inv[0, 0] * (Ztrn_X - mean_vector[0]) ** 2 +
                               (variance_cov_matrix_inv[0, 1] + variance_cov_matrix_inv[1, 0])
                               * (Ztrn_X - mean_vector[0]) * (Ztrn_Y - mean_vector[1]) + variance_cov_matrix_inv[1, 1] * (
                                       Ztrn_Y - mean_vector[1]) ** 2))

    plt.figure(figsize=(12, 12))
    plt.contour(np.linspace(0, 50), np.linspace(0, 50), Z)
    plt.scatter(x=Ztrn_x, y=Ztrn_y, c='blue', alpha=0.5)
    plt.axis('equal')
    plt.title('Attribute A4 against attribute A7 and label is 0')
    plt.xlabel('A4')
    plt.ylabel('A7')
    plt.legend(["Class0"], loc="upper right")
    plt.grid()
    plt.savefig('q1.9.jpg',bbox_inches='tight')
    plt.show()


iaml212cw2_q1_9()  # comment this out when you run the function


# Q1.10
def iaml212cw2_q1_10():
    Ztrn = []
    for i in range(len(Xtrn)):
        if Ytrn[i] == 0 and Xtrn[i][4] > 1:
            Ztrn.append(np.array([Xtrn[i][4], Xtrn[i][7]]))
    Ztrn = np.array(Ztrn)
    some_points = list(zip(np.linspace(0, 50), np.linspace(0, 50)))
    # gnb = GaussianNB(priors=[1.0, 0])
    gnb = GaussianNB()
    gnb.fit(Ztrn, np.zeros(len(Ztrn)))
    # some_points
    # print(gnb.sigma_)
    # print(gnb.theta_)
    # print(gnb.class_prior_)

    mean_vector = [gnb.theta_[0][0], gnb.theta_[0][1]]
    variance_cov_matrix = [[gnb.sigma_[0][0], 0], [0, gnb.sigma_[0][1]]]
    print('The mean vector is {0}'.format(gnb.theta_.reshape(2, )))
    print('Covariance matrix is {0}'.format(variance_cov_matrix))
    # print(variance_cov_matrix)
    variance_cov_matrix_inv = np.linalg.inv(variance_cov_matrix)
    variance_cov_matrix_det = np.linalg.det(variance_cov_matrix)

    Ztrn_x = [point[0] for point in Ztrn]
    Ztrn_y = [point[1] for point in Ztrn]

    Ztrn_X, Ztrn_Y = np.meshgrid(np.linspace(0, 55), np.linspace(0, 55))
    coe = 1.0 / ((2 * np.pi) ** 2 * variance_cov_matrix_det) ** 0.5
    Z = coe * np.e ** (-0.5 * (variance_cov_matrix_inv[0, 0] * (Ztrn_X - mean_vector[0]) ** 2 +
                               (variance_cov_matrix_inv[0, 1] + variance_cov_matrix_inv[1, 0])
                               * (Ztrn_X - mean_vector[0]) * (Ztrn_Y - mean_vector[1]) + variance_cov_matrix_inv[1, 1] * (
                                       Ztrn_Y - mean_vector[1]) ** 2))

    plt.figure(figsize=(12, 12))
    plt.contour(np.linspace(0, 55), np.linspace(0, 55), Z)
    plt.scatter(x=Ztrn_x, y=Ztrn_y, c='blue', alpha=0.5)
    plt.axis('equal')
    plt.title('Attribute A4 against attribute A7 and label is 0, Assuming naive-Bayes')
    plt.xlabel('A4')
    plt.ylabel('A7')
    plt.legend(["Class0"], loc="upper right")
    plt.grid()
    plt.rcParams.update({'font.size': 16})
    plt.savefig('q1.10.jpg',bbox_inches='tight')
    plt.show()


iaml212cw2_q1_10()  # comment this out when you run the function
