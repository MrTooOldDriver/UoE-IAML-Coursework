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
from iaml_cw2_helpers import *
from iaml212cw2_my_helpers import *
X, Y = load_Q1_dataset()
print('X: ', X.shape, 'Y: ', Y.shape)
Xtrn = X[100:, :]; Ytrn = Y[100:] #traning dataset
Xtst = X[0: 100, :]; Ytst = Y[0: 100] #test dataset

print_versions()
# <----

# Q1.1
def iaml212cw2_q1_1():
    fig, ax = plt.subplots(3, 3)
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
        ax[i].hist([Xa, Xb], bins=15)


iaml212cw2_q1_1()   # comment this out when you run the function

# Q1.2
# def iaml212cw2_q1_2():


#
# iaml212cw2_q1_2()   # comment this out when you run the function

# Q1.4
# def iaml212cw2_q1_4():


#
# iaml212cw2_q1_4()   # comment this out when you run the function

# Q1.5
# def iaml212cw2_q1_5():


#
# iaml212cw2_q1_5()   # comment this out when you run the function

# Q1.6
# def iaml212cw2_q1_6():


#
# iaml212cw2_q1_6()   # comment this out when you run the function

# Q1.8
# def iaml212cw2_q1_8():


#
# iaml212cw2_q1_8()   # comment this out when you run the function

# Q1.9
# def iaml212cw2_q1_9():


#
# iaml212cw2_q1_9()   # comment this out when you run the function

# Q1.10
# def iaml212cw2_q1_10():
#
# iaml212cw2_q1_10()   # comment this out when you run the function
