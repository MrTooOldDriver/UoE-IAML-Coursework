
##########################################################
#  Python module template for helper functions of your own (IAML Level 10)
#  Note that:
#  - Those helper functions of your own for Questions 1 and 2 should be defined in this file.
#  - You can decide function names by yourself.
#  - You do not need to include this header in your submission.
##########################################################
import numpy as np

def calculate_r(x,y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_diff = np.array([x-x_mean for x in x])
    x_diff_sqr = np.array([(x-x_mean)**2 for x in x])
    y_diff = np.array([y-y_mean for y in y])
    y_diff_sqr = np.array([(y-y_mean)**2 for y in y])
    return np.sum(x_diff*y_diff) / np.sqrt(np.sum(x_diff_sqr)*np.sum(y_diff_sqr))

def describe_data(input_array):
    return [np.min(input_array), np.max(input_array), np.mean(input_array), np.std(input_array)]