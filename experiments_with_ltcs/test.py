import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Run on GPU

import tensorflow.compat.v1 as tf
import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU
import argparse


if __name__ == "__main__":
     
     # losses2_result = 2.56
     # losses2_result_array = np.array([losses2_result])            
     # np.savetxt('result_APMealTime/result_losses2_result.txt', losses2_result_array, fmt='%f')
     # with open('result_APMealTime/result_losses2_result.txt', 'a') as f:
     #      f.write('\n')
     # # tyMean = tf.reduce_mean(tf.reduce_mean(t_y,axis=0),axis=0)
     # tyMean2 = 1.655
     # print("ty: {}".format(tyMean2)) # save this ty in a file
     # tyMean2_result = np.array([tyMean2]) 
     # np.savetxt('result_APMealTime/result_ty_result.txt', tyMean2_result, fmt='%f')
     # with open('result_APMealTime/result_ty_result.txt', 'a') as f1:
     #      f1.write('\n')
     

     for i in range(1, 11):
            train_x = np.loadtxt(f"data/har/train_test_combined/TrainTestLNNData/Preg1_TrainDataAPVR3_{i}.txt")
            print(train_x, '\n')

            