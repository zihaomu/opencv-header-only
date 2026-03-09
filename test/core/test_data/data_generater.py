# this file contains python test data generated code.
# It's used to generate test input and output code for C++ layer defined in layer.h
# NOTE: This script must be run at root directory of minfer.
import numpy as np
import math
import sys
import os

ROOT_PATH = "./test/core/test_data/data"
# random seed
np.random.seed(0)

def generate_random_numpy_npy():
    data = np.random.rand(10, 12).astype(np.float32)
    print(data)
    np.save(ROOT_PATH + "/random10x12.npy", data)

def transpose_nd_data():
    data0 = np.random.rand(10, 12, 13).astype(np.float32)

    data1 = np.array(data0.transpose(2, 1, 0), order='C', copy=True)
    data2 = np.array(data0.transpose(0, 2, 1), order='C', copy=True)
    data3 = np.array(data0.transpose(2, 0, 1), order='C', copy=True)
    data4 = np.array(data0.transpose(1, 0, 2), order='C', copy=True)

    print("data 0", data0.flatten()[:30])
    print("data 1", data1.flatten()[:30])
    print("data 2", data2.flatten()[:30])
    
    np.save(ROOT_PATH + "/trans_3d_0_i.npy", data0)
    np.save(ROOT_PATH + "/trans_3d_1_o.npy", data1.reshape(13, 12, 10))
    np.save(ROOT_PATH + "/trans_3d_2_o.npy", data2)
    np.save(ROOT_PATH + "/trans_3d_3_o.npy", data3)
    np.save(ROOT_PATH + "/trans_3d_4_o.npy", data4)
    # return np.transpose(data, shape)

def mat_mul_data_generater():
    data0 = np.random.rand(10, 12, 13).astype(np.float32)
    data1 = np.random.rand(10, 13, 14).astype(np.float32)
    data2 = np.random.rand(10, 12, 13).astype(np.float32)
    data2 = np.random.rand(10, 12, 13).astype(np.float32)
    
    data3 = np.matmul(data0, data1)
    data4 = np.matmul(data0, data2.transpose(0, 2, 1))
    
    data2_0 = np.random.rand(10, 12, 15).astype(np.float32)
    data2_1 = np.matmul(data0.transpose(0, 2, 1), data2_0)
    
    data2_2 = np.random.rand(10, 15, 12).astype(np.float32)
    data2_3 = np.matmul(data0.transpose(0, 2, 1), data2_2.transpose(0, 2, 1))
    
    print("data 0", data0.flatten()[:30])
    print("data 1", data1.flatten()[:30])
    print("data 2", data2.flatten()[:30])
    print("data 3", data3.flatten()[:30])
    print("data 4", data4.flatten()[:30])
    
    print("data 5", data2_0.flatten()[:30])
    print("data 6", data2_1.flatten()[:30])
    
    print("data 7", data2_2.flatten()[:30])
    print("data 8", data2_3.flatten()[:30])
    
    np.save(ROOT_PATH + "/matmul_0_i.npy", data0)
    np.save(ROOT_PATH + "/matmul_1_i.npy", data1)
    np.save(ROOT_PATH + "/matmul_1_o.npy", data3)
    
    np.save(ROOT_PATH + "/matmul_2_i.npy", data2)
    np.save(ROOT_PATH + "/matmul_2_o.npy", data4)
    
    np.save(ROOT_PATH + "/matmul_3_i.npy", data2_0)
    np.save(ROOT_PATH + "/matmul_3_o.npy", data2_1)
    
    np.save(ROOT_PATH + "/matmul_4_i.npy", data2_2)
    np.save(ROOT_PATH + "/matmul_4_o.npy", data2_3)
    
    
    
    
def main():
    # FeadForward_layer_data_generater()
    # generate_random_numpy_npy()
    transpose_nd_data()
    
    # mat_mul_data_generater()

main()