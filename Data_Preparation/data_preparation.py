import pickle
import numpy as np
import matplotlib.pyplot as plt

def Data_Prepar_1():

    file_path_1 = '/root/autodl-fs/zhijiangwork/RealData/noise_1280.npy'

    file_path_2 = '/root/autodl-fs/zhijiangwork/RealData/label_1280.npy'
    
    nosiy_data = np.load(file_path_1, allow_pickle=True)/200
    clean_data = np.load(file_path_2, allow_pickle=True)/200
    nosiy_data = nosiy_data - nosiy_data.mean(axis=1, keepdims=True)
    clean_data = clean_data - clean_data.mean(axis=1, keepdims=True)


    print('nosiy_data.shape, clean_data.shape', nosiy_data.shape, clean_data.shape)

    test_start = 278+565+176+33+183+185+280
    test_end = 278+565+176+333+183+185+280+288

    # 按照原始数据的第101到200个划分
    X_test = nosiy_data[test_start:test_end]
    y_test = clean_data[test_start:test_end]

    # 获取训练集
    X_train = np.concatenate((nosiy_data[:test_start], nosiy_data[test_end:]), axis=0)
    y_train = np.concatenate((clean_data[:test_start], clean_data[test_end:]), axis=0)

    column_index = 83

    print("Data from both files has been segmented, merged, and saved successfully.")

    X_train = np.expand_dims(X_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)

    X_test = np.expand_dims(X_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)


    print("Data has been segmented successfully.")

    # 可视化 X_train 和 y_train 中的同一列数据
    plt.figure(figsize=(12, 6))
    # plt.plot(X_train[column_index], label='Noisy Signal')
    # plt.plot(y_train[column_index], label='Original Signal')
    plt.plot(X_test[column_index], label='Noisy Signal')
    plt.plot(y_test[column_index], label='Original Signal')
    plt.title(f'Comparison of Noisy and Original Signal (Column Index: {column_index})')
    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.show()
    plt.close()

    Dataset = [X_train, y_train, X_test, y_test]

    print('Dataset ready to use.')

    return Dataset
    
# 平均101次的MCG数据集
def Data_Prepar_2():

    file_path_1 = '/root/autodl-fs/zhijiangwork/RealData_101/noise_1280.npy'

    file_path_2 = '/root/autodl-fs/zhijiangwork/RealData_101/label_1280.npy'

    with open(file_path_1, 'rb') as file1:
        nosiy_data = pickle.load(file1)

    with open(file_path_2, 'rb') as file2:
        clean_data = pickle.load(file2)

    nosiy_data = np.array(nosiy_data)/200
    clean_data = np.array(clean_data)/200

    nosiy_data = nosiy_data - nosiy_data.mean(axis=1, keepdims=True)
    clean_data = clean_data - clean_data.mean(axis=1, keepdims=True)

    print('nosiy_data.shape, clean_data.shape', nosiy_data.shape, clean_data.shape)

    test_start = 555+176+318+176
    test_end = 555+176+318+176+170

    # 按照原始数据的第101到200个划分
    X_test = nosiy_data[test_start:test_end]
    y_test = clean_data[test_start:test_end]

    # 获取训练集
    X_train = np.concatenate((nosiy_data[:test_start], nosiy_data[test_end:]), axis=0)
    y_train = np.concatenate((clean_data[:test_start], clean_data[test_end:]), axis=0)

    column_index = 83

    X_train = np.expand_dims(X_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)

    X_test = np.expand_dims(X_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    # 可视化 X_train 和 y_train 中的同一列数据
    plt.figure(figsize=(12, 6))
    # plt.plot(X_train[column_index], label='Noisy Signal')
    # plt.plot(y_train[column_index], label='Original Signal')
    plt.plot(X_test[column_index], label='Noisy Signal')
    plt.plot(y_test[column_index], label='Original Signal')
    plt.title(f'Comparison of Noisy and Original Signal (Column Index: {column_index})')
    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.show()
    plt.close()


    Dataset = [X_train, y_train, X_test, y_test]

    return Dataset

# 模拟的ECG数据集
def Data_Prepar_3():


    file_path_1 = '/root/autodl-fs/zhijiangwork/ECGtoMCG_2/normalized_MCG.pkl'

    file_path_2 = '/root/autodl-fs/zhijiangwork/ECGtoMCG_2/normalized_ECG.pkl'

    with open(file_path_1, 'rb') as file1:
        nosiy_data = pickle.load(file1)

    with open(file_path_2, 'rb') as file2:
        clean_data = pickle.load(file2)

    nosiy_data = np.array(nosiy_data)
    clean_data = np.array(clean_data)
    nosiy_data = nosiy_data - nosiy_data.mean(axis=1, keepdims=True)
    clean_data = clean_data - clean_data.mean(axis=1, keepdims=True)

    print('nosiy_data.shape, clean_data.shape', nosiy_data.shape, clean_data.shape)

    test_start = 400
    test_end = 500

    # 按照原始数据的第101到200个划分
    X_test = nosiy_data[test_start:test_end]
    y_test = clean_data[test_start:test_end]

    # 获取训练集
    X_train = np.concatenate((nosiy_data[:test_start], nosiy_data[test_end:]), axis=0)
    y_train = np.concatenate((clean_data[:test_start], clean_data[test_end:]), axis=0)

    column_index = 83

    X_train = np.expand_dims(X_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    # 可视化 X_train 和 y_train 中的同一列数据
    plt.figure(figsize=(12, 6))
    # plt.plot(X_train[column_index], label='Noisy Signal')
    # plt.plot(y_train[column_index], label='Original Signal')
    plt.plot(X_test[column_index], label='Noisy Signal')
    plt.plot(y_test[column_index], label='Original Signal')
    plt.title(f'Comparison of Noisy and Original Signal (Column Index: {column_index})')
    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.show()
    plt.close()


    Dataset = [X_train, y_train, X_test, y_test]
    return Dataset

# 模拟的MCG数据集
def Data_Prepar_4():

    noise_path_1 = '/root/autodl-fs/zhijiangwork/MCGnoise/noise_1280_channel3.npy'
    noise_path_2 = '/root/autodl-fs/zhijiangwork/MCGnoise/noise_1280_channel3.npy'
    Label_path = '/root/autodl-fs/zhijiangwork/MCGnoise/label_1280.npy'


    tmr_data = np.load(noise_path_1, allow_pickle=True)
    print('data1 shape', tmr_data.shape)
    data_label = np.load(Label_path, allow_pickle=True)
    print('data_label shape', data_label.shape)

   
    noise_data = np.array(tmr_data)/200
    clean_data = np.array(data_label)/200
   
    test_start = 6957-930
    test_end = 6957

    # 按照原始数据进行训练集测试集划分，X_test 测试集的带噪，y_test 验证集标签
    X_test = noise_data[test_start:test_end]
    y_test = clean_data[test_start:test_end]

    # 获取训练集，X_train 训练集的带噪，y_train 训练集的标签
    X_train = np.concatenate((noise_data[:test_start], noise_data[test_end:]), axis=0)
    y_train = np.concatenate((clean_data[:test_start], clean_data[test_end:]), axis=0)

    # 判断标签与带噪数据的极差
    test_difference_max = np.max(np.abs(X_test - y_test), axis=1)
    train_difference_max = np.max(np.abs(X_train - y_train), axis=1)

    # 去除异常段 
    X_test = X_test[test_difference_max < 0.525]
    y_test = y_test[test_difference_max < 0.525]
    X_train = X_train[train_difference_max < 0.525]
    y_train = y_train[train_difference_max < 0.525]

    column_index = 83

    X_train = np.expand_dims(X_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)

    # 可视化 X_train 和 y_train 中的同一列数据
    plt.figure(figsize=(12, 6))
    # plt.plot(X_train[column_index], label='Noisy Signal')
    # plt.plot(y_train[column_index], label='Original Signal')
    plt.plot(X_test[column_index], label='Noisy Signal')
    plt.plot(y_test[column_index], label='Original Signal')
    plt.title(f'Comparison of Noisy and Original Signal (Column Index: {column_index})')
    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.show()
    plt.close()


    Dataset = [X_train, y_train, X_test, y_test]
    return Dataset


