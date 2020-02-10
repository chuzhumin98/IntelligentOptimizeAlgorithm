import numpy as np

def common_transfer():
    path = 'data/australian.dat'
    path_out = 'data/australian_libsvm.dat'
    file = open(path, 'r')
    file_out = open(path_out, 'w')
    features_num = 15
    while True:
        line = file.readline()
        if line:
            features = line.split(' ', features_num - 1)
            print(features)
            line_out = features[features_num - 1].strip()
            for i in range(features_num - 1):
                line_out += ' {}:{}'.format(i + 1, features[i])
            file_out.write(line_out + '\n')
        else:
            break
    file.close()
    file_out.close()

def normal_transfer():
    path = 'data/australian.dat'
    path_out = 'data/australian_normal_libsvm.dat'
    file = open(path, 'r')
    file_out = open(path_out, 'w')
    features_num = 15
    length = 690
    data = np.zeros([features_num, length], dtype=np.float32)
    cnt = 0
    while True:
        line = file.readline()
        if line:
            features = line.split(' ', features_num - 1)
            for i in range(features_num):
                data[i, cnt] = float(features[i])
            cnt += 1
        else:
            break
    file.close()

    max_values = np.max(data, axis=1).reshape([features_num, 1])
    min_values = np.min(data, axis=1).reshape([features_num, 1])
    data = (data - np.tile(min_values, [1, length])) / np.tile(max_values - min_values, [1, length])

    for i in range(length):
        line = str(int(data[features_num - 1, i]))
        for j in range(features_num - 1):
            line += ' {}:{}'.format(j + 1, data[j, i])
        file_out.write(line + '\n')
    file_out.close()

def onehot_transfer():
    attributes_pernum = [1, 1, 1, 3, 14, 9, 1, 1, 1, 1, 1, 3, 1, 1, 1]

    path = 'data/australian.dat'
    path_out = 'data/australian_onehot_libsvm.dat'
    file = open(path, 'r')
    file_out = open(path_out, 'w')
    features_num = sum(attributes_pernum)
    length = 690
    data = np.zeros([features_num, length], dtype=np.float32)
    cnt = 0
    while True:
        line = file.readline()
        if line:
            features = line.split(' ', features_num - 1)
            cnt_feature = 0
            for i in range(len(attributes_pernum)):
                if attributes_pernum[i] == 1:
                    data[cnt_feature, cnt] = float(features[i])
                    cnt_feature += 1
                else:
                    index = int(features[i])
                    data[cnt_feature + index - 1, cnt] = 1
                    cnt_feature += attributes_pernum[i]
            cnt += 1
        else:
            break
    file.close()

    for i in range(length):
        line = str(int(data[features_num - 1, i]))
        for j in range(features_num - 1):
            line += ' {}:{}'.format(j + 1, data[j, i])
        file_out.write(line + '\n')
    file_out.close()

def onehot_normal_transfer():
    attributes_pernum = [1, 1, 1, 3, 14, 9, 1, 1, 1, 1, 1, 3, 1, 1, 1]

    path = 'data/australian.dat'
    path_out = 'data/australian_onehot_normal_libsvm.dat'
    file = open(path, 'r')
    file_out = open(path_out, 'w')
    features_num = sum(attributes_pernum)
    length = 690
    data = np.zeros([features_num, length], dtype=np.float32)
    cnt = 0
    while True:
        line = file.readline()
        if line:
            features = line.split(' ', features_num - 1)
            cnt_feature = 0
            for i in range(len(attributes_pernum)):
                if attributes_pernum[i] == 1:
                    data[cnt_feature, cnt] = float(features[i])
                    cnt_feature += 1
                else:
                    index = int(features[i])
                    data[cnt_feature + index - 1, cnt] = 1
                    cnt_feature += attributes_pernum[i]
            cnt += 1
        else:
            break
    file.close()

    max_values = np.max(data, axis=1).reshape([features_num, 1])
    min_values = np.min(data, axis=1).reshape([features_num, 1])
    data = (data - np.tile(min_values, [1, length])) / np.tile(np.maximum( max_values - min_values, 1e-10), [1, length])

    for i in range(length):
        line = str(int(data[features_num - 1, i]))
        for j in range(features_num - 1):
            line += ' {}:{}'.format(j + 1, data[j, i])
        file_out.write(line + '\n')
    file_out.close()

if __name__ == '__main__':
    print('data2libsvm_format.py')

    onehot_normal_transfer()




