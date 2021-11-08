import numpy as np
import pandas as pd
import os
import full_arrange


def IIM_of_Li(data):

    [evidence_number, data_frame_number] = data.shape
    for i in range(evidence_number):
        print(str(i) + 'th evidence is ' + str(data[i, :]))

    # 对于识别框架内每一个命题
    new_data = np.zeros((evidence_number, data_frame_number + 1))
    for k in range(data_frame_number):
        # 计算相容矩阵
        R = np.zeros((evidence_number, evidence_number))
        for i in range(evidence_number):
            for j in range(evidence_number):
                R[i,j] = (data[i,k] * data[j,k]) / (np.square(data[i,k]) + np.square(data[j,k])) * 2
                if np.isnan(R[i,j]):
                    R[i,j] = 0
        print(R)
        # 计算绝对相容度
        D = np.zeros((evidence_number,1))
        for i in range(evidence_number):
            D[i,0] = sum(R[i,:]) - 1
        # print("+++++++++")
        # print(D)
        # 计算可信度
        W = np.zeros((evidence_number,1))
        for i in range(evidence_number):
            W[i,0] = D[i,0] / (evidence_number - 1)
        # print("-------------")
        # print(W)

        # 根据可行度重新确定该证据对应命题的权重，对基本概率指派函数进行处理，得到新的mass函数
        for i in range(evidence_number):
            new_data[i, k] = data[i, k] * W[i, 0]

    # 提取不确定事件 \Theta
    for i in range(evidence_number):
        new_data[i, data_frame_number] = 1 - sum(new_data[i, 0:-1])
        print(str(i) + 'th new_evidence is ' + str(new_data[i, :]))

    return new_data


def IIM_of_sun(data):
    [evidence_number, data_frame_number] = data.shape
    for i in range(evidence_number):
        print(str(i) + 'th evidence is ' + str(data[i, :]))

    #冲突矩阵k_m
    k_m = np.zeros((evidence_number,evidence_number))
    for i in range(evidence_number):
        for j in range(evidence_number):
            sum3 = 0
            for k in range(data_frame_number):
                sum3 = sum3 + data[i,k] * (sum(data[j,:]) - data[j,k])
            k_m[i,j] = sum3

    k_sun = 0
    for i in range(evidence_number):
        for j in range(evidence_number):
            if i < j:
                k_sun = k_sun + k_m[i,j]
    #全部证据可信度
    epsilon = k_sun / (evidence_number * (evidence_number - 1) / 2)
    #平均概率
    q = np.zeros((1, data_frame_number))
    for i in range(data_frame_number):
        q[0,i] = sum(data[:,i]) / evidence_number

    # print('epsilon为 ' + str(epsilon))
    # print('q为 ' + str(q))
    return epsilon, q



def DS_fusion_method(data):
    new_data = data
    [evidence_number, data_frame_number] = new_data.shape
    if evidence_number > 3:
        print('can not do')
        return

    #算出全部组合
    combination = full_arrange.full_arrange(range(data_frame_number), evidence_number)
    count = 0
    for i in combination:
        # print(i)
        count = count + 1
    print('组合一共 ' + str(count) + ' 个')

    #先算归一化因子K
    sum1 = 0
    for k in combination:
        count = 0
        small_set = set(k)
        for i in small_set:
            if i in set(range(data_frame_number - 1)):
                count = count + 1
        if count >= 2:
            multi = 1
            for i in range(evidence_number):
                multi = multi * new_data[i,k[i]]
            sum1 = sum1 + multi
    K = 1 - sum1
    print('归一化因子K为 ' + str(K))

    #计算每一个特征融合概率
    fusion = np.zeros((1,data_frame_number))
    for i in range(data_frame_number - 1):
        list1 = [i, data_frame_number - 1]
        small_combination = full_arrange.full_arrange(list1, evidence_number)
        full_info_list = (np.ones((1,evidence_number)) * (data_frame_number - 1)).tolist()[0]
        full_info_list = [int(i) for i in full_info_list]
        small_combination.remove(full_info_list)
        sum2 = 0
        for j in small_combination:
            multi = 1
            for k in range(evidence_number):
                multi = multi * new_data[k,j[k]]
            sum2 = sum2 + multi
        fusion[0,i] = sum2 / K

    multi = 1
    for i in range(evidence_number):
        multi = multi * new_data[i,data_frame_number - 1]
    fusion[0,data_frame_number - 1] = multi / K

    print('DS Fusion ' + str(fusion))
    return fusion, K

def use_DS_method_of_sun(data):
    [epsilon, q] = IIM_of_sun(data)
    add_line = np.zeros((data.shape[0]))
    data_with_all = np.c_[data, add_line]
    [fusion, K] = DS_fusion_method(data_with_all)
    num = fusion.shape[1]
    for i in range(num - 1):
        fusion[0, i] = K * fusion[0, i] + (1 - K) * epsilon * q[0, i]

    fusion[0, num - 1] = (1 - K) * (1 - epsilon)
    #
    # label.append('ALL')
    fusion_all = np.c_['0,2', data_with_all, fusion]
    fusion_all = pd.DataFrame(data=fusion_all,  index=['ProcessData', 'Alert','c', 'Fusion'])
    print(fusion_all)
    return fusion_all



if __name__ == '__main__':
    #测试data
    data = [[0.9, 0.1]]
    data.append([0.8, 0.2])

    data = np.array(data)


    # new_data = [[0.2459, 0.1224, 0.24, 0.3917]]
    # new_data.append([0, 0.2894, 0.06, 0.6506])
    # # new_data.append([0.2951, 0.056, 0.24, 0.4139])
    # new_data = np.array(new_data)

    # data_with_all = [[0.5, 0.2, 0.3, 0]]
    # data_with_all.append([0, 0.9, 0.1, 0])
    # data_with_all.append([0.6, 0.1, 0.3, 0])
    # data_with_all = np.array(data_with_all)

    # 测试Li的方法
    new_data = IIM_of_Li(data)
    DS_fusion_method(new_data)

    # # 测试sun方法
    use_DS_method_of_sun(data)
