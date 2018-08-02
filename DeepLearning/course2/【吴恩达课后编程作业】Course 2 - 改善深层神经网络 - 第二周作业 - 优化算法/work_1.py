# Author:yu
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io
import sklearn
import sklearn.datasets

import opt_utils
import testCase

#%matplotlib inline #如果你用的是Jupyter Notebook请取消注释
plt.rcParams['figure.figsize'] = (7.0,4.0)#set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

##梯度下降
def update_parameters_with_gd(parameters,grads,learning_rate):
    '''
    使用梯度下降更新参数
    :param parameters: 字典,包含了要更新的参数:
        parameters['W'+str(l)] = Wl
        parameter['b'+str(l)] = bl
    :param grads: 字典,包含了每一个梯度值用以更新参数
        grads['dW'+str(l)] = dWl
        grads['db'+str(l)] = dbl
    :param learning_rate: 学习率
    :return: 
        parameters - 字典,包含了更新后的参数
    '''

    L = len(parameters)//2#神经网络层数

    #更新每个参数
    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate*grads['db'+str(l+1)]

    return parameters


# ##随机梯度下降
# #仅作比较,不运行
# #批量梯度下降,又叫梯度下降
#
# X = data_input
# Y = labels
#
# parameters = initialize_parameters(layers_dims)
# for i range(0,num_iterations):
#     #前向传播
#     A,cache = forward_propagation(X,parameters)
#     #计算损失
#     cost = compute_cost(A,Y)
#     #反向传播
#     grads = backward_propagation(X,Y,cache)
#     #更新参数
#     parameters = update_parameters(parameters,grads)
#
# #随机梯度下降算法
# X = data_input
# Y = label
# parameters = initialize_parameters(layers_dims)
# for i in range(0,num_iterations):
#     for j in m:
#         #前向传播
#         A,cahce = forward_propagation(X,parameters)
#         #计算成本
#         cost = compute_cost(A,Y)
#         #后向传播
#         grads = backward_propagation(X,Y,cahce)
#         #更新参数
#         parameters = update_parameters(parameters,grads)


###mini-batch步骤:1,打乱训练集的次序,2,切分成2的幂大小
def random_mini_batches(X,Y,mini_batch_size = 64,seed = 0 ):
    '''
    从(X,Y)中创建一个随机的mini-batch列表
    :param X: 输入数据,维度为(输入节点数量,样本数量)
    :param Y: 对应的是X的标签,[1|0](蓝|红),维度为(1,样本的数量)
    :param mini_batch_size: 每个mini-batch的样本数量
    :param seed: 
    :return: 
    mini-batches  -一个同步列表,维度为(mini_batch_X,mini_batch_Y)
    '''
    np.random.seed(seed)#指定随机种子
    m = X.shape[1]
    mini_batches = []

    #第一步,打乱顺序
    permutation = list(np.random.permutation(m))#它会返回一个长度为m的随机数组，且里面的数是0到m-1
    shuffled_X = X[:,permutation]#将每一列的数据a按permutation的顺序来重新排列
    shuffled_Y = Y[:,permutation].reshape((1,m))

    '''
    X和Y如何根据permutation来打乱顺序的
    x = np.array([[1,2,3,4,5,6,7,8,9],
                    [9,8,7,6,5,4,3,2,1]])
    y = np.array([1,0,1,0,1,0,1,0,1]])
    
    random_mini_batches(x,y
    permutation = [7,2,1,4,8,6,3,0,5]
    shuffled_X = [[8,3,2,5,9,7,4,1,6]
                    [2,7,8,5,1,3,6,9,4]]
    shauffled_Y = [[0 1 0 1 1 1 0 1 0]]
    '''

    #第二部,分割
    num_complete_minibatches = math.floor(m/mini_batch_size) #把你的训练集分割成多少份,请注意，如果值是99.99，那么返回值是99，剩下的0.99会被舍弃
    for k in range(0,num_complete_minibatches):
        mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    #如果训练集的大小刚好是mini_batch_size的整数倍,那么这里已经处理完了
    #如果训练集的大小不是mini_batch_size的整数倍,那么最后肯定会剩下一些,我们要把他处理了

    if m%mini_batch_size!=0:
        #获取最后剩余部分
        mini_batch_X = shuffled_X[:,mini_batch_size*num_complete_minibatches:]
        mini_batch_Y = shuffled_Y[:,mini_batch_size*num_complete_minibatches:]

        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

# #测试random_mini_batches
# print("-------------测试random_mini_batches-------------")
# X_assess,Y_assess,mini_batch_size = testCase.random_mini_batches_test_case()
# mini_batches = random_mini_batches(X_assess,Y_assess,mini_batch_size)
#
# print("第1个mini_batch_X 的维度为：",mini_batches[0][0].shape)
# print("第1个mini_batch_Y 的维度为：",mini_batches[0][1].shape)
# print("第2个mini_batch_X 的维度为：",mini_batches[1][0].shape)
# print("第2个mini_batch_Y 的维度为：",mini_batches[1][1].shape)
# print("第3个mini_batch_X 的维度为：",mini_batches[2][0].shape)
# print("第3个mini_batch_Y 的维度为：",mini_batches[2][1].shape)

#包含动量的梯度下降

#初始化
def initialize_velocity(parameters):
    '''
    初始化速度,velocity是一个字典
            -keys:'dW1','db1',...,'dWL','dbL'
            -values:与相应的梯度/参数维度相同的值为零的矩阵
    :param parameters: 一个字典,包含以下参数
                parameters['W'+str(l)] = Wl
                parameters['b'+str(l)] = bl
    :return: 
        v - 一个字典变量,包含了以下参数:
            v['dW'+str(l)] = dWl的速度
            v['db'+str(l)]=dbl的速度
    '''
    L = len(parameters)//2#神经网络层数
    v = {}
    for l in range(L):
        v['dW' +str(l+1)] = np.zeros_like(parameters['W'+str(l+1)])
        v['db'+str(l+1)] = np.zeros_like(parameters['b'+str(l+1)])

    return v

#更新参数
def update_parameters_with_momentun(parameters,grads,v,beta,learning_rate):
    '''
    使用动量更新参数
    :param parameters:一个字典类型的变量,包含以下字段
                parameters['W'+str(l(] = Wl
                parameters['b'+str(l)] = bl
    :param grads: 一个包含梯度值的字典变量,具有以下字段
                grads['dW'+str(l)] = dWl
                grads['db'+str(l)] = dbl
    :param v: 包含当前速度的字典变量,具有以下字段:
            v['dW'+str(l)] = ...
            v['db'+str(l)] = ...
    :param beta: 超参数,动量,实数
    :param learning_rate: 学习率,实数
    :return: 
    parameters = 更新后的参数字典
    v -包含了更新后的速度变量
    '''
    L = len(parameters)//2
    for l in range(L):
        #计算速度
        v['dW'+str(l+1)] = beta* v['dW'+str(l+1)]+(1-beta)*grads['dW'+str(l+1)]
        v['db'+str(l+1)] = beta*v['db'+str(l+1)]+(1-beta)*grads['db'+str(l+1)]

        #更新参数
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)]

    return parameters,v

#Adam算法
def initialize_adam(parameters):
    '''
    初始化v和s,他们都是字典类型的向量,都包含了以下字段
    -key:'dW1','db1',...'dWL','dbL'
    -values:与对应的梯度/参数相同维度的值为零的numpy矩阵
    
    :param parameters: -包含了以下参数的字典变量
            parameters['W'+str(l)] = W1
            parameters['b'+str(l)] = bl
    :return: 
    v - 包含梯度的指数加权平均值,字段如下:
        v['dW'+str(l)] = ...
        v['db'+str(l)] = ...
    s - 包含平方梯度的指数加权平均值,字段如下:
        s['dW'+str(l)] = ...
        s['db'+str(l)] = ...
    '''

    L = len(parameters)//2
    v= {}
    s = {}

    for l in range(L):
        v['dW'+str(l+1)] = np.zeros_like(parameters['W'+str(l+1)])
        v['db'+str(l+1)] = np.zeros_like(parameters['b'+str(l+1)])

        s['dW'+str(l+1)] = np.zeros_like(parameters['W'+str(l+1)])
        s['db'+str(l+1)] = np.zeros_like(parameters['b'+str(l+1)])

    return(v,s)
# #测试initialize_adam
# print("-------------测试initialize_adam-------------")
# parameters = testCase.initialize_adam_test_case()
# v,s = initialize_adam(parameters)

# print('v["dW1"] = ' + str(v["dW1"]))
# print('v["db1"] = ' + str(v["db1"]))
# print('v["dW2"] = ' + str(v["dW2"]))
# print('v["db2"] = ' + str(v["db2"]))
# print('s["dW1"] = ' + str(s["dW1"]))
# print('s["db1"] = ' + str(s["db1"]))
# print('s["dW2"] = ' + str(s["dW2"]))
# print('s["db2"] = ' + str(s["db2"]))

def update_parameters_with_adam(parameters,grads,v,s,t,learning_rate = 0.01,beta1 = 0.9,beta2 =0.999,epsilon = 1e-8):
    '''
    使用adam更新参数
    :param parameters: 包含了一下字段的字典
    parameters['W'+str(l)] = Wl
    parameter['b'+str(l)] = b1
    :param grads: 包含了梯度值的字典,有以下key值:
    grads['dW'+str(l)] = dWl
    grads['db'+str(l)] = dbl
    :param v: Adam的变量,第一梯度的移动平均值,是一个字典类型的变量
    :param s: Adam的变量,平方梯度的移动平均值,是一个字典类型的变量
    :param t:当前迭代的次数
    :param learning_rate: 学习率
    :param beta1: 动量,超参数,用于第一阶段,使得曲线的Y值不从0开始(参见天气数据的那个图)
    :param beta2: RMSprop的一个参数,超参数
    :param epsilon: 防止除零操作(分母为0)
    :return: 
    parameters-更新后的参数
    v-第一个梯度的移动平均值,是一个字典类型的变量
    s-平方梯度的移动平均值,是一个字典类型的变量
    '''
    L = len(parameters)//2
    v_corrected = {}#偏差修正后的值
    s_corrected = {}#偏差修正后的值

    for l in range(L):
        #梯度移动平均值,输入:'v,grads,beta1',输出:'v'
        v['dW'+str(l+1)] = beta1*v['dW'+str(l+1)]+(1-beta1)*grads['dW'+str(l+1)]
        v['db'+str(l+1)] = beta1*v['db'+str(l+1)]+(1-beta1)*grads['db'+str(l+1)]

        #计算第一阶段的偏差修正后的估计值,输入'v,beta1,t',输出:'v_corrected'
        v_corrected['dW'+str(l+1)]=v['dW'+str(l+1)]/(1-np.power(beta1,t))
        v_corrected['db'+str(l+1)]=v['db'+str(l+1)]/(1-np.power(beta1,t))

        #计算平方梯度的移动平均值,输入:'s,grads,beta2',输出's'
        s['dW'+str(l+1)] = beta2*s['dW'+str(l+1)] +(1-beta2)*np.square(grads['dW'+str(l+1)])
        s['db'+str(l+1)] = beta2 *s['db'+str(l+1)]+(1-beta2)*np.square(grads['db'+str(l+1)])

        #计算第二阶段的偏差修正后的估计值,输入:'s,beta2,t',输出:'s_corrected'
        s_corrected['dW'+str(l+1)] = s['dW'+str(l+1)]/(1-np.power(beta2,t))
        s_corrected['db'+str(l+1)] = s['db'+str(l+1)]/(1-np.power(beta2,t))

        #更新参数,输入:'parameters,learning_rate,v_vorrected,s_corrected,epsilon'.输出:'parameters'
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate*(v_corrected['dW'+str(l+1)]/np.sqrt(s_corrected['dW'+str(l+1)]+epsilon))
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate*(v_corrected['db'+str(l+1)]/np.sqrt(s_corrected['db'+str(l+1)]+epsilon))

    return(parameters,v,s)

#测试update_with_parameters_with_adam
# print("-------------测试update_with_parameters_with_adam-------------")
# parameters , grads , v , s = testCase.update_parameters_with_adam_test_case()
# update_parameters_with_adam(parameters,grads,v,s,t=2)

# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print('v["dW1"] = ' + str(v["dW1"]))
# print('v["db1"] = ' + str(v["db1"]))
# print('v["dW2"] = ' + str(v["dW2"]))
# print('v["db2"] = ' + str(v["db2"]))
# print('s["dW1"] = ' + str(s["dW1"]))
# print('s["db1"] = ' + str(s["db1"]))
# print('s["dW2"] = ' + str(s["dW2"]))
# print('s["db2"] = ' + str(s["db2"]))

##加载数据集
train_X,train_Y = opt_utils.load_dataset(is_plot = False)

##定义模型
def model(X,Y,layers_dims,optimizer,learning_rate = 0.0007,
          mini_batch_size = 64,beta = 0.9,beta1 = 0.9,beta2 = 0.999,
          epsilon = 1e-8,num_epochs = 10000,print_cost = True,is_plot = True ):
    '''
    可以运行在不同优化器模式下的三层神经网络模型.
    :param X: 输入数据,维度为(2,输入数据集里面样本数量))
    :param Y: 与X对应的标签
    :param layers_dims: 包含层数和节点数量的列表
    :param optimizer: 字符出类型的参数,用于选择优化类型,['gd'|'momentum'|'adam']
    :param learning_rate: 学习率
    :param mini_batch_size: 每个小批量数据集的大小
    :param beta: 用于动量优化的一个超参数
    :param beta1:用于计算梯度后的指数衰减的估计的超参数 
    :param beta2: 用于计算平方梯度后的指数衰减的估计的超参数
    :param epsilon: 用于在Adam中避免除零操作的超参数,一般不更改
    :param num_epochs: 整个训练集的遍历次数
    :param print_cost: 是否打印误差值,每遍历1000次数据打印一次,但是每100次记录一个误差值,又称每1000代打印一次
    :param is_plot: 是否会制出曲线图
    :return: 
    parameters - 包含了学习后的参数
    '''
    L = len(layers_dims)
    costs = []
    t = 0#每学习完一个minibatch就增加1
    seed = 10#随机种子

    #初始化参数
    parameters = opt_utils.initialize_parameters(layers_dims)

    #选择优化器
    if optimizer == 'gd':
        pass #不使用任何优化器,直接使用梯度下降算法
    elif optimizer == 'momentum':
        v = initialize_velocity(parameters)#使用动量
    elif optimizer == 'adam':
        v,s = initialize_adam(parameters)#使用adam优化
    else:
        print('optimizer参数错误,程序退出')
        exit(1)

    #开始学习
    for i in range(num_epochs):
        #定义随机minibatches,我们每次便利数据集之后增加种子以重新排列数据,使每次数据的顺序都不同
        seed = seed+1
        minibatches = random_mini_batches(X,Y,mini_batch_size,seed)

        for minibatch in minibatches:
            #选择一个minibatch
            (minibatch_X,minibatch_Y) = minibatch

            #前向传播
            A3,cache = opt_utils.forward_propagation(minibatch_X,parameters)

            #计算误差
            cost = opt_utils.compute_cost(A3,minibatch_Y)

            #反向传播
            grads = opt_utils.backward_propagation(minibatch_X,minibatch_Y,cache)

            #更新参数
            if optimizer == 'gd':
                parameters = update_parameters_with_gd(parameters,grads,learning_rate)
            elif optimizer=='momentum':
                parameters,v = update_parameters_with_momentun(parameters,grads,v,beta,learning_rate)
            elif optimizer == 'adam':
                t = t+1
                parameters,v,s = update_parameters_with_adam(parameters,grads,v,s,t,learning_rate,beta1,beta2,epsilon)
        #记录误差值
        if i%100 == 0:
            costs.append(cost)
            #是否打印误差值
            if print_cost and i%1000 == 0:
                print('第'+str(i)+'次遍历整个数据集,当前误差值:'+str(cost))
    #是否绘制曲线图
    if is_plot:
        plt.plot(costs)
        plt.xlabel('epochs(per 100)')
        plt.ylabel('cost')
        plt.title('Learning rate = '+str(learning_rate))
        plt.show()

    return parameters

#使用普通的梯度下降
# layers_dims = [train_X.shape[0],5,2,1]
# parameters = model(train_X, train_Y, layers_dims, optimizer="gd",is_plot=True)

# #预测
# predictions = opt_utils.predict(train_X,train_Y,parameters)
#
# #绘制分类图
# plt.title('Model with Gradient Descent optimization')
# axes = plt.gca()#当前的图表和子图可以使用plt.gcf()和plt.gca()获得，分别表示Get Current Figure和Get Current Axes
# axes.set_xlim([-1.5,2.5])
# axes.set_ylim([-1,1.5])
# opt_utils.plot_decision_boundary(lambda x:opt_utils.predict_dec(parameters,x.T),train_X,train_Y)

# layers_dims = [train_X.shape[0],5,2,1]
# #使用动量的梯度下降
# parameters = model(train_X,train_Y,layers_dims,beta=0.9,optimizer='momentum',is_plot=True)
#
# #预测
# preditions = opt_utils.predict(train_X,train_Y,parameters)
#
# #绘制分类图
# plt.title("Model with Momentum optimization")
# axes = plt.gca()
# axes.set_xlim([-1.5, 2.5])
# axes.set_ylim([-1, 1.5])
# opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)


#Adam优化后的梯度下降
layers_dims = [train_X.shape[0],5,2,1]
#使用Adam优化的梯度下降
parameters = model(train_X,train_Y,layers_dims,optimizer='adam',is_plot=True)

# #预测
# preditions = opt_utils.predict(train_X,train_Y,parameters)
#
# #绘制分类图
# plt.title("Model with Adam optimization")
# axes = plt.gca()
# axes.set_xlim([-1.5, 2.5])
# axes.set_ylim([-1, 1.5])
# opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(parameters, x.T), train_X, train_Y)
