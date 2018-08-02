# Author:yu
# 1. 初始化参数：
#     1.1：使用0来初始化参数。
#     1.2：使用随机数来初始化参数。
#     1.3：使用抑梯度异常初始化参数（参见视频中的梯度消失和梯度爆炸）。
# 2. 正则化模型：
#     2.1：使用二范数对二分类模型正则化，尝试避免过拟合。
#     2.2：使用随机删除节点的方法精简模型，同样是为了尝试避免过拟合。
# 3. 梯度校验  ：对模型使用梯度校验，检测它是否在梯度下降的过程中出现误差过大的情况。


import numpy as np
import matplotlib.pyplot as plt
import  sklearn
import  sklearn.datasets
import  init_utils ##第一部分,初始化
import reg_utils ##第二部分,正则化
import gc_utils ##第三部分,梯度校验
#%matplotlib inline #如果你使用的是Jupyter Notebook,请取消注释
plt.rcParams['figure.figsize'] = (7.0,4.0)#set default size of plots图片像素
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
#显示的图表大小为10,图形的插值是以最近为原则,图像颜色是灰色???

train_X,train_Y,test_X,test_Y = init_utils.load_dataset(is_plot = False)
# plt.show()


#尝试三种初始化方法,1初始化为0,2初始化为随机数,3抑制度异常初始化
def model(X,Y,learning_rate = 0.01,num_iterations = 15000,print_cost = True,initialization='he',is_polt = True):
    '''
    实现一个三层的神经网络:linear->relu->linear->relu->linear->sigmoid
    :param X: 输入的数据,维度为(2,要训练/测试的数量)
    :param Y: 标签,[0/1],维度为(1,对应的是输入的数据的标签)
    :param learning_rate: 学习速率
    :param num_iterations: 迭代的次数
    :param print_cost: 是否打印成本值,每次迭代1000次打印一次
    :param initialize: 字符串类型,初始化的类型['zero'|'random'|'he']
    :param is_plot: 是否绘制梯度下降的曲线图
    :return: 
    parameters -学习后的参数
    '''

    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0],10,5,1]

    #选择初始化参数类型
    if initialization == 'zeros':
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == 'random':
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == 'he':
        parameters = initialize_parameters_he(layers_dims)
    else:
        print('错误的初始化参数!程序退出')
        exit

    #开始学习
    for i in range(0,num_iterations):
        #前向传播
        a3 ,cache = init_utils.forward_propagation(X,parameters)

        #计算成本
        cost = init_utils.compute_loss(a3,Y)

        #反向传播
        grads = init_utils.backward_propagation(X,Y,cache)

        #更新参数
        parameters = init_utils.update_parameters(parameters,grads,learning_rate)

        #记录成本
        if i%1000 == 0:
            costs.append(cost)
            #打印成本
            if print_cost:
                print('第'+str(i)+'次迭代,成本值为'+str(cost))

    #学习完毕,绘制成本曲线
    if is_polt:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations(per hundreds)')
        plt.title('Learning rate ='+str(learning_rate))
        plt.show()

        #返回学习完毕后的参数
        return parameters

def initialize_parameters_zeros(layers_dims):
    '''
    将模型的参数全部设置为0
    :param layers_dims: 列表,模型的层数和对应每一层的节点的数量
    :return: parameters-包含所有W和b的字典
    W1-权重矩阵,维度为(layers_dims[1],layers_dims[0])
    b1-偏置向量,维度为(layers_dims[1],1)
    ...
    WL-权重矩阵,维度为(layers_dims[L],layers_dims[L-1)
    bL-偏置向量,维度为(layers_dims[L],1)
    '''

    parameters= {}
    L= len(layers_dims)#网络层数
    for l in range(1,L):
        parameters['W'+str(l)] = np.zeros((layers_dims[l],layers_dims[l-1]))
        parameters['b'+str(l)] = np.zeros((layers_dims[l],1))

        #使用断言确保我的数据格式是正确的
        assert(parameters['W'+str(l)]).shape == (layers_dims[l],layers_dims[l-1])
        assert(parameters['b'+str(l)]).shape == (layers_dims[l],1)

    return parameters

# parameters = model(train_X,train_Y,initialization='zeros',is_polt = True)
#
# print('训练集:')
# predictions_train = init_utils.predict(train_X,train_Y,parameters)
# print('测试集:')
# predictions_test = init_utils.predict(test_X,test_Y,parameters)
#
# print('predivtions_train = '+str(predictions_train))
# print('predictions_test = '+str(predictions_test))
# plt.title('Model with Zeros initialization')
# axes = plt.gca()
# #plt.gcf()和plt.gca()获得，分别表示"Get Current Figure"和"Get Current Axes"
# axes.set_xlim([-1.5,1.5])
# axes.set_ylim([-1.5,1.5])
# init_utils.plot_decision_boundary(lambda x:init_utils.predict_dec(parameters,x.T),train_X,train_Y)
#不能正常工作,没法破坏对称性

##随机初始化
def initialize_parameters_random(layers_dims):
    '''
    :param layers_dims:列表,模型的层数和对应每一层的节点数量 
    :return: 
    parameters-包含了所有W和b的字典
    W1-权重矩阵,维度为(layers_dims[1],layers_dims[0])
    b1-偏置向量,维度为(layers_dims[1],1)
    ...
    WL-权重矩阵,维度为(layers_dims[L],layers_dims[L-1])
    b1-偏置向量,维度为(layers_dims[L],1)
    '''
    np.random.seed(30)#指定随机种子
    parameters = {}
    L = len(layers_dims)

    for l in range(1,L):
        parameters['W'+str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*10##进行10倍缩放
        parameters['b'+str(l)] = np.zeros((layers_dims[l],1))

        #使用断言确保数据格式正确
        assert(parameters['W'+str(l)].shape == (layers_dims[l],layers_dims[l-1]))
        assert(parameters['b'+str(l)]).shape == (layers_dims[l],1)
    return parameters
# parameters = model(train_X,train_Y,initialization='random',is_polt=True)
# print('训练集:')
# predictions_train = init_utils.predict(train_X,train_Y,parameters)
# print('测试集:')
# predictions_test = init_utils.predict(test_X,test_Y,parameters)
# print(predictions_train)
# print(predictions_test)
# plt.title('Model with large random initialization')
# axes = plt.gca()
# axes.set_xlim([-1.5,1.5])
# axes.set_ylim([-1.5,1.5])
# init_utils.plot_decision_boundary(lambda x:init_utils.predict_dec(parameters,x.T),train_X,train_Y)
#初始化系数太大,梯度爆炸,导致一开始的误差很大

##抑制梯度异常初始化
def initialize_parameters_he(layers_dims):
    '''
    :param layers_dims: 列表,模型的层数和对应每一层的节点的数量
    :return: 
    parameters - 包含了权重矩阵W和b的字典
    W1-权重矩阵,维度为(layers_dims[1],layers_dims[0])
    b1-偏置向量,维度为(layers_dims[1],1)
    ...
    WL - 权重矩阵,维度为(layers_dims[L],layers_dims[L-1])
    b1 - 偏置向量,维度为(layers_dims[L],1)
    '''
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for l in range(1,L):
        parameters['W'+str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2/layers_dims[l-1])
        parameters['b'+str(l)] = np.zeros((layers_dims[l],1))

        #使用断言保证数据格式正确
        assert(parameters['W'+str(l)].shape == (layers_dims[l],layers_dims[l-1]))
        assert(parameters['b'+str(l)].shape == (layers_dims[l],1))

    return parameters

parameters = model(train_X, train_Y, initialization = "he",is_polt=True)
print("训练集:")
predictions_train = init_utils.predict(train_X, train_Y, parameters)
print("测试集:")
init_utils.predictions_test = init_utils.predict(test_X, test_Y, parameters)

plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)

# 不同的初始化方法可能导致性能最终不同
#
# 随机初始化有助于打破对称，使得不同隐藏层的单元可以学习到不同的参数。
#
# 初始化时，初始值不宜过大。
#
# He初始化搭配ReLU激活函数常常可以得到不错的效果