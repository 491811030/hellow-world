# Author:yu
import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases
from dnn_utils import sigmoid,sigmoid_backward,relu,relu_backward
import lr_utils

np.random.seed(1)

#对于一个两层的神经网络,神经网络的输入输出层不计入层数,神经网络层数等于隐藏层加一

#初始化函数
def initialize_parameters(n_x,n_h,n_y):
    '''
    此函数是为了初始化两层网络参数而使用的函数
    :param n_x: 输入层节点的数量
    :param n_h: 隐藏层节点数量
    :param n_y: 输出层节点数量
    :return: 
    parameters - 包含你的参数的python字典
    W1-权重矩阵,维度为(n_h,n_x)
    b1-偏向量,维度为(n_h,1)
    W2-权重矩阵,维度为(n_y,n_h)
    b2-偏向量,维度为(n_y,1)
    '''
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))
    #使用断言保证数据格式正确
    assert(W1.shape == (n_h,n_x))
    assert(b1.shape == (n_h,1))
    assert(W2.shape == (n_y,n_h))
    assert(b2.shape == (n_y,1))
    parameters = {
        'W1':W1,
        'b1':b1,
        'W2':W2,
        'b2':b2
    }
    return parameters

##初始化多层网络参数
def initialize_parameters_deep(layers_dims):
    '''
    此函数是为了初始化多层网络参数而使用的函数
    :param layers_dims: 包含网络中每个图层的节点数量的列表
    :return: 
    :parameters - 包含参数'W1','b1',...,'WL','bL'的字典:
        W1-权重矩阵,维度为(layers_dims[1],layers_dims[1-1])
        b1-偏向量,维度为(layers_dims[1],1)
    '''
    np.random.seed(3)
    parameters = {}
    L=len(layers_dims)
    for l in range(1,L):
        parameters['W'+str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])/np.square(layers_dims[l-1])
        parameters['b'+str(l)] = np.zeros((layers_dims[l],1))

        #确保数据格式正确
        assert(parameters['W'+str(l)].shape == (layers_dims[l],layers_dims[l-1]))
        assert(parameters['b'+str(l)].shape == (layers_dims[l],1))
    return parameters

##前向传播函数三步骤:1.linear,2.linear->activation,3.[linear->relu]*(L-1)->linear->sigmoid

def linear_forward(A,W,b):
    '''
    实现前向传播的线性部分.
    :param A: 来自上一层(或输入数据)的激活,维度为(上一层节点的数量,示例的数量
    :param W: 权重矩阵,numpy数组,维度为(当前图层的节点数量,前一层的节点数量
    :param b: 偏向量,numpy向量,维度为(当前图层节点数量,1)
    :return: 
    Z-激活函数的输入,也称为预激活参数
    cache-一个包含'A','W','b'的字典,存储这些变量以有效的计算后向传递
    '''
    Z = np.dot(W,A)+b
    assert(Z.shape == (W.shape[0],A.shape[1]))
    cache = (A,W,b)
    return Z,cache


##线性激活部分[linear->activation]

def linear_activation_forward(A_prev,W,b,activation):
    '''
    实现linear->activation这一层的前向传播
    :param A_prev: 来自上一层(或输入层)的激活,维度为(上一层的节点数量,示例数)
    :param W:权重矩阵,numpy数组,维度为(当前的节点数量,前一层的大小
    :param b: 偏向量,numpy阵列,维度为(当前层的节点数量,1)
    :param activation: 选择在此层中使用的就激活函数名,字符串类型,['sigmoid'|'relu']
    :return: 
    A-激活函数的输出,也称为激活后的值
    cache-一个包含'linear_cache'和'activation_cache'的的字典，我们需要存储它以有效地计算后向传递
    '''
    if activation == 'sigmoid':
        Z,linear_cache= linear_forward(A_prev,W,b)
        A,activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = relu(Z)

    assert (A.shape == (W.shape[0],A_prev.shape[1]))
    cache = (linear_cache,activation_cache)
    return A,cache

##多层模型的前向传播计算模型代码:
def L_modek_forward(X,parameters):
    '''
    实现[linear->relu]*(l-1)->linear->sigmoid计算前向传播
    :param x: 数据,numpy数组,维度为(输入节点数量,示例数)
    :param parameters: initialize_parameters_deep()的输出
    :return: AL-最后的激活值
    caches-包含以下内容的缓存列表:
        linear_relu_forward()的每个cache(有L-1个,索引为0到L-2)
        linear_sigmoid_forward()的cache(中有一个,索引为L-1)
    '''
    caches = []
    A=X
    L = len(parameters)//2
    for i in range(1,L):
        A_prev = A
        A,cache=linear_activation_forward(A_prev,parameters['W'+str(i)],parameters['b'+str(i)],'relu')
        caches.append(cache)
    AL,cache = linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],'sigmoid')
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))
    return AL,caches


#计算成本
def compute_cost(AL,Y):
    '''
    实施公式的成本函数
    :param AL: 与标签预测相对应的概率向量,维度为(1,示例数量)
    :param Y: 标签向量(例如:如果不是猫,则为0,如果是猫则为1),维度为(1,数量)
    :return: 
    cost-交叉熵成本
    '''
    m=Y.shape[1]
    cost = -np.sum(np.multiply(np.log(AL),Y)+np.multiply(np.log(1-AL),1-Y))/m
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost

##反向传播
##需要用dZ[l]来计算三个输出(dW[l],db[l],dA[l])
##与前向传播类似,用三个步骤来构建反向传播:
##linear后向计算
##linear->activation后向计算,其中acactivationjisuantrluhouzhesigmoid的结果
##[linear->relu]*(L-1)->sigmoid后向计算

def linear_backward(dZ,cache):
    '''
    为单层实现反向传播的线性部分(第L层)
    :param dZ: 相当于(当前第l层)线性输出的成本梯度
    :param cache: 来自当前层前向传播的值的元组(A_prev,W,b)
    :return: 
    dA_prev-相当于激活(前一层l-1)的成本梯度,与A_prev维度相同
    dW-相当于W(当前层l)的成本梯度,与W的维度相同
    db-相当于b(当前层)等成本梯度,与b维度形同
    '''
    A_prev,W,b = cache
    m=A_prev.shape[1]
    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ,axis = 1,keepdims = True)/m
    dA_prev = np.dot(W.T,dZ)

    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    return dA_prev,dW,db

#线性激活部分
def linear_activation_backward(dA,cache,activation='relu'):
    '''
    实现linear->activation层的后向传播
    :param dA: 当前层l的激活后的梯度值
    :param cache: 我们存储的用于有效计算反向传播的值的元组(值为linear_cache,activation_cache)
    :param activation: 要在此层中使用的激活函数名,字符串类型,['sigmoid'|'relu]
    :return: 
    dA_prev-相当于激活(前一层)的 成本梯度值,与W的维度相同
    dW-相当于W(当前层l)的成本梯度值,与W的维度相同
    db-相当于b(当前层l)的成本梯度值,与b的维度相同
    '''
    linear_cache,activation_cache=cache
    if activation == 'relu':
        dZ = relu_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
    return dA_prev,dW,db

#多层后向传播
def L_model_backward(AL,Y,caches):
    '''
    对[linear->relu]*(l-1)->linear->sigmoid组执行反向传播,就是多层网络的后向传播
    :param AL: 概率向量,正向传播的输出(L_model_forward())
    :param Y: 标签向量(例如:如果不是猫,则为0,如果是猫则为1),维度为(1,数量)
    :param caches: 包含以下内容的cache列表:
                linear_activation_forward('relu)的cache,不包含输出层
                linear_activation_forward('sigmoid')的cache
    :return: 
        grads-具有梯度值的字典
            grads['dA'+str(l)]=...
            grads['dW'+str(l)]=...
            grads['db'+str(l)]=...
    '''
    grads = {}
    L=len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y,AL)-np.divide(1-Y,1-AL))

    current_cache = caches[L-1]
    grads['dA'+str(L)],grads['dW'+str(L)],grads['db'+str(L)]=linear_activation_backward(
        dAL,current_cache,'sigmoid')
    for l in reversed(range(L-1)):
        current_cache=caches[l]
        dA_prev_temp,dW_temp,db_temp = linear_activation_backward(grads['dA'+str(l+2)],current_cache,'relu')
        grads['dA'+str(l+1)] = dA_prev_temp
        grads['dW'+str(l+1)] = dW_temp
        grads['db'+str(l+1)] = db_temp

    return grads

##更新参数
def update_parameters(parameters,grads,learning_rate):
    '''
    使用梯度下降更新参数
    :param parameters: 包含参数的字典
    :param grads: 包含梯度值的字典,是L_model_backward的输出
    :param learning_rate: 学习率
    :return: 
    parameters-包含更新参数的字典
            参数['W'+str(l)]=...
            参数['b'+str(l)]=...
    '''
    L = len(parameters)//2#整除
    for l in range(L):
        parameters['W'+str(l+1)] = parameters['W'+str(l+1)]-learning_rate*grads['dW'+str(l+1)]
        parameters['b'+str(l+1)] = parameters['b'+str(l+1)]-learning_rate*grads['db'+str(l+1)]

    return parameters


##搭建两层神经网络

def two_layer_model(X,Y,layer_dims,learning_rate=0.0075,num_iterations=3000,print_cost = False,isPlot = True):
    '''
    实现一个两层的神经网络,[linear->relu]->[liinear->sigmoid]
    :param X: 输入的数据,维度为(n_x,例子数)
    :param Y: 标签,向量,0为非猫,1为猫,维度为(1,数量)
    :param layer_dims: 层数的向量,维度为(n_x,n_h,n_y)
    :param larean_rate: 学习率
    :param num_iterations: 迭代次数
    :param print_cost: 是否打印成本值,每100次打印一次
    :param isPlot: 是否会之出误差值的图谱
    :return: 
    parametres-一个包含W1,b1,W2,b2的字典变量
    '''
    np.random.seed(1)
    grads={}
    costs = []
    (n_x,n_h,n_y) = layer_dims
    #初始化参数
    parameters = initialize_parameters(n_x,n_h,n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    #开始迭代
    for i in range(0,num_iterations):
        #前向传播
        A1,cache1 = linear_activation_forward(X,W1,b1,'relu')
        A2,cache2 = linear_activation_forward(A1,W2,b2,'sigmoid')

        #计算成本
        cost = compute_cost(A2,Y)

        #后向传播
        ##初始化后向传播
        dA2 = -(np.divide(Y,A2)-np.divide(1-Y,1-A2))

        ##向后传播,输入:'dA2,cache2,cache1'.输出'dA1,dW2,db2;还有dA0,dW1,db1'.
        dA1,dW2,db2 = linear_activation_backward(dA2,cache2,'sigmoid')
        dA0,dW1,db1 = linear_activation_backward(dA1,cache1,'relu')

        ##向后传播的数据保存到grads
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        #更新参数
        parameters = update_parameters(parameters,grads,learning_rate)
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        #打印成本值
        if i%100 == 0:
            #记录成本
            costs.append(cost)
            #是否打印成本值
            if print_cost:
                print('第',i,'次迭代,成本值为:',np.squeeze(costs))

    #迭代完成,根据条件绘制图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations(per tens)')
        plt.title('learning rate = '+str(learning_rate))
        plt.show()

    #返回parameters
        return parameters

train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = lr_utils.load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y

n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x,n_h,n_y)

parameters = two_layer_model(train_x, train_set_y, layer_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True,isPlot=True)


##预测
def predict(X,y,parameters):
    '''
    该函数用于预测L层神经网络的效果
    :param X: 测试集
    :param y: 标签
    :param parameters:训练模型的参数 
    :return: 
    p-给定数据集X的预测
    '''
    m = X.shape[1]
    n = len(parameters)//2#神经网络层数
    p = np.zeros((1,m))

    #根据参数前向传播
    probas ,caches = L_modek_forward(X,parameters)

    for i in range(0,probas.shape[1]):
        if probas[0,i]>0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    print('准确度为:'+str(float(np.sum((p==y))/m)))
    return p

