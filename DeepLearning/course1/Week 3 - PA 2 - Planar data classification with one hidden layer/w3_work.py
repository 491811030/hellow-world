# Author:yu
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import  plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets
#%matplotlib inline
np.random.seed(1)#设置一个随机种子，以保证接下来的步骤中于作业答案一致
X,Y = load_planar_dataset()
# plt.scatter(X[0,:],X[1,:],c=Y,s = 40,cmap = plt.cm.Spectral)
# #Y是0或1,c表示点的颜色，s表示点的大小，cmap表示绘图的主题得到c的颜色
# plt.show()
shape_X = X.shape
shape_Y =Y.shape
m = Y.shape[1]
print('X的维度为：'+str(shape_X))
print('Y的维度为：'+str(shape_Y))
print('数据集里面的数据有:'+str(m)+'个')
# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T,Y.T)
# plot_decision_boundary(lambda x:clf.predict(x),X,Y)
# plt.title('Logistic Regression')#图标题
# LR_predictions = clf.predict(X.T)#预测结果
# print('逻辑回归的准确性：%d'%float((np.dot(Y,LR_predictions)+np.dot(1-Y,1-LR_predictions))
#                           /float(Y.size)*100)+'%'+'(正确标记的数据点所占的百分比')
#

##构建神经网络：1定义神经网络结构（输入单元的数量，隐藏单元的数量等），2初始化模型参数，3循环：实施前向传播，计算损失，实现后向传播，哥您参数
def layer_size(X,Y):
    '''
    :param X:输入数据集，维度为（输入的数量，训练/测试的数量 
    :param Y: 标签，维度为（输出的数量，训练/测试的数量）
    :return: 
    n_x-输入层的数量
    n_h-隐藏层的数量
    n_y-输出层的数量
    '''
    n_x = X.shape[0]#输入层
    n_h = 4#隐藏层，硬编码为4
    n_y = Y.shape[0]#输出层
    return(n_x,n_h,n_y)

#初始化模型参数
def initialize_parameters(n_x,n_h,n_y):
    '''
    :param n_x:输入层节点数量 
    :param n_h: 隐藏层节点数量
    :param n_y: 输出层节点数量
    :return: 
    parameters:包含参数的字典
    W1-权重矩阵，维度为（n_h,n_x)
    b1-偏向量,维度为(n_h,1)
    W2-权重矩阵,维度为（n_y,n_h)
    b2 - 偏向量,维度为（n_y,1)
    '''
    np.random.seed(2)#指定一个随机种子，以便输出一致
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros(shape=(n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros(shape = (n_y,1))
    #使用断言确保数据正确
    assert(W1.shape == (n_h,n_x))
    assert(b1.shape == (n_h,1))
    assert(W2.shape ==(n_y,n_h))
    assert(b2.shape == (n_y,1))
    parameters = {
        'W1':W1,
        'b1':b1,
        'W2':W2,
        'b2':b2
    }
    return parameters

#前向传播，步骤：使用字典类型的parameters(它是initialize_parameters()的输出）检索每个参数
#实现前向传播，计算Z[1],A[1],Z[2]和A[2]（训练集里面所有例子的预测向量）。
#反向传播所需的值存储在'cache'中，cache将作为反向传播函数的输入

def forward_propagation(X,parameters):
    '''
    :param X:维度为(n_x,m)的输入数据. 
    :param parameters: 初始化函数(initialize_parameters)
    :return: 
    A2 - 使用sigmoid()函数计算的第二次激活后的数值
    cache-包含'Z1','A1','Z2','A2'的字典类型变量
    '''
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    #前向计算A2
    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    #使用断言确保数据格式正确
    assert(A2.shape ==(1,X.shape[1]))
    cache = {
        'Z1':Z1,
        'A1':A1,
        'Z1':Z2,
        'A2':A2
    }
    return (A2,cache)

##构建计算成本函数
def compute_cost(A2,Y,parameters):
    '''
    :param A2:使用sigmoid()函数计算第二次激活后的数值 
    :param Y: 'True'标签向量,维度为(1,数量)
    :param parameters: 一个包含W1,B1,W2,B2的字典类型的变量
    :return: 
    成本-交叉熵公式给出的方程
    '''
    m=Y.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']

    #计算成本
    logprobs = np.multiply(np.log(A2),Y)+np.multiply((1-Y),np.log(1-A2))
    cost = -np.sum(logprobs)/m
    cost = float(np.squeeze(cost))

    assert(isinstance(cost,float))
    return cost

##后向传播
def backward_propagation(parameters,cache,X,Y):
    '''
    使用数学公式搭建反向传播函数
    :param parameters: 包含我们的参数的一个字典类型参数
    :param cache: 包含'Z1','A1','Z2','A2'的字典类型变量
    :param X: 输入数据,维度为(2,数量)
    :param Y: 'True'标签,维度为(1,数量)
    :return: 
    grads - 包含w和b的导数的一个字典类型的变量
    '''
    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1' ]
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = (1/m)*np.sum(dZ2,axis = 1,keepdims = True)
    dZ1 = np.multiply(np.dot(W2.T,dZ2),1 - np.power(A1,2))
    dW1 = (1/m)*np.dot(dZ1,X.T)
    db1 = (1/m)*np.sum(dZ1,axis = 1,keepdims = True)
    grads = {
        'dW1':dW1,
        'db1':db1,
        'dW2':dW2,
        'db2':db2
    }
    return grads

##更新参数
def update_parameters(parameters,grads,learing_rate = 1.2):
    '''
    s使用梯度下降更新参数
    :param parameters: 包含参数的字典类型变量
    :param grads: 包含导数值的字典类型的变量
    :param learing_rate: 学习速率
    :return: 
    :parameters-包含更新参数的字典类型的变量
    '''
    W1,W2 = parameters['W1'],parameters['W2']
    b1,b2 = parameters['b1'],parameters['b2']

    dW1,dW2 = grads['dW1'],grads['dW2']
    db1,db2 = grads['db1'],grads['db2']

    W1 = W1 - learing_rate*dW1
    b1 = b1 - learing_rate*db1
    W2 = W2 - learing_rate*dW2
    b2 = b2 - learing_rate*db2

    parameters={
        'W1':W1,
        'b1':b1,
        'W2':W2,
        'b2':b2
    }
    return parameters

##整合到nn_model中
def nn_model(X,Y,n_h,num_iterations,print_cost=False):
    '''
    :param X:数据集,维度为(2,示例数) 
    :param Y: 标签,维度为(1,示例数)
    :param n_h: 隐藏层的数量
    :param num_iterations: 梯度下降循环中的迭代次数
    :param print_cost: 如果为True,则每1000次迭代打印一次成本数值
    :return: 
    parameters - 模型学习的参数,它们可以用来进行预测
    '''
    np.random.seed(3)#指定随机种子
    n_x = layer_size(X,Y)[0]
    n_y = layer_size(X,Y)[2]

    parameters = initialize_parameters(n_x,n_h,n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    for i in range(num_iterations):
        A2 ,cache = forward_propagation(X,parameters)
        cost = compute_cost(A2,Y,parameters)
        grads = backward_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,learing_rate=0.5)
        if print_cost:
            if i%1000==0:
                print('第',i,'次循环,成本为:'+str(cost))
    return parameters

##预测
def predict(parameters,X):
    '''
    使用学习的的参数,为X中的每个示例预测一个类
    :param parameters: 包含参数的字典类型的变量
    :param X: 输入数据(n_x,m)
    :return: 
    predictions-我们模型预测的向量(红色:0/蓝色:1)
    '''
    A2,cache = forward_propagation(X,parameters)
    predictions = np.round(A2)
    return predictions

##正式运行
parameters = nn_model(X,Y,n_h=4,num_iterations=10000,print_cost=True)

#绘制边界
plot_decision_boundary(lambda  x:predict(parameters,x.T),X,Y)
plt.title('Decision Boundary for hidden layer size'+str(4))

predictions = predict(parameters,X)
print('准确率:%d'%float((np.dot(Y,predictions.T)+np.dot(1-Y,1-predictions.T))/float(Y.size)*100)+'%')
plt.show()