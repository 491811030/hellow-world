# Author:yu

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils   #第一部分，初始化
import reg_utils    #第二部分，正则化
import gc_utils     #第三部分，梯度校验
#%matplotlib inline #如果你使用的是Jupyter Notebook，请取消注释。
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#一维线性

#前向传播
def forward_propagation(x,theta):
    '''
    实现图中呈现的线性前向传播(计算J)(J(theta) = theta*x)
    :param x: 一个实数输入
    :param theta: 参数,也是一个实数
    :return: 
    J-函数J的值,用公式J(theta) = theta*x
    '''
    J= np.dot(theta,x)
    return J

#反向传播
def backward_propagation(x,theta):
    '''
    计算J相对于θ的导数
    :param x: 一个实数输入
    :param theta: 参数,也是一个实数
    :return: 
    dtheta - 相对于θ的成本梯度
    '''
    dtheta = x
    return dtheta

#梯度检查
def gradient_check(x , theta,epsilon = 1e-7):
    '''
    实现图中的反向传播
    :param x: 一个实值输入
    :param theta: 参数,也是一个实数
    :param epsilon: 使用公式(3)计算输入的微小偏移以计算近似梯度
    :return: 
    近似梯度和后向传播梯度之间的差异
    '''
    ##计算公式(3)的左侧gradapprox
    thetaplus = theta +epsilon
    thetaminus = theta - epsilon
    J_plus = forward_propagation(x,thetaplus)
    J_minus = forward_propagation(x,thetaminus)
    gradapprox = (J_plus-J_minus)/(2*epsilon)

    #检查gradapprox是否做够接近backward_propagation()的输出
    grad = backward_propagation(x,theta)

    numerator = np.linalg.norm(grad-gradapprox)##求范数,默认为2范数
    denominator = np.linalg.norm(grad)+np.linalg.norm(gradapprox)
    difference = numerator / denominator##归一化处理


    if difference<1e-7:
        print('梯度检查:梯度正常')
    else:
        print('梯度检查:梯度超出阈值')
    return difference


###高维参数时
def forward_propagation_n(X,Y,parameters):
    '''
    实现图中的前向传播(并计算成本)
    :param X: 训练集为m个例子
    :param Y: m个示例标签
    :param parameters: 包含参数'W1','b1','W2','b2','W3','b3'的python字典 
            W1  - 权重矩阵，维度为（5,4）
            b1  - 偏向量，维度为（5,1）
            W2  - 权重矩阵，维度为（3,5）
            b2  - 偏向量，维度为（3,1）
            W3  - 权重矩阵，维度为（1,3）
            b3  - 偏向量，维度为（1,1）
    :return:
    cost - 成本函数(logistic)
    '''
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    #linear->relu->linear->relu->linear->sigmoid
    Z1 = np.dot(W1,X)+b1
    A1 = gc_utils.relu(Z1)

    Z2 = np.dot(W2,A1)+b2
    A2 = gc_utils.relu(Z2)

    Z3 = np.dot(W3,A2)+b3
    A3 = gc_utils.sigmoid(Z3)

    #计算成本
    logprobs = np.multiply(-np.log(A3),Y)+np.multiply(-np.log(1-A3),1-Y)
    cost = (1/m)*np.sum(logprobs)

    cache = (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3)
    return cost,cache

def backward_propagation_n(X,Y,cache):
    '''
    实现图中所示的反向传播
    :param X: 输入数据点(输入节点数量,1)
    :param Y: 标签
    :param cache: 来自forward_propagation_n()的cache输出
    :return: 
    gradients - 一个字典,其中包含与每个参数\激活和激活前变量相关的成本梯度
    '''
    m = X.shape[1]
    (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3,)= cache
    dZ3 = A3-Y
    dW3 = (1/m)*np.dot(dZ3,A2.T)
    dW3 = 1./m *np.dot(dZ3,A2.T)
    db3 = 1./m*np.sum(dZ3,axis=1,keepdims=True)

    dA2 = np.dot(W3.T,dZ3)
    dZ2 = np.multiply(dA2,np.int64(A2>0))#乘以一个零一矩阵
    dW2 = 1./m*np.dot(dZ2,A1.T)
    db2 = 1./m*np.sum(dZ2,axis = 1,keepdims=True)

    dA1 = np.dot(W2.T,dZ2)
    dZ1 = np.multiply(dA1,np.int64(A1>0))
    dW1 = 1./m*np.dot(dZ1,X.T)
    #db1 = 4./m*sum(dZ1,axis = 1,keepdims=True)#should not mutiply 4
    db1 = 1./m*np.sum(dZ1,axis=1,keepdims = True)

    gradients = {'dZ3':dZ3,'dW3':dW3,'db3':db3,
                'dA2':dA2,'dZ2':dZ2,'dW2':dW2,'db2':db2
                'dA1':dA1,'dZ1':dZ1,'dW1':dW1,'db1':db1}

    return gradients



def gradient_check_n(parameters,gradients,X,Y,epsilon = 1e-7):
    '''
    检查backward_propagation_n是否正确计算forward_propagation_n输出的成本梯度
    :param parameters: 包含参数'W1','b1','W2','b2','W3','b3'的python字典
    grad_output_propagation_n的输出包含与参数相关的成本梯度
    :param gradients: 
    :param X: 输入数据点,维度为(输入节点数量,1)
    :param Y: 标签
    :param epsilon计算输入的微小偏移以计算近似梯度: 
    :return: 
    近似梯度和后向传播梯度之间的差异
    '''
    #初始化参数
    parameters_values , keys = gc_utils.dictionary_to_vector(parameters)#keys用不到,parameters_values是一个n行1列的矩阵
    grad = gc_utils.gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters,1))
    J_minus = np.zeros((num_parameters,1))
    gradapprox = np.zeros(num_parameters,1)

    #计算gradapprox
    for i in range(num_parameters):
        #计算J_plus[i].输入'parameters_values,epsilon'.输出='J_plus[i]'
        thetaplus = np.copy(parameters_values)
        thetaplus[i][0]=thetaplus[i][0]+epsilon
        J_plus[i],cache = forward_propagation_n(X,Y,gc_utils.vector_to_dictionary(thetaplus))

        #计算J_minus[i].输入:'parameters_values,epsilon',输出='J_minus[i]'
        thetaminus = np.copy(parameters_values)
        thetaminus[i][0]=thetaminus[i][0] - epsilon
        J_minus[i],cache = forward_propagation_n(X,Y,gc_utils.vector_to_dictionary(thetaminus))

        #计算gradapprox[i]
        gradapprox[i] = (J_plus[i]-J_minus[i])/(2*epsilon)

    #通过计算差异比较gradapprox和后向传播梯度
    numerator = np.linalg.norm(grad-gradapprox)
    denominator = np.linalg.norm(grad)+np.linalg.norm(gradapprox)
    difference = numerator/denominator

    if difference<1e-7:
        print('梯度检查:梯度正常')
    else:
        print('梯度检查:梯度超出阈值')

    return difference