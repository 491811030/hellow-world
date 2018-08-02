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
train_X,train_Y,test_X,test_Y = reg_utils.load_2D_dataset(is_plot = False)
# plt.show()
#对比不使用正则化,使用L2正则化,使用随机节点删除

def model(X,Y,learning_rate = 0.3,num_iterations = 30000,print_cost = True,is_plot = True,lambd = 0,keep_prob = 1):
    '''
    实现一个三层的神经网络:linear->relu->linear->relu->linear->sigmoid
    :param X: 输入的数据,维度为(2,要训练/测试的数量)
    :param Y: 标签,[0(蓝色)|1(红色)],维度为(1,对应的是输入的数据的标签)
    :param learing_rate: 学习速率
    :param num_iterations: 迭代次数
    :param print_cost: 是否打印成本值
    :param lambd: 正则化的超参数,实数
    :param keep_prob: 随机删除节点的概率
    :return: 
    parameters-学习后的参数
    '''
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0],20,3,1]

    #初始化参数
    parameters = reg_utils.initialize_parameters(layers_dims)

    #开始学习
    for i in range(0,num_iterations):
        #前向传播
        ##是否删除随机节点
        if keep_prob == 1:
            ###不删除随机节点
            a3,cache = reg_utils.forward_propagation(X,parameters)
        elif keep_prob<1:
            ###随即删除节点
            a3 , cache = forward_propagation_with_dropout(X,parameters,keep_prob)
        else:
            print('keep_prob参数错误,程序退出.')
            exit

        #计算成本
        ##是否使用二范数
        if lambd == 0:
            ###不使用L2正则化
            cost = reg_utils.compute_cost(a3,Y)
        else:
            ###使用L2正则化
            cost = compute_cost_with_regularization(a3,Y,parameters,lambd)

        ##反向传播
        ##可以同时使用L2正则化和随机删除节点,但本次实验不同时使用
        assert(lambd == 0 or keep_prob == 1)

        ##两个参数的使用情况
        if(lambd == 0 and keep_prob == 1):
            ###不适用L2正则化和不使用随即删除节点
            grads = reg_utils.backward_propagation(X,Y,cache)
        elif lambd!=0:
            ###使用L2正则化,不使用随即删除节点
            grads = backward_propagation_with_regularization(X,Y,cache,lambd)
        elif keep_prob<1:
            ###使用随即删除节点,不使用L2正则化
            grads = backward_propagation_with_dropout(X,Y,cache,keep_prob)

        #更新参数
        parameters = reg_utils.update_parameters(parameters,grads,learning_rate)

        #记录并打印成本
        if i%1000 == 0:
            ##记录成本
            costs.append(cost)
            if(print_cost and i%10000==0):
                #打印成本
                print('第'+str(i)+'次迭代,成本值为:'+str(cost))

    #是否绘制成本曲线图
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations(x1,000)')
        plt.title('Learning rate = '+str(learning_rate))
        plt.show()

    #返回学习后的参数
    return parameters


###不是用正则化,输出损失曲线
# parameters = model(train_X,train_Y,is_plot=True)
# print('训练集')
# predictions_train = reg_utils.predict(train_X,train_Y,parameters)
# print('测试集')
# predictions_test = reg_utils.predict(test_X,test_Y,parameters)

#画出边界,在无正则化时,分割曲线存在过拟合现象
# plt.title('model without regularization')
# axes = plt.gca()
# axes.set_xlim([-0.75,0.40])
# axes.set_ylim([-0.75,0.65])
# reg_utils.plot_decision_boundary(lambda x:reg_utils.predict_dec(parameters,x.T),train_X,train_Y)

#使用正则化
#L2正则化
def compute_cost_with_regularization(A3,Y,parameters,lambd):
    '''
    实现L2正则化计算成本
    :param A3: 正向传播的结果,维度为(输出节点数量,训练/测试的数量)
    :param Y: 标签向量,与数据一一对应,维度为(输出节点数量,驯良/测试的数量)
    :param parameters: 包含模型学习后的参数的字典
    :param lambd: 
    :return: 
    cost - 使用公式2计算出来的正则化损失函值
    '''

    m=Y.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    cross_entropy_cost = reg_utils.compute_cost(A3,Y)

    L2_regularization_cost = lambd*(np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3)))/(m*2)
    cost = cross_entropy_cost + L2_regularization_cost

    return cost

#当然,因为改变了成本函数,我们也必须改变后向传播函数,所有的梯度都必须根据这个新的成本值来计算
def backward_propagation_with_regularization(X,Y,cache,lambd):
    '''
    实现我们添加了L2正则化的模型的后向传播
    :param X: 输入数据集,维度为(输入节点数量,数据集里面的数量
    :param Y: 标签,维度为(输出节点数量,数据集里面的数量
    :param cache: 来自forward_propagation()的cache输出
    :param lambd: regularization超参数,实数
    :return: 
    gradients - 一个包含了每个参数\激活值和预激活值变量的替独子点
    '''
    m = X.shape[1]
    (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3)=cache
    dZ3 = A3-Y

    dW3 = (1/m)*np.dot(dZ3,A2.T)+((lambd*W3)/m)
    db3 = (1/m)*np.sum(dZ3,axis = 1,keepdims=True)

    dA2 = np.dot(W3.T,dZ3)
    dZ2 = np.multiply(dA2,np.int64(A2>0))
    dW2 = (1/m)*np.dot(dZ2,A1.T)+((lambd*W2)/m)
    db2 = (1/m)*np.sum(dZ2,axis = 1,keepdims=True)

    dA1 = np.dot(W2.T,dZ2)
    dZ1 = np.multiply(dA1,np.int64(A1>0))
    dW1 = (1/m)*np.dot(dZ1,X.T)+((lambd*W1)/m)
    db1 = (1/m)*np.sum(dZ1,axis =1,keepdims = True)

    gradients = {
        'dZ3':dZ3,'dW3':dW3,'db3':db3,'dA2':dA2,
        'dZ2':dZ2,'dW2':dW2,'db2':db2,'dA1':dA1,
        'dZ1':dZ1,'dW1':dW1,'db1':db1
    }

    return gradients

# parameters = model(train_X, train_Y, lambd=0.7,is_plot=True)
# print("使用正则化，训练集:")
# predictions_train = reg_utils.predict(train_X, train_Y, parameters)
# print("使用正则化，测试集:")
# predictions_test = reg_utils.predict(test_X, test_Y, parameters)
# plt.title("Model with L2-regularization")
#
#
# axes = plt.gca()
# axes.set_xlim([-0.75,0.40])
# axes.set_ylim([-0.75,0.65])
# reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)



##随机删除节点dropout

def forward_propagation_with_dropout(X,parameters,keep_prob=0.5):
    '''
    实现具有随机舍弃节点的前向传播
    linear->relu+dropout->linear->relu->dropout->linear->sigmoid
    :param X: 输入数据集,维度为(2,示例数)1
    :param parameters: 包含参数'W1','b1','W2','b2','W3','b3'的Python字典
    W1 - 权重矩阵,维度为(20,2)
    b1 - 偏向量,维度为(20,1)
    W2 - 权重矩阵,维度为(3,20)
    b2 - 偏向量,维度为(3,1)
    W3 - 权重矩阵,维度为(1,3)
    b3 - 偏向量,维度为(1,1)
    keep_prob - 随即删除的概率,实数
    
    :param keep_prob: 
    :return:
     A3 - 最后的激活值,维度为(1,1),正向传播的输出
     cache-存储了一些用于反向传播的元组
    '''
    np.random.seed(1)

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    #linear->relu->linear->relu->linear->sigmoid
    Z1 = np.dot(W1,X)+b1
    A1 = reg_utils.relu(Z1)

    #下面的步骤1-4对应于上述的步骤1-4
    D1=np.random.rand(A1.shape[0],A1.shape[1])#步骤1:初始化D1 = np.random.rand(...,...)
    D1 = D1<keep_prob                         #步骤2:将D1的值转化为0或1(使用keep_prob作为阈值)
    A1 = A1*D1                                #步骤3:舍弃A1的一些节点(将它的值变为0或False)
    A1 = A1/keep_prob                         #步骤4:缩放未舍弃的节点(不为0)的值
    '''
    不理解的同学运行一下下面的代码就知道了
    import numpy as np
    np.random.seed(1)
    A1 = np.random.randn(1,3)
    
    D1 = np.random.rand(A1.shape[0],A1.shape[1])
    keep_prob = 0.5
    D1 = D1 < keep_prob
    print(D1)
    
    A1 = 0.01
    A1 = A1*D1
    A1 = A1 /keep_prob
    print(A1)
    '''

    Z2 = np.dot(W2,A1)+b2
    A2 = reg_utils.relu(Z2)

    #下面的步骤1-4对应上述的步骤1-4
    D2 = np.random.rand(A2.shape[0],A2.shape[1])  #步骤1:初始化矩阵D2 = nprandom.rand(...,...)
    D2 = D2<keep_prob                             #步骤2:将D2的值转换为0或1(使用keep_prob作为阈值)
    A2 = A2*D2                                    #步骤3:舍弃A2的一些节点(将它的值变为0或False)
    A2 = A2/keep_prob                             #步骤4:缩放未舍弃的节点(不为0)的值

    Z3 = np.dot(W3,A2)+b3
    A3 = reg_utils.sigmoid(Z3)

    cache = (Z1,D1,A1,W1,b1,Z2,D2,A2,W2,b2,Z3,A3,W3,b3)

    return A3,cache

##改变后向传播算法
def backward_propagation_with_dropout(X,Y,cache,keep_prob):
    '''
    实现我们随即删除的模型的后向模型
    :param X: 输入的数据集,维度为(2,示例数)
    :param Y: 标签,维度为(输出的节点数量,示例数量)
    :param cache: 来自forward_propagation_with_dropout()的cache输出
    :param keep_prob: 随即删除的概率,实数
    :return: 
    gradients - 一个关于每个参数\激活值和预激活变量的梯度值的字典
    
    '''
    m = X.shape[1]
    (Z1,D1,A1,W1,b1,Z2,D2,A2,W2,b2,Z3,A3,W3,b3) = cache

    dZ3 = A3 - Y
    dW3 = (1/m)*np.dot(dZ3,A2.T)
    db3 = 1./m * np.sum(dZ3,axis = 1,keepdims = True)
    dA2 = np.dot(W3.T,dZ3)

    dA2 = dA2 *D2 #步骤1:使用正向传播期间相同的节点,舍弃那些关闭的节点
    dA2 = dA2/keep_prob#步骤2:缩放未舍弃的节点(不为0)的值

    dZ2 = np.multiply(dA2,np.int64(A2>0))
    dW2 = 1./m*np.dot(dZ2,A1.T)
    db2 = 1./m*np.sum(dZ2,axis = 1,keepdims = True)

    dA1 = np.dot(W2.T,dZ2)

    dA1 = dA1*D1 #步骤1:使用正向传播期间相同的节点,舍弃那些关闭的节点
    dA1 = dA1/keep_prob#步骤2:缩放未舍弃的节点(不为0)的值

    dZ1 = np.multiply(dA1,np.int64(A1>0))
    dW1 = 1/m*np.dot(dZ1,X.T)
    db1 = 1./m*np.sum(dZ1,axis = 1,keepdims=True)

    gradients = {
        'dZ3':dZ3,'dW3':dW3,'db3':db3,'dA2':dA2,
        'dZ2':dZ2,'dW2':dW2,'db2':db2,'dA1':dA1,
        'dZ1':dZ1,'dW1':dW1,'db1':db1
    }
    return gradients

parameters = model(train_X, train_Y, keep_prob=0.86, learning_rate=0.3,is_plot=True)

print("使用随机删除节点，训练集:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("使用随机删除节点，测试集:")
reg_utils.predictions_test = reg_utils.predict(test_X, test_Y, parameters)

plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)