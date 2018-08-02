# Author:yu

#2.1.0TensorFlow模型
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mping
import tensorflow as tf
from tensorflow.python.framework import  ops

import cnn_utils

#%matplot inline
np.random.seed(1)

##2.1.1创建placeholders

def create_placeholders(n_H0,n_W0,n_C0,n_y):
    '''
    为session创建占位符
    :param n_H0: 实数,输入图像的高度
    :param n_W0: 实数,输入图像的宽度
    :param n_C0: 实数,输入图像的通道数
    :param n_y: 实数,分类数
    :return: 
    X-输入数据的占位符,维度为[None,n_H0,n_W0,n_C0],类型为'float'
    Y-输入数据的标签的占位符,维度为[None,n_y],类型为'float'
    '''
    X = tf.placeholder(tf.float32,[None,n_H0,n_W0,n_C0])
    Y = tf.placeholder(tf.float32,[None,n_y])

    return X,Y

# X , Y = create_placeholders(64,64,3,6)
# print ("X = " + str(X))
# print ("Y = " + str(Y))

##2.1.2初始化参数
def initialize_parameters():
    '''
    初始化权值矩阵,这里我们把权值矩阵硬编码
    W1:[4,4,3,8]
    W2:[2,2,8,16]
    :return: 
    包含了tensor类型的W1,W2的字典
    '''
    tf.set_random_seed(1)

    W1 = tf.get_variable('W1',[4,4,3,8],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W2',[2,2,8,16],initializer=tf.contrib.layers.xavier_initializer(seed = 0))

    parameters = {
        'W1':W1,
        'W2':W2
    }

    return parameters
# tf.reset_default_graph()
# with tf.Session() as sess_test:
#     parameters = initialize_parameters()
#     init = tf.global_variables_initializer()
#     sess_test.run(init)
#     print("W1 = " + str(parameters["W1"].eval()[1,1,1]))
#     print("W2 = " + str(parameters["W2"].eval()[1,1,1]))
#
#     sess_test.close()

#2.1.2前向传播

def forward_propagation(X,parameters):
    '''
    实现前向传播
    conv2d->relu->maxpool->conv2d->relu->maxpool->flatten->fullyconnected
    :param X: 输入数据的palceholder,维度为(输入节点数量,样本数量)
    :param parameters: 包含了'W1'和'W2'的python字典
    :return: 
    Z3-最后一个linear节点的输出
    
    '''
    W1 = parameters['W1']
    W2 = parameters['W2']

    #Conv2d :步伐:1,填充方式:'same
    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
    #relu:
    A1 = tf.nn.relu(Z1)
    #max pool:窗口大小:8*8,步伐:8*8,填充方式:'SAME
    P1 = tf.nn.max_pool(A1,ksize=[1,8,8,1],strides=[1,8,8,1],padding='SAME')

    #Conv2d:步伐:1,填充方式:SAME
    Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding='SAME')
    #relu:
    A2 = tf.nn.relu(Z2)
    #max pool:过滤器大小:4*4,步伐:4*4,填充方式:'SAME'
    P2 = tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')

    #一维化上一层的输出
    P = tf.contrib.layers.flatten(P2)

    #全连接层(FC),s使用没有非线性激活函数的全连接层
    Z3 = tf.contrib.layers.fully_connected(P,6,activation_fn=None)

    return Z3

# tf.reset_default_graph()
# np.random.seed(1)
#
# with tf.Session() as sess_test:
#     X,Y = create_placeholders(64,64,3,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#
#     init  = tf.global_variables_initializer()
#     sess_test.run(init)
#
#     a = sess_test.run(Z3,{X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
#     print('Z3 = '+str(a))
#
#     sess_test.close()

#2.1.3计算成本

def compute_cost(Z3,Y):
    '''
    计算成本
    :param Z3:正向传播最后一个linear节点的输出,维度为(6,样本数) 
    :param Y: 标签向量的placeholder,和Z3的维度相同
    :return: 
    cost-计算后的成本
    '''
    #tf.reduce_mean：计算的是平均值，使用它来计算所有样本的损失来得到总成本。
    #tf.nn.softmax_cross_entropy_with_logits(logits = Z3 , lables = Y)：计算softmax的损失函数。这个函数既计算softmax的激活，也计算其损失，
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))
    return cost

# tf.reset_default_graph()
# with tf.Session() as sess_test:
#     np.random.seed(1)
#     X,Y = create_placeholders(64,64,3,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#     cost = compute_cost(Z3,Y)
#
#     init = tf.global_variables_initializer()
#     sess_test.run(init)
#     a = sess_test.run(cost,{X:np.random.randn(4,64,64,3),Y:np.random.randn(4,6)})
#     print('cost = '+str(a))
#
#     sess_test.close()

##2.1.4构建模型
X_train_orig , Y_train_orig , X_test_orig , Y_test_orig , classes = cnn_utils.load_dataset()
# index = 6
# plt.imshow(X_train_orig[index])
# print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = cnn_utils.convert_to_one_hot(Y_train_orig, 6).T
Y_test = cnn_utils.convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}

def model(X_train,Y_train,X_test,Y_test,learning_rate = 0.009,
num_epochs = 100,minibatch_size = 64,print_cost = True,isPlot =True):
    '''
    使用TensorFlow实现三层的卷积神经网络
    conv2d->relu->maxpool->>conv2d->relu->maxpool->flatten->fullyconnnected
    :param X_train: 训练数据,维度为(None,64,64,3)
    :param Y_train: 训练数据对应的标签,维度为(Nne,n_y=6
    :param X_test: 测试数据,维度为(None,64,64,3)
    :param Y_test: 训练数据对应的标签,维度为(None,n_y=6)
    :param learning_rate: 学习率
    :param num_epochs: 遍历整个数据集的次数
    :param minibatch_size: 每个小批量数据块的大小
    :param print_cost: 是否打印成本值,每遍历100次整个数据集打印一次
    :param isPlot: 是否绘制图谱
    :return: 
    train_accuracy-实数,训练集的准确度
    test_accuracy-实数,测试集的准确度
    parameters-学习后的参数
    '''
    ops.reset_default_graph()#能后重新运行模型而不覆盖tf变量
    tf.set_random_seed(1)#确保你的数据和我的一样
    seed = 3#指定numpy的随机种子
    (m,n_H0,n_W0,n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    #为当前维度创建占位符
    X, Y = create_placeholders(n_H0,n_W0,n_C0,n_y)

    #初始化参数
    parameters = initialize_parameters()

    #前向传播
    Z3 = forward_propagation(X,parameters)

    #计算成本
    cost = compute_cost(Z3,Y)

    #反向传播,由于框架已经实现了反向传播,我们只需要选择一个优化器就行
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #全局初始化所有变量
    init = tf.global_variables_initializer()

    #开始运行
    with tf.Session() as sess:
        #初始化参数
        sess.run(init)
        #开始便利数据集
        for epoch in range(num_epochs):
            minibatch_cost = 0
            num_minibatches = int(m/minibatch_size)#获取数据块数量
            seed=seed+1
            minibatches = cnn_utils.random_mini_batches(X_train,Y_train,minibatch_size,seed)
            #对米格数据块进行处理
            for minibatch in minibatches:
                #选择一个数据块
                (minibatch_X,minibatch_Y) = minibatch
                #最小化这个数据块的成本
                _ ,temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})

                #累加数据块的成本值
                minibatch_cost += temp_cost/num_minibatches

            #是否打印成本
            if print_cost:
                #每5代打印一次
                if epoch%5==0:
                    print('当前是第'+str(epoch)+'代,成本值为:'+str(minibatch_cost))

            #记录成本
            if epoch%1 == 0:
                costs.append(minibatch_cost)
        #数据处理完毕,绘制成本曲线
        if isPlot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterstions(per tens)')
            plt.title('Learning rate ='+str(learning_rate))
            plt.show()

        #开始预测数据
        ##计算当前的预测情况
        predict_op = tf.arg_max(Z3,1)
        corrent_prediction = tf.equal(predict_op,tf.arg_max(Y,1))

        ##计算准确度
        accuracy = tf.reduce_mean(tf.cast(corrent_prediction,'float'))
        print('corrent_prediction accuary = '+str(accuracy))

        train_accuracy =accuracy.eval({X:X_train,Y:Y_train})
        test_accuracy = accuracy.eval(({X:X_test,Y:Y_test}))

        print('训练及准确度:'+str(train_accuracy))
        print('测试集准确度:'+str(test_accuracy))

        return(train_accuracy,test_accuracy,parameters)

_, _, parameters = model(X_train, Y_train, X_test, Y_test,num_epochs=150)