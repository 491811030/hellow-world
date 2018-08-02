# Author:yu
#使用tensorFlow构建神经网络
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import tf_utils
import time
X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes = tf_utils.load_dataset()
# index=11
# plt.imshow(X_train_orig[index])
# plt.show()
# print(np.squeeze(Y_train_orig))#从数组的形状中删除单维度条目，即把shape中为1的维度去掉

X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0],-1).T#每一列就是一个样本
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0],-1).T

#归一化数据
X_train = X_train_flatten/255
X_test = X_test_flatten/255

#转换为独热矩阵
Y_train = tf_utils.convert_to_one_hot(Y_train_orig,6)
Y_test = tf_utils.convert_to_one_hot(Y_test_orig,6)

# print('训练集样本数 = '+str(X_train.shape[1]))
# print('测试集样本数 = '+str(X_test.shape[1]))
# print('X_train.shape:'+str(X_train.shape))
# print('Y_train.shape:'+str(Y_train.shape))
# print('X_test.shape:'+str(X_test.shape))
# print('Y_test.shape:'+str(Y_test.shape))

#创建placeholders
def creat_placeholders(n_x,n_y):
    '''
    为TensorFlow创建占位符
    :param n_x: 一个实数,图片向量的大小(64*64*3 = 1228)
    :param n_y: 一个实数,分类数(从0到5,所以n_Y=6)
    :return: 
    x-一个数据输入的占位符,维度为[n_x,None],dtype = 'float'
    y-一个对应输入的标签的占位符,维度为[n_y,None],dtype = 'float'
    
    提示:
    使用None,因为它让我们可以灵活处理占位符提供的样本数量.事实上,测试/训练期间的样本数量是不同的。
    '''

    X =tf.placeholder(tf.float32,[n_x,None],name = 'X')
    Y = tf.placeholder(tf.float32,[n_y,None],name='Y')

    return X,Y

#2.2初始化参数
def initialize_parameters():
    '''
    初始化神经网络的参数,参数的维度如下:
    W1:[25,12288]
    b1:[25,1]
    W2:[12,25]
    b2:[12,1]
    W3:[6.12]
    b3:[6,1]
    :return: 
    parameters - 包含了W和b的字典
    '''

    tf.set_random_seed(1)#指定随机种子

    W1= tf.get_variable('W1',[25,12288],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable('b1',[25,1],initializer=tf.zeros_initializer())
    W2 = tf.get_variable('W2',[12,25],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable('b2',[12,1],initializer=tf.zeros_initializer())
    W3 = tf.get_variable('W3',[6,12],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable('b3',[6,1],initializer=tf.zeros_initializer())

    parameters = {
        'W1':W1,
        'b1':b1,
        'W2':W2,
        'b2':b2,
        'W3':W3,
        'b3':b3
    }
    return parameters

tf.reset_default_graph() #用于清除默认图形堆栈并重置全局默认图形。

# with tf.Session() as sess:
#     parameters = initialize_parameters()
#     print("W1 = " + str(parameters["W1"]))
#     print("b1 = " + str(parameters["b1"]))
#     print("W2 = " + str(parameters["W2"]))
#     print("b2 = " + str(parameters["b2"]))

##2.3前向传播
def forward_propagation(X,parameters):
    '''
    实现一个模型的前向传播,模型结构是linera->rel->linear->relu->linear->softmax
    :param X: 输入数据的占位符,维度为(输入节点数量,样本数量)
    :param parameters: 包含了W和b的参数的输出
    :return: 
    Z3-最后一个linear节点的输出
    '''
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1,X),b1)  #Z1 = np.dot(W1,X)+b1
    #Z1 = tf.matmul(W1,X)+b1 #也可以这样写
    A1 = tf.nn.relu(Z1)  #A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2) #Z2 = np.dot(W2,a1)+b2
    A2 = tf.nn.relu(Z2) #A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)#Z3 = np.dot(W3,Z2)+b3

    return Z3

# tf.reset_default_graph()#用于清除默认图形堆栈并重置全局默认图形。
# with tf.Session() as sess:
#     X,Y = creat_placeholders(12288,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#     print('Z3 = '+str(Z3))

def compute_cost(Z3,Y):
    '''
    计算成本
    :param Z3:前向传播的结果 
    :param Y: 标签,一个占位符,和Z3的维度相同
    :return: 
    cost-成本值
    
    '''
    logits=tf.transpose(Z3)#转置
    labels = tf.transpose(Y)#转置

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    #计算出平均交叉熵损失


    return cost

# tf.reset_default_graph()
# with tf.Session() as sess:
#     X,Y = creat_placeholders(12288,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#     cost = compute_cost(Z3,Y)
#     print('cost = '+str(cost))


##2.6构建模型
def model(X_train,Y_train,X_test,Y_test,
          learning_rate = 0.0001,num_epochs = 1500,minibatch_size = 32,
          print_cost = True,is_plot = True):
    '''
    实现一个三层的TensorFlow神经网络:linear->relu->linear->relu->linear->softmax
    :param X_train: 训练集,维度为(输入大小(输入节点数量) = 12288,样本数量=1080)
    :param Y_train: 训练集分类数量,维度为(输出大小(输出节点数量)=6,样本数量=1080)
    :param X_test: 测试集,维度为(输入大小(输入节点数量)=12288,,样本数量= 120)
    :param Y_test: 测试集分类数量,维度为(输出大小(输出节点数量) = 6,样本数量= 120)
    :param learning_rate: 学习速率
    :param num_epochs: 整个训练集的遍历次数
    :param minibatch_size: 每个小批量数据集的大小
    :param print_cost: 是否打印成本,每100代打印一次
    :param is_plot: 是否绘制曲线
    :return: 
    parameters - 学习后的参数
    '''
    ops.reset_default_graph() #能够重新运行模型而不覆盖tf变量
    tf.set_random_seed(1)
    seed = 3
    (n_x,m) = X_train.shape#获取输入节点数量和样本数
    n_y = Y_train.shape[0]#获取输出节点数量
    costs = []#成本集

    #给X和Y创建placeholder
    X,Y=creat_placeholders(n_x,n_y)
    #初始化参数
    parameters = initialize_parameters()

    #前向传播
    Z3 = forward_propagation(X,parameters)

    #计算成本
    cost = compute_cost(Z3,Y)

    #反向传播,使用Adam优化
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #初始化所有变量
    init = tf.global_variables_initializer()

    #开始会话并计算
    with tf.Session() as sess:
        #初始化
        sess.run(init)

        #正常训练的循环
        for epoch in range(num_epochs):
            epoch_cost = 0#每代的成本
            num_minibatches = int(m/minibatch_size)#minibatch的总数量
            seed =seed +1
            minibatches = tf_utils.random_mini_batches(X_train,Y_train,minibatch_size,seed)

            for minibatch in minibatches:

                #选择一个minibatch
                (minibatch_X,minibatch_Y) = minibatch

                #数据已经准备好,开始运行session
                _,minibatch_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})

                #计算这个minibatch在这一代中所占的误差
                epoch__cost = epoch_cost +minibatch_cost/num_minibatches

            #记录并打印成本
            ##记录成本
            if epoch%5==0:
                costs.append(epoch__cost)
                #是否打印:
                if print_cost and epoch%100 == 0:
                    print('epoch = '+str(epoch)+'epoch_cost'+str(epoch__cost))

        #是否绘制图谱
        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations(per ten)')
            plt.title('learning rate ='+str(learning_rate))
            plt.show()

        #保存学习后的参数
        parameters = sess.run(parameters)
        print('参数已保存到session')

        #计算当前的预测结果
        correct_prediction = tf.equal(tf.argmax(Z3),tf.argmax(Y))

        #计算准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

        print('训练集的准确率:',accuracy.eval({X:X_train,Y:Y_train}))
        print('测试集的准确率:',accuracy.eval({X:X_test,Y:Y_test}))

        return parameters


#开始时间
start_time = time.clock()
#开始训练
parameters = model(X_train, Y_train, X_test, Y_test)
#结束时间
end_time = time.clock()
#计算时差
print("CPU的执行时间 = " + str(end_time - start_time) + " 秒" )