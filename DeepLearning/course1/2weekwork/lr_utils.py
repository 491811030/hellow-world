# Author:yu
import numpy as np
import h5py
import matplotlib.pyplot as plt

def load_dataset():
    train_dataset = h5py.File('D:/DeepLearning/【吴恩达课后编程作业】第二周 - PA1 - 具有神经网络思维的Logistic回归/datasets/train_catvnoncat.h5','r')
    train_set_x_orig = np.array(train_dataset['train_set_x'][:])
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File('D:/DeepLearning/【吴恩达课后编程作业】第二周 - PA1 - 具有神经网络思维的Logistic回归/datasets/test_catvnoncat.h5','r')
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])

    classes = np.array(test_dataset['list_classes'][:])

    train_set_y_orig = train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))

    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes

train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = load_dataset()
# index = 25
#
# plt.imshow(train_set_x_orig[index])
# plt.show()

m_train=train_set_y.shape[1]#训练集里图片的数量
m_test = test_set_y.shape[1]#测试集里图片的数量
num_px = train_set_x_orig.shape[1]#训练,测试集里面图片的宽度和高度

# print('训练集的数量:m_train = '+str(m_train))
# print('测试集的数量:m_test = '+str(m_test))
# print('每张图片的宽/高:num_px = '+str(num_px))
# print('每张图片的大小:('+str(num_px)+','+str(num_px)+',3')
# print('训练集_图片的维数:'+str(train_set_x_orig.shape))
# print('训练集_标签的维数'+str(train_set_y.shape))
# print('测试集_图片的维数:'+str(test_set_x_orig.shape))
# print('测试集_标签的维数:'+str(test_set_y.shape))

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig[0],-1).T
#-1的意思是以前面的行为准,剩下的列电脑自己算

#数据标准化
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

def sigmoid(z):
    '''
    :param z: 任何大小的标量或numpy数组.
    :return: s  - sigmoid(z)
    '''
    s = 1/(1+np.exp(-z))
    return s

def initialize_with_zeros(dim):
    '''
    此函数为w创建一个维度为(dim,1)的θ向量,并将b初始化为θ
    :param dim: 我们想要的w矢量的大小(或者这种情况系的参数数量)
    :return: w-维度为(dim,1)的初始化向量
    b-初始化的标量(对应于偏差)
    '''
    w = np.zeros(shape=(dim,1))
    b = 0
    #使用断言确保我要的数据是正确的
    assert(w.shape == (dim,1))#w的维度是(dim,1)
    assert(isinstance(b,float) or isinstance(b,int))#b的类型是float或int
    return(w,b)

#计算成本函数及其渐变的函数propagate()
def propagate(w,b,X,Y):
    '''
    实现前向和后向的成本函数及其递归
    :param w: 权重,大小不等的数组(num_px*num_px*3,1)
    :param b: 偏差,一个标量
    :param X: 矩阵类型为(num_px*num_px*3,训练数量)
    :param Y: 真正的'标签'矢量(如果非猫则为0,如果是猫则为1),矩阵维度为(1,训练数量)
    :return: cost-逻辑回归的负对数似然成本
    dw-相对于w的损失梯度,因此与w相同的形状
    db- 相对于b的巡视梯度,因此与b的形状相同
    '''
    m = X.shape[1]

    #正向传播
    A= sigmoid(np.dot(w.T,X)+b)#计算激活函数
    cost = (-1/m)*np.sum(Y*np.log(A)+(1-Y)*(np.log(1-A)))#计算成本

    #反向传播
    dw = (1/m)*np.dot(X,(A-Y).T)
    db = (1/m)*np.sum(A-Y)

    #使用断言确保我的数据是正确的
    assert(dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.square(cost)
    assert (cost.shape == ())

    #创建一个字典,把dw和db保存起来
    grads = {
        'dw':dw,
        'db':db,

    }
    return(grads,cost)

#优化w和b
def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost = False):
    '''
    此函数通过运行梯度下降算法来优化w和b
    :param w: 权重,大小不等的数组(num_px*num_px*3,1)
    :param b: 偏差,一个标量
    :param X: 维度为(num_px*num_px*3,训练数据的数量)的数组
    :param Y: 真正的'标签'矢量(如果非猫则为0,如果是猫则为1),矩阵维度为(1,训练数据的数量)
    :param num_iterations: 优化循环的迭代次数
    :param learning_rate: 梯度下降更新规则的学习率
    :param print_cost: 每100补打印一次损失值
    :return: params-包括权重w和偏差b的字典
    grads-包含权重和偏差相对于成本函数的梯度的字典
    成本-优化期间计算的所有成本列表,将用于绘制学习曲线
    提示:
    我们需要写下两个步骤并遍历它们:
    1>计算当前参数的成本和梯度,使用propagate().
    2>使用w和b的梯度下降法更新参数
    '''
    costs = []
    for i in range(num_iterations):
        grads,cost = propagate(w,b,X,Y)
        dw = grads['dw']
        db = grads['db']
        w = w-learning_rate*dw
        b = b-learning_rate*db

        #记录成本
        if i%100 == 0:
            costs.append(cost)
        #打印是成本数据
        if (print_cost) and (i%100 == 0):
            print('迭代的次数:%i,误差值:%f'%(i,cost))
    params =  {
        'w' : w,
        'b':b
    }
    grads={
        'dw':dw,
        'db':db
    }
    return(params,grads,costs)

#实现预测函数
def predict(w,b,X):
    '''
    使用学习逻辑回归参数logistic(w,b)预测的标签是0还是1,
    :param w: 权重,大小不等的数组(num_px*num_ox*3,1
    :param b: 偏差,一个标量
    :param x: 维度为(num_px*num_px*3,训练数据的数量)的数据
    :return: 
    Y_prediction - 包含X中所有预测[0|1]的一个numpy数组
    '''

    m= X.shape[1]#图片的数量
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    #计预测猫在图片中出现的概率
    A=sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        #将概率a [0,i]转换为实际预测p[0,i]
        Y_prediction[0,i] = 1 if A[0,i]>0.5 else 0
    #使用断言
    assert(Y_prediction.shape == (1,m))
    return Y_prediction
