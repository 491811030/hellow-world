# Author:yu
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import tf_utils
import time

#%matplotlib inline #如果你使用的是jupyter notebook取消注释
np.random.seed(1)

y_hat = tf.constant(36,name='y_hat')#定义y_hat为固定值36
y = tf.constant(39,name='y')#定义y为固定值39

loss = tf.Variable((y-y_hat)**2,name='loss')#为损失函数创建一个变量

init = tf.global_variables_initializer()#运行之后的初始化session.run(init),损失变量将被初始化并准备计算

# with tf.Session() as session:#创建一个session并打印输出
#     session.run(init)#初始化变量
#     print(session.run(loss))#打印损失值

a = tf.constant(2)
# b = tf.constant(10)
# c = tf.multiply(a,b)
#
# print(c)

# sess = tf.Session()
# print(sess.run(c))
#记得初始化变量，然
# 后创建一个session来运行它。

#占位符（placeholders）,占位符是一个对象，它的值只能在稍后指定，要指定占位符的值，可以使用一个feed字典（feed_dict变量）来传入，
#利用feed_dict来改变x的值
# x= tf.placeholder(tf.int64,name='x')
# print(sess.run(2*x,feed_dict = {x:3}))
# sess.close()



##线性函数
#我们通过计算以下等式来开始编程：Y=WX+b
def linear_function():
    '''
    实现一个线性函数功能:
    初始化W,类型为tensor的随机变量,维度为(4,3)
    初始化X,类型为tensor的随机变量,维度为(3,1)
    初始化b,类型为tensor的随机变量,维度为(4,1)
    返回:
    result-运行了session后的结果,运行的是Y=WX+b 
    '''
    np.random.seed(1)#指定随机种子

    X=np.random.randn(3,1)
    W=np.random.randn(4,3)
    b = np.random.randn(4,1)

    Y = tf.add(tf.matmul(W,X),b)#tf.matmul是矩阵乘法
    #Y = tf.matmul(W,X)+b#也可以写成这样子

    #创建一个session并运行它
    sess = tf.Session()
    result = sess.run(Y)

    #session使用完毕,关闭它
    sess.close()

    return result

# print("result = " +  str(linear_function()))


##计算sigmoid
def sigmoid(z):
    '''
    实现使用sigmoid函数计算z
    :param z: 输入的值,标量或矢量
    :return: 
    result-用sigmoid计算z的值
    '''

    #创建一个占位符x,名字叫'x'
    x = tf.placeholder(tf.float32,name='x')

    #计算sigmoid(z)
    sigmoid = tf.sigmoid(x)

    #创建一个会话:
    with tf.Session() as sess:
        result = sess.run(sigmoid,feed_dict={x:z})
    return result

# print ("sigmoid(0) = " + str(sigmoid([0,1,2])))
# print ("sigmoid(12) = " + str(sigmoid(12)))

##计算成本
#实现成本函数，需要用到的是：
#tf.nn.sigmoid_cross_entropy_with_logits(logits = ..., labels = ...)

##独热编码,y中共有c个类,把y变成有c行的矩阵,每一列只有之前类别-1的行为1

def one_hot_matrix(lables,C):
    '''
    创建一个矩阵,其中第i行对应第i个类号,第j列对应第j个训练样本
    所以如果第j个样本对应着第i个标签,那么entry(i,j)将会是1
    :param label: 标签向量
    :param C: 分类数
    :return: 
    one_hot-独热矩阵
    '''

    #创建一个tf.constant,赋值为C,名字为C
    C = tf.constant(C,name='C')

    #使用tf.one_hot,注意一下axis
    one_hot_martrix = tf.one_hot(indices = lables,depth = C,axis = 0)
    #axis=0表示对列操作
    #创建一个session
    sess = tf.Session()

    #运行session
    one_hot = sess.run(one_hot_martrix)

    #关闭session
    sess.close()
    return one_hot
# labels = np.array([1,2,3,0,2,1])
# one_hot = one_hot_matrix(labels,C = 4)
# print(str(one_hot))

#1.5初始化为0和1
def ones(shape):
    '''
    创建一个维度为shape的变量,其值全为1
    :param shape: 你要创建的数组的维度
    :return: 
    ones - 只包含1的数组
    '''
    #使用tf.ones()
    ones = tf.ones(shape)

    #创建会话
    sess = tf.Session()

    #运行回话
    ones = sess.run(ones)

    #关闭会话
    sess.close()
    return ones
print ("ones = " + str(ones([3])))