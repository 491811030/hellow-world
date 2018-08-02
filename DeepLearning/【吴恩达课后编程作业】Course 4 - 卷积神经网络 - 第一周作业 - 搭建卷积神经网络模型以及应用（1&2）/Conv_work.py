# Author:yu
import numpy as np
import h5py
import matplotlib.pyplot as plt

#%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0,4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#ipython很好用，但是如果在ipython里已经import过的模块修改后需要重新reload就需要这样
#在执行用户代码前，重新装入软件的扩展和模块。
#%load_ext autoreload
#autoreload 2：装入所有 %aimport 不包含的模块。
#%autoreload 2

np.random.seed(1)      #指定随机种子


##1.3.1边界填充
#constant连续一样的值填充,有constant_values = (x,y)时前面用x填充,后面用y填充,缺省参数是为constant_values=(0,0)

#a = np.pad(a,((0,0),(1,1),(0,0),(3,3),(0,0)),'constant',constant_values=(..,..))
#
# #例子
# import numpy as np
# arr3D = np.array([[[1,1,2,2,3,4],
#                    [1,1,2,2,3,4],
#                    [1,1,2,2,3,4]],
#                   [[0,1,2,3,4,5],
#                    [0,1,2,3,4,5],
#                    [0,1,2,3,4,5]],
#                   [[1,1,2,2,3,4],
#                    [1,1,2,2,3,4],
#                    [1,1,2,2,3,4]]])
# print('constant:\n'+str(np.pad(arr3D,((0,0),(1,1),(2,2)),'constant')))
# # 第一个参数是待填充数组
# # 第二个参数是填充的形状，（2，3）表示前面两个，后面三个.在多维上表示相对于最高维(0,0)的填充形状,(1,1)表示下一个维度上的形状,
# #(2,2)表示在最后一个维度下的填充形状
# # 第三个参数是填充的方法

def zero_pad(X,pad):
    '''
    把数据集X的图像边界全部使用0来扩充pad个宽度和高度
    :param X: 图像数据集,维度为(样本数,图像高度,图像宽度,图像通道数)
    :param pad: 整数,每个图像在垂直和水平维度上的填充量
    :return: 
    X_paded-扩充后的图像数据集,维度为(样本数,图像高度+2*pad,图像宽度+2*pad,图像通道数)
    '''
    X_paded = np.pad(X,(
        (0,0),#样本数,不填充
        (pad,pad),#图像高度,你可以视为上面填充x个,下面填充y个(x,y)
        (pad, pad),#图像宽度,你可以视为上面填充x个,下面填充y个(x,y)
        (0,0)),#通道数,不填充
        'constant',constant_values=0#连续一样的值填充
     )
    return X_paded
np.random.seed(1)
x = np.random.randn(4,3,3,2)
x_paded = zero_pad(x,2)
#查看信息
# print ("x.shape =", x.shape)
# print ("x_paded.shape =", x_paded.shape)
# print ("x[1, 1] =", x[1, 1])
# print ("x_paded[1, 1] =", x_paded[1, 1])
#
# #绘制图
# fig , axarr = plt.subplots(1,2)  #一行两列
# axarr[0].set_title('x')
# axarr[0].imshow(x[0,:,:,0])
# axarr[1].set_title('x_paded')
# axarr[1].imshow(x_paded[0,:,:,0])

#1.3.2单步卷积
def conv_single_step(a_slice_prev,W,b):
    '''
    在前一层的激活输出的一个片段上应用一个由参数W定义的过滤器.
    这里的切片大小与过滤器大小相同
    :param a_slice_prev: 输入数据的一个片段,维度为(过滤器大小,过滤器大小,上一通道数)
    :param W: 权重参数,包含在了一个矩阵中,维度为(过滤器大小,过滤器大小,上一通道数)
    :param b: 偏置参数,包含在了一个矩阵中,维度为(1,1,1)
    :return: 
    Z-在输入数据的片X上卷积滑动窗口(w,b)的结果
    '''
    s = np.multiply(a_slice_prev,W)+b
    Z = np.sum(s)
    return Z
np.random.seed(1)

#这里切片大小和过滤器大小相同
a_slice_prev = np.random.randn(4,4,3)
W = np.random.randn(4,4,3)
b = np.random.randn(1,1,1)

Z = conv_single_step(a_slice_prev,W,b)

#print("Z = " + str(Z))

##1.3.3 卷积神经网络,前向传播

def conv_forward(A_prev,W,b,hparameters):
    '''
    实现卷积函数的前向传播
    :param A_prev:上一层激活输出矩阵,维度为(M,n_H_prev,n_W_prev,n_C_prev,(样本数量,上一层图像的高度,上一层图像的宽度,上一层过滤器数量) 
    :param W: 权重矩阵,维度为(f,f,n_C_prev,n_C),(过滤器大小,过滤器大小,上一层的过滤器数量,这一层的过滤器数量)
    :param b: 偏置矩阵,维度为(1,1,1,n_C),(1,1,1,这一层的过滤数量)
    :param hparameters: 包含了"stride"与"pad"的超参数字典.
    :return: 
    Z-卷积输出,维度为(m,n_H,n_W,n_C),(样本数,图像的高度,图像的宽度,过滤器数量)
    cahce -缓存了一些反向传播函数conv_backward()需要的一些数据
    '''

    #获取来自上一层数据的基本信息
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape

    #获取权重矩阵的基本信息
    (f,f,n_C_prev,n_C)=W.shape

    #获取超参数hparameters的值
    stride = hparameters['stride']
    pad = hparameters['pad']

    #计算卷积后的图像的宽度高度,参考上面公式,使用int()来进行板除
    n_H = int((n_H_prev - f +2*pad)/stride)+1
    n_W = int((n_W_prev-f +2*pad)/stride)+1

    #使用0来初始化卷积输出Z
    Z= np.zeros((m,n_H,n_W,n_C))

    #通过A_prev创建填充过了的A_prev_pad
    A_prev_pad = zero_pad(A_prev,pad)

    for i in range(m):#便利样本
        a_prev_pad = A_prev_pad[i]#选择第i个样本的扩充后的激活矩阵
        for h in range(n_H):#在输出的垂直轴上上循环
            for w in range(n_W):#在输出的水平轴上循环
                for c in range(n_C):#循环便利输出的通道)
                    #定位当前的切片位置
                    vert_start = h*stride #竖向,开始的位置
                    vert_end = vert_start+f#竖向,结束的位置
                    horiz_start = w*stride#横向,开始的位置
                    horiz_end = horiz_start+f#横向,结束的位置
                    #切片位置定位好了我们就把它取出来,需要注意的是我们是'穿透'取出来的,
                    #自行脑补一下吸管插入一层层的橡皮泥就明白了
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    #执行单步卷积
                    Z[i,h,w,c]=conv_single_step(a_slice_prev,W[:,:,:,c],b[0,0,0,c])

    #数据处理完毕,验证数据格式是否正确
    assert(Z.shape == (m,n_H,n_W,n_C))

    #存出一些缓存值,以便反向传播
    cache = (A_prev,W,b,hparameters)

    return(Z,cache)

np.random.seed(1)

A_prev = np.random.randn(10,4,4,3)
W = np.random.randn(2,2,3,8)
b = np.random.randn(1,1,1,8)

hparameters = {"pad" : 2, "stride": 1}

Z , cache_conv = conv_forward(A_prev,W,b,hparameters)

print("np.mean(Z) = ", np.mean(Z))
print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])


#1.4.1池化层的前向传播

def pool_forward(A_prev,hparameters,mode = 'max'):
    '''
    实现池化层的前向传播
    :param A_prev: 输入的数据,维度为(m,n_H_prev,n_W_prev,n_C_prev)
    :param hparameters: 包含了'f'和'stride'的超参数字典
    :param mode: 模式选择['max'|'average']
    :return: 
    A-池化层的输出,维度为(m,n_H,n_W,n_C)
    cache -存储了一些反向传播需要用到的值,包含了输入和超参数的字典.
    '''

    #获取数据的基本信息
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape

    #获取超参数输的信息
    f = hparameters['f']
    stride = hparameters['stride']

    #计算输出维度
    n_H = int((n_H_prev - f)/stride)+1
    n_W = int((n_W_prev - f)/stride)+1
    n_C = n_C_prev

    #初始化输出矩阵
    A = np.zeros((m,n_H,n_W,n_C))

    for i in range(m):#遍历样本
        for h in range(n_H):#在输出的垂直轴上循环
            for w in range(n_W):#在输出水平轴上循环
                for c in range(n_C):#循环便利输出的通道
                    #定位当前的切片位置
                    vert_start = h*stride#竖向,开始的位置
                    vert_end = vert_start +f#竖向,结束的位置
                    horiz_start = w*stride#横向,开始的位置
                    horiz_end = horiz_start+f#横向,结束的位置
                    #定位完毕,开始切割
                    a_slice_prev = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]


                    #对切片进行池化操作
                    if mode == 'max':
                        A[i,h,w,c] = np.max(a_slice_prev)
                    elif mode == 'average':
                        A[i,h,w,c] = np.mean(a_slice_prev)
    #池化完毕,校验数据格式
    assert(A.shape == (m,n_H,n_W,n_C))

    #校验完毕,开始存储用于反向传播的值
    cache = (A_prev,hparameters)

    return A,cache
np.random.seed(1)
A_prev = np.random.randn(2,4,4,3)
hparameters = {"f":4 , "stride":1}

A , cache = pool_forward(A_prev,hparameters,mode="max")
A, cache = pool_forward(A_prev, hparameters)
print("mode = max")
print("A =", A)
print("----------------------------")
A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print("A =", A)