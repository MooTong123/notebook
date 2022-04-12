#!/usr/bin/env python
# coding: utf-8

# # Numpy库简介

# **NumPy**是一个功能强大的**Python**库，主要用于对**多维数组**执行计算。NumPy这个词来源于两个单词-- Numerical和Python。NumPy提供了大量的库函数和操作，可以帮助程序员轻松地进行数值计算。在**数据分析和机器学习领域**被广泛使用。他有以下几个特点：
# * numpy内置了**并行运算**功能，当系统有多个核心时，做某种计算时，numpy会自动做并行计算。
# 
# 
# * Numpy底层使用**C语言**编写，内部解除了**GIL（全局解释器锁）**，其对数组的操作速度不受Python解释器的限制，**效率远高于纯Python代码**。
# 
# 
# * 有一个强大的N维数组对象Array（一种类似于列表的东西）:**ndarray**
# 
# 
# * 实用的线性代数、傅里叶变换和随机数生成函数。
# 
# 总而言之，numpy是一个**非常高效**的用于**处理数值型运算**的包。

# ## Numpy数组和Python列表性能对比：
# 比如我们想要对一个Numpy数组和Python列表中的每个元素进行求和,代码如下：

# In[1]:


import time
import random
import numpy as np

# Python 列表方式
t1 = time.time()
a = []
for x in range(100000):
    a.append(x**2)
t2 = time.time()
print("Python列表所消耗的时间为：{:.4f}秒".format(t2-t1))

# numpy 方式
t3 = time.time()
b = np.arange(100000)**2
t4 = time.time()
print("numpy方式所消耗的时间为：{:.4f}秒".format(t4-t3))


# 从中我们看到ndarray的计算速度要快很多，这个简单的例子快了接近于50倍。
# 
# 机器学习的最大特点就是大量的数据运算，那么如果没有一个快速的解决方案，那可能现在python也在机器学习领域达不到好的效果。
# 
# Numpy专门针对ndarray的操作和运算进行了设计，所以数组的存储效率和输入输出性能远优于Python中的嵌套列表，数组越大，Numpy的优势就越明显。

# # N维数组ndarray

# ## ndarray的属性
# | 属性名字 |                  属性解释 |
# | :-----:| :----: |
# | **ndarray.shape** | 数组维度的元组 |
# | ndarray.ndim | 数组维数 |
# | ndarray.size | 数组中的元素数量 |
# | ndarray.itemsize | 一个数组元素的长度（字节） |
# | **ndarray.dtype** | 数组元素的类型 |
# 
# 下面着重介绍一下ndarray的**形状(ndarray.shape)**和**类型(ndarray.dtype)**

# ## ndarray的形状（ndarray.shape）
# 首先创建一些数组。

# In[2]:


# 创建不同形状的数组
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([1, 2, 3, 4])
c = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])


# 分别打印出形状

# In[3]:


# 二维数组
a.shape


# In[4]:


# 一维数组
b.shape


# In[5]:


# 三维数组
c.shape


# ## ndarray的类型
# 
# | 名称 | 描述 | 简写 |
# | :-----:| :----: | :----: |
# | np.bool | 用一个字节存储的布尔类型（True或False） | 'b' |
# | np.int8 | 一个字节大小，-128 至 127 | 'i1' |
# | np.int16 | 整数，-32768 至 32767 | 'i2' |
# | np.int32| 整数，-2^31 至 2^32 -1 | 'i4' |
# | **np.int64** | 整数，-2^63 至 2^63 - 1 | 'i8' |
# | np.uint8 | 无符号整数，0 至 255 | 'u1' |
# | np.uint16 | 无符号整数，0 至 65535 | 'u2' |
# | np.uint32 | 无符号整数，0 至 2^32 - 1 | 'u4' |
# | np.uint64 | 无符号整数，0 至 2^64 - 1 | 'u8' |
# | np.float16 | 半精度浮点数：16位，正负号1位，指数5位，精度10位 | 'f2' |
# | np.float32 | 单精度浮点数：32位，正负号1位，指数8位，精度23位 | 'f4' |
# | **np.float64** | 双精度浮点数：64位，正负号1位，指数11位，精度52位 | 'f8' |
# | np.complex64 | 复数，分别用两个32位浮点数表示实部和虚部 | 'c8' |
# | np.complex128 | 复数，分别用两个64位浮点数表示实部和虚部 | 'c16' |
# | np.object_ | python对象 | 'O' |
# | np.string_ | 字符串 | 'S' |
# | np.unicode_ | unicode类型 | 'U' |
# 
# * 注意：若不指定，**整数默认int64**，**小数默认float64**

# 我们可以看到，Numpy中关于数值的类型比Python内置的多得多，这是因为Numpy为了能高效处理处理海量数据而设计的。举个例子，比如现在想要存储上百亿的数字，并且这些数字都不超过254（一个字节内），我们就可以将**dtype设置为int8**，这样就**比默认使用int64更能节省内存空间**了。类型相关的操作如下：

# ### 默认的数据类型:

# In[6]:


# 新建一个数组，展示其默认数据类型
a1 = np.array([1, 2, 3])
a1.dtype


# ### 指定dtype：

# In[7]:


# 新建一个数组，并指定数据类型
a1 = np.array([1, 2, 3], dtype=np.int64)
a1.dtype


# ### 修改dtype：ndarray.astype()

# In[8]:


# 新建一个数组，展示其默认数据类型
a1 = np.array([1, 2, 3])
print(a1.dtype)

# 修改数组的数据类型
a2 = a1.astype(np.int64)
print(a2.dtype)


# # numpy基本操作

# ## 生成ndarray数组

# ### 生成0和1的数组
# * **np.ones(shape, dtype)**
# * np.ones_like(a, dtype)
# * **np.zeros(shape, dtype)**
# * np.zeros_like(a, dtype)
# 
# 一般用于初始化参数列表等，深度学习经常使用。
# 
# 举例：生成一个4行8列的全是1的数组

# In[9]:


# 生成一个4行8列的全是1的数组
ones = np.ones([4, 8])
ones


# 举例：生成一个和ones形状相同的全0数组

# In[10]:


# 生成一个和数组ones形状相同的全0数组
np.zeros_like(ones)


# ### 从现有数组生成
# * np.array(object, dtype): 深拷贝
# * np.asarray(ndarray, dtype)：浅拷贝

# In[11]:


# 新建一个数组
original_array = np.array([[1, 2, 3], [4, 5, 6]])

# 深拷贝
a1 = np.array(original_array)
a1


# In[12]:


# 浅拷贝
a2 = np.asarray(original_array)
a2


# 当我们改变原数组的元素时，a1的元素不变，a2的元素跟着改变了

# In[13]:


# 我们修改原始数组的值
original_array[0, 0] = 100
original_array


# In[14]:


# 检查深拷贝的a1数组里面的值是否发生变化
a1


# In[15]:


# 检查浅拷贝的a2数组里面的值是否发生变化
a2


# ### 生成固定范围的数组
# * **np.linspace**(start, stop, num, endpoint):等差数列 (指定数量)
# 
# 
# * **np.arange**(start,stop, step, dtype)：等差数列 (指定步长)
# 
# 
# * np.logspace(start,stop, num)：等比数列

# #### np.linspace(start, stop, num, endpoint)
# * 创建等差数组 — 指定数量
# * 参数
#     * start:序列的起始值
#     * stop:序列的终止值
#     * num:要生成的等间隔样例数量，默认为50
#     * endpoint:序列中是否包含stop值，默认为ture

# In[16]:


# 在0-100区间内，生成11个目标的等差数列
np.linspace(start=0, stop=100, num=11)


# #### np.arange(start,stop, step, dtype)
# * 创建等差数组 — 指定步长
# * 参数
#     * step:步长,默认值为1

# In[17]:


# 在10到50区间内，生成步长为2的等差数列
np.arange(start=10, stop=50, step=2)


# #### np.logspace(start,stop, num,base)
# * 创建等比数列
# * 参数:
#     * num:要生成的等比数列数量，默认为50
#     * base：底数，默认是10

# In[18]:


# 生成10^x
np.logspace(start=0, stop=2, num=3)


# ### 生成随机数组
# #### 正态分布
# * **np.random.randn()**
#     * 从标准正态分布返回一个或多个样本值
#     * 标准正态分布：均值为0，标准差为1
#     
#     
# * **np.random.normal(loc,scale,size)**
#     * 正态分布，需要设置均值和标准差
#     * loc:均值,默认是0
#     * scale:标准差，默认是1
#     * size: 输出的shape

# In[19]:


# 生成size为2x3的标准正态分布（均值为0，标准差为1）的样本值
np.random.randn(2,3)


# In[20]:


# 生成size为2x4的正态分布（均值为1.75，标准差为1）的样本值
np.random.normal(loc=1.75, scale=1, size=(2,4))


# #### 均匀分布
# * **np.random.rand()**
#     * 返回[0-1]内的一组均匀分布的样本值
#     
#     
# * **np.random.randint(low,high=None,size=None)**
#     * 从一个均匀分布中随机采样，生成一个整数或N维整数数组
#     * 取数范围：若high不为None时，取[low,high)之间随机整数，否则取值[0,low)之间随机整数。
#     
#     
# * **np.random.uniform(low,high=None,size=None)**
#     * 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high。

# In[21]:


# 生成0-1之间的符合均匀分布的一个样本值
print(np.random.rand())

# 生成0-1之间的符合均匀分布的，size为2x4的一组样本值
print(np.random.rand(2,4))


# In[22]:


# 生成[1,5)之间的符合均匀分布的size为2x4的一组整数值
np.random.randint(low=1,high=5,size=(2,4))


# In[23]:


# 生成[1,5)之间的符合均匀分布的size为2x4的一组样本值
np.random.uniform(low=1,high=5,size=(2,4))


# ## 数组的索引和切片
# 一维、二维、三维的数组如何索引？
# * 一维：直接进行索引,切片（类似于Python列表的索引切片）
# * 二维：对象[:,:] -- 先行后列
# * 三维：对象[:,:,:] -- 对应三个维度

# ### 一维举例

# In[24]:


# 生成一维数组
a1 = np.arange(0,10)
a1


# In[25]:


# 索引：想要获取下标为2的元素
a1[2]


# In[26]:


# 切片：想要获取下标为2到5的元素
a1[2:6]


# ### 二维举例

# In[27]:


# 生成一个3x8列的二维数组
a2 = np.arange(0,24).reshape((3,8))
a2


# In[28]:


# 索引：想要获取第1行的第7个元素
a2[0,6]


# In[29]:


# 切片：想要所有行的第0-6列元素
a2[:,0:6]


# ### 三维举例

# In[30]:


# 生成一个2x3x4的三维数组
a3 = np.arange(0,24).reshape(2,3,4)
a3


# In[31]:


# 索引：想要获取第一个维度的第0行的第0个元素
a3[0,0,0]


# In[32]:


# 切片：想要获取第1个维度的前两行的第0至2列
a3[0,:2,:3]


# ## 数组形状修改
# * **ndarray.reshape(shape, order)**
#     * reshape是将数组转换成指定的形状，然后返回转换后的结果，对于原数组的形状是不会发生改变的  
#     
#     
# * **ndarray.resize(new_shape)**
#     * resize是将数组转换成指定的形状，会直接修改数组本身。并不会返回任何值。
#     
#     
# * **ndarray.T**
#     * 数组的转置

# In[33]:


# 新建一个3x4的数组
a = np.random.randint(0,10,size=(3,4))
a


# In[34]:


# 使用reshape进行修改数组形状
a1 = a.reshape((2,6))
print("转换前的数组:{}".format(a))
print("转换后的数组:{}".format(a1))


# In[35]:


# 使用resize进行修改数组形状,直接修改原数组a,没有返回值
a.resize((2,6))
print(a)
print(a.resize((2,6)))


# In[36]:


# 使用.T进行数组的转置
print(a)
print(a.T)


# ## 数组类型修改
# * **ndarray.astype(type)**
#     * 修改成指定数组类型
#     
#     
# * **ndarray.tostring([order])**
#     * 构建一个包含ndarray的原始字节数据的字节字符串

# In[37]:


# 新建一个数组,检查其dtype
a = np.arange(0,24).reshape(3,8)
print(a)
print(a.dtype)


# In[38]:


# 更改a的dtype，改为int8
a1 = a.astype(np.int8)
a1


# In[39]:


# 构造包含数组中原始数据字节的Python字节
a.tostring()


# ## 数组去重
# * np.unique()

# In[40]:


# 新建一个由重复元素的数组
temp = np.array([[1, 2, 3, 4],[3, 4, 5, 6]])

# 检查数组里不重复的元素有哪些
np.unique(temp)


# ## 数组间的计算（广播机制）

# ### 广播原则
# **如果两个数组的后缘维度（trailing dimension，即从末尾开始算起的维度）的轴长度相符或其中一方的长度为1，则认为他们是广播兼容的。广播会在缺失和（或）长度为1的维度上进行。** 看以下案例分析：
# * shape为(3,8,2)的数组能和(8,3)的数组进行运算吗？
#     * 分析：不能，因为按照广播原则，从后面往前面数，(3,8,2)和(8,3)中的2和3不相等，所以不能进行运算。 
#     
#     
# * shape为(3,8,2)的数组能和(8,1)的数组进行运算吗？
#     * 分析：能，因为按照广播原则，从后面往前面数，(3,8,2)和(8,1)中的2和1虽然不相等，但是因为有一方的长度为1，所以能参与运算。
#     
#     
# * shape为(3,1,8)的数组能和(8,1)的数组进行运算吗？
#     * 分析：能，因为按照广播原则，从后面往前面数，(3,1,4)和(8,1)中的4和1虽然不相等且1和8不相等，但是因为这两项中有一方的长度为1，所以能参与运算。
# 

# ### 数组与数的计算
# 在Python列表中，想要对列表中所有的元素都加一个数，要么采用map函数，要么循环整个列表进行操作。但是NumPy因为**广播机制**的原因，**数组可以直接在数组上进行操作**。示例代码如下：

# In[41]:


# 新建一个数组
a = np.arange(0,8)
a


# In[42]:


a + 1


# In[43]:


a * 2


# ### 数组与数组的计算
# 根据**广播机制**的定义，不同数组之间的运算需要形状相同（或者其中一个维度为1，这样就会对这个维度进行扩展，使得两个数组形状变得相同）。示例代码如下：

# In[44]:


# 新建两个数组
a1 = np.random.randint(1,10,size=(3,2,2))
a2 = np.random.randint(10,20,size=(2,1))
print(a1)
print(a2)


# In[45]:


# 根据广播机制，是可以进行计算的
a1 + a2


# In[46]:


# 新建两个数组
a1 = np.random.randint(1,10,size=(3,2,2))
a2 = np.random.randint(10,20,size=(3,1))
print(a1)
print(a2)


# In[47]:


# 根据广播机制，是不可以进行计算的
# a1 + a2


# ### 不同数组的组合
# 如果有多个数组想要组合在一起，也可以通过一些函数来实现。
# * np.vstack: 将数组按垂直方向进行叠加。数组的列数必须相同才能叠加。
# 
# 
# * np.hstack:将数组按水平方向进行叠加。数组的行必须相同才能叠加。
# 
# 
# * **np.concatenate([],axis)**:将两个数组进行叠加，但是具体是按水平方向还是按垂直方向。则要看axis的参数，如果axis=0，那么代表的是往垂直方向（行）叠加，如果axis=1，那么代表的是往水平方向（列）上叠加，如果axis=None，那么会将两个数组组合成一个一维数组。需要注意的是，如果往水平方向上叠加，那么行必须相同，如果是往垂直方向叠加，那么列必须相同。示例代码如下：

# In[48]:


# 新建两个数组
a = np.array([[1,2],[3,4]]) # 2x2
b = np.array([[5,6]]) # 1x2


# In[49]:


# 按行进行叠加
np.concatenate((a, b), axis=0)


# In[50]:


# 按列进行叠加
np.concatenate((a, b.T), axis=1)


# In[51]:


# 不设置axis，组合成一维数组
np.concatenate((a, b), axis=None)


# ### 数组的切割
# 通过hsplit和vsplit以及array_split可以将一个数组进行切割。
# * np.hsplit:按照水平方向进行切割。
# 
# 
# * np.vsplit:按照垂直方向进行切割。
# 
# 
# * np.array_split:用于指定切割方式，axis=1代表按照列，axis=0代表按照行。

# #### np.hsplit
# 按照水平方向进行切割。用于指定**分割成几列**，可以使用数字来代表分成几部分，也可以使用数组来代表分割的地方。示例代码如下：

# In[52]:


# 新建一个数组
a = np.arange(16.0).reshape(4, 4)
a


# In[53]:


#使用hsplit，分割成两部分
np.hsplit(a,2)


# In[54]:


# #代表在下标为1的地方切一刀，下标为2的地方切一刀，分成三部分
np.hsplit(a,[1,2])


# #### np.vsplit
# 按照垂直方向进行切割。用于指定**分割成几行**，可以使用数字来代表分成几部分，也可以使用数组来代表分割的地方。示例代码如下：

# In[55]:


#代表按照行总共分成2个数组
np.vsplit(a,2) 


# In[56]:


#代表按照行进行划分，在下标为1的地方和下标为2的地方分割
np.vsplit(a,(1,2))


# #### np.array_split
# 用于指定切割方式，在切割的时候需要指定是按照行还是按照列，axis=1代表按照列，axis=0代表按照行。示例代码如下：

# In[57]:


#按照垂直方向切割成2部分
np.array_split(a,2,axis=0)


# # ndarray运算

# ## 逻辑运算-布尔索引
# 如果想要操作符合某一条件的数据，应该怎么操作？
# 
# 假设现在有10名同学，5名功课的数据。现在想要筛选出前5名同学成绩小于60的数据，并设置为0分。

# In[58]:


# 生成10名同学5名功课的数据
score = np.random.randint(40,100,(10,5))
score


# In[59]:


# 提取前5名同学的成绩
test_score = score[:5,:]

# 逻辑判断
test_score < 60


# In[60]:


# 布尔索引：将满足条件的值设置为0
test_score[test_score < 60] = 0
test_score


# ## 通用判断函数
# * np.all；验证任何一个元素是否都符合条件
# * np.any：验证是否有一个元素符合条件

# In[61]:


# 检验数组score里面的值是否都大于60
np.all(score > 60)


# In[62]:


# 检验数组score里面的值是否有小于60的值
np.any(a < 60)


# ## np.where（三元运算符）
# 通过使用np.where能够进行更加复杂的运算
# * np.where(condition,[x, y])

# In[63]:


# 判断前5名同学的前4门课程中，成绩中大于60的设为1，否则设为0
temp_score = score[:5, :4]
temp_score


# In[64]:


# 成绩中大于60的设为1，否则设为0
np.where(temp_score > 60, 1, 0)


# ## 统计运算
# 统计指标也是我们分析问题的一种方式,常见的指标如下：
# 
# |  函数名称   | 描述  |
# | :-----:| :----: |
# | np.min|计算元素的最小值| 
# | np.max|计算元素的最大值|   
# | np.median|计算元素的中位数|   
# |np.mean|计算元素的均值|   
# | np.std|计算元素的标准差|    
# | np.var|计算元素的方差|
# | np.sum|计算元素的和|
# | np.prod|计算元素的积|
# | np.argmin|找出最小值的索引|
# | np.argmax|找出最大值的索引|
# 
# * 注意：里面的参数都有axis指定行或列，详细的axis解释在后文有详细说明，若是在搞不清楚，axis=0或axis=1都试试即可。

# ## 数组排序
# * np.sort():指定轴进行排序。默认是使用数组的最后一个轴进行排序。
# 
# 
# * np.argsort():返回排序后的下标值。

# In[65]:


# 新建一个数组
a = np.random.randint(0,10,size=(3,5))
a


# In[66]:


# 轴的顺序为[0,1],np.sort默认是按照最后一个轴进行排序
# 因为最后一个轴是1，所以就是将最里面的元素进行排序
b = np.sort(a)
b


# In[67]:


# axis=0，表示跨行进行排序，即对每一列的值进行排序
c = np.sort(a,axis=0)
c


# # axis理解
# 一般来讲，我们处理的都是二维表，axis=0指的是行，axis=1，指的是列。但其实不是这么简单理解的，下面来说明来解释一下axis轴的概念。
# 
# 简单来说，**axis=0就是跨行进行计算，axis=1就等于跨列进行计算**。下面的举例说明。

# In[68]:


# 新建一个数组
a = np.array([[0,1,2],[3,4,5]])
a


# 现在我们想要计算每一行的最小值，就是第一行[0,1,2]的最小值是0，第二行[3,4,5]的最小值为3，我们期望得到的是一个[0,3]这样的数组。
# 
# 这时候axis就应该等于1，表示跨列进行计算，因为[0,1,2]是属于第1列第2列第3列的数字。
# 
# 类似的比如求每行的均值，对每行进行排序，都相当于跨列进行操作，所以axis=1.
# 

# In[69]:


# 计算每行的最小值
np.min(a,axis=1)


# 现在我们想要删除第1列的数据,axis应该等于多少呢？
# 
# axis应该等于1，因为删除相当于是对第2列进行操作

# In[70]:


# 删除第2列的数据
np.delete(a,1,axis=1)

