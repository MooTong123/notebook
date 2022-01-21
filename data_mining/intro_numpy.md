# Numpy库简介

**NumPy**是一个功能强大的**Python**库，主要用于对**多维数组**执行计算。NumPy这个词来源于两个单词-- Numerical和Python。NumPy提供了大量的库函数和操作，可以帮助程序员轻松地进行数值计算。在**数据分析和机器学习领域**被广泛使用。他有以下几个特点：

* 1. numpy内置了**并行运算**功能，当系统有多个核心时，做某种计算时，numpy会自动做并行计算。
* 2. Numpy底层使用**C语言**编写，内部解除了**GIL（全局解释器锁）**，其对数组的操作速度不受Python解释器的限制，**效率远高于纯Python代码**。
* 3. 有一个强大的N维数组对象Array（一种类似于列表的东西）:**ndarray**
* 4. 实用的线性代数、傅里叶变换和随机数生成函数。

总而言之，numpy是一个**非常高效**的用于**处理数值型运算**的包。

## Numpy数组和Python列表性能对比：
比如我们想要对一个Numpy数组和Python列表中的每个元素进行求和,代码如下：


```python
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
```

    Python列表所消耗的时间为：0.0449秒
    numpy方式所消耗的时间为：0.0010秒
    

从中我们看到ndarray的计算速度要快很多，这个简单的例子快了接近于50倍。

机器学习的最大特点就是大量的数据运算，那么如果没有一个快速的解决方案，那可能现在python也在机器学习领域达不到好的效果。

Numpy专门针对ndarray的操作和运算进行了设计，所以数组的存储效率和输入输出性能远优于Python中的嵌套列表，数组越大，Numpy的优势就越明显。

# N维数组ndarray

## ndarray的属性
| 属性名字 |                  属性解释 |
| :-----:| :----: |
| **ndarray.shape** | 数组维度的元组 |
| ndarray.ndim | 数组维数 |
| ndarray.size | 数组中的元素数量 |
| ndarray.itemsize | 一个数组元素的长度（字节） |
| **ndarray.dtype** | 数组元素的类型 |

下面着重介绍一下ndarray的**形状(ndarray.shape)**和**类型(ndarray.dtype)**

## ndarray的形状（ndarray.shape）
首先创建一些数组。


```python
# 创建不同形状的数组
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([1, 2, 3, 4])
c = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
```

分别打印出形状


```python
# 二维数组
a.shape
```




    (2, 3)




```python
# 一维数组
b.shape
```




    (4,)




```python
# 三维数组
c.shape
```




    (2, 2, 3)



## ndarray的类型

| 名称 | 描述 | 简写 |
| :-----:| :----: | :----: |
| np.bool | 用一个字节存储的布尔类型（True或False） | 'b' |
| np.int8 | 一个字节大小，-128 至 127 | 'i1' |
| np.int16 | 整数，-32768 至 32767 | 'i2' |
| np.int32| 整数，-2^31 至 2^32 -1 | 'i4' |
| **np.int64** | 整数，-2^63 至 2^63 - 1 | 'i8' |
| np.uint8 | 无符号整数，0 至 255 | 'u1' |
| np.uint16 | 无符号整数，0 至 65535 | 'u2' |
| np.uint32 | 无符号整数，0 至 2^32 - 1 | 'u4' |
| np.uint64 | 无符号整数，0 至 2^64 - 1 | 'u8' |
| np.float16 | 半精度浮点数：16位，正负号1位，指数5位，精度10位 | 'f2' |
| np.float32 | 单精度浮点数：32位，正负号1位，指数8位，精度23位 | 'f4' |
| **np.float64** | 双精度浮点数：64位，正负号1位，指数11位，精度52位 | 'f8' |
| np.complex64 | 复数，分别用两个32位浮点数表示实部和虚部 | 'c8' |
| np.complex128 | 复数，分别用两个64位浮点数表示实部和虚部 | 'c16' |
| np.object_ | python对象 | 'O' |
| np.string_ | 字符串 | 'S' |
| np.unicode_ | unicode类型 | 'U' |

* 注意：若不指定，**整数默认int64**，**小数默认float64**

我们可以看到，Numpy中关于数值的类型比Python内置的多得多，这是因为Numpy为了能高效处理处理海量数据而设计的。举个例子，比如现在想要存储上百亿的数字，并且这些数字都不超过254（一个字节内），我们就可以将**dtype设置为int8**，这样就**比默认使用int64更能节省内存空间**了。类型相关的操作如下：

### 默认的数据类型:


```python
import numpy as np
a1 = np.array([1, 2, 3])
a1.dtype
```




    dtype('int32')



### 指定dtype：


```python
import numpy as np
a1 = np.array([1, 2, 3], dtype=np.int64)
# 或者 a1 = np.array([1,2,3],dtype="i8")
a1.dtype
```




    dtype('int64')



### 修改dtype：ndarray.astype()


```python
import numpy as np
a1 = np.array([1, 2, 3])
print(a1.dtype)

a2 = a1.astype(np.int64)
print(a2.dtype)
```

    int32
    int64
    

# numpy基本操作

## 生成ndarray数组

### 生成0和1的数组
* **np.ones(shape, dtype)**
* np.ones_like(a, dtype)
* **np.zeros(shape, dtype)**
* np.zeros_like(a, dtype)

一般用于初始化参数列表等，深度学习经常使用。

举例：生成一个4行8列的全是1的数组


```python
ones = np.ones([4, 8])
ones
```




    array([[1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.]])



举例：生成一个和ones形状相同的全0数组


```python
np.zeros_like(ones)
```




    array([[0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0.]])



### 从现有数组生成
* np.array(object, dtype): 深拷贝
* np.asarray(ndarray, dtype)：浅拷贝


```python
original_array = np.array([[1, 2, 3], [4, 5, 6]])
a1 = np.array(original_array)  # 深拷贝
a1
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
a2 = np.asarray(original_array)  # 浅拷贝
a2
```




    array([[1, 2, 3],
           [4, 5, 6]])



当我们改变原数组的元素时，a1的元素不变，a2的元素跟着改变了


```python
original_array[0, 0] = 100
original_array
```




    array([[100,   2,   3],
           [  4,   5,   6]])




```python
a1
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
a2
```




    array([[100,   2,   3],
           [  4,   5,   6]])



### 生成固定范围的数组
* **np.linspace**(start, stop, num, endpoint):等差数列 (指定数量)
* **np.arange**(start,stop, step, dtype)：等差数列 (指定步长)
* np.logspace(start,stop, num)：等比数列

#### np.linspace(start, stop, num, endpoint)
* 创建等差数组 — 指定数量
* 参数
    * start:序列的起始值
    * stop:序列的终止值
    * num:要生成的等间隔样例数量，默认为50
    * endpoint:序列中是否包含stop值，默认为ture


```python
# 在0-100区间内，生成11个目标的等差数列
np.linspace(start=0, stop=100, num=11)
```




    array([  0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,  90., 100.])



#### np.arange(start,stop, step, dtype)
* 创建等差数组 — 指定步长
* 参数
    * step:步长,默认值为1


```python
# 在10到50区间内，生成步长为2的等差数列
np.arange(start=10, stop=50, step=2)
```




    array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42,
           44, 46, 48])



#### np.logspace(start,stop, num,base)
* 创建等比数列
* 参数:
    * num:要生成的等比数列数量，默认为50
    * base：底数，默认是10


```python
# 生成10^x
np.logspace(start=0, stop=2, num=3)
```




    array([  1.,  10., 100.])



### 生成随机数组
#### 正态分布
* **np.random.randn()**
    * 从标准正态分布返回一个或多个样本值
    * 标准正态分布：均值为0，标准差为1
* **np.random.normal(loc,scale,size)**
    * 正态分布，需要设置均值和标准差
    * loc:均值,默认是0
    * scale:标准差，默认是1
    * size: 输出的shape


```python
# 生成size为2x3的标准正态分布（均值为0，标准差为1）的样本值
np.random.randn(2,3)
```




    array([[ 0.6533699 , -0.46007395,  0.47254584],
           [ 0.58634246, -0.25327346, -1.09442417]])




```python
# 生成size为2x4的正态分布（均值为1.75，标准差为1）的样本值
np.random.normal(loc=1.75, scale=1, size=(2,4))
```




    array([[2.48567415, 1.13241629, 0.24196215, 0.31020365],
           [1.66919719, 1.68048261, 3.41271193, 3.40125795]])



#### 均匀分布
* **np.random.rand()**
    * 返回[0-1]内的一组均匀分布的样本值
* **np.random.randint(low,high=None,size=None)**
    * 从一个均匀分布中随机采样，生成一个整数或N维整数数组
    * 取数范围：若high不为None时，取[low,high)之间随机整数，否则取值[0,low)之间随机整数。
* **np.random.uniform(low,high=None,size=None)**
    * 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high。


```python
# 生成0-1之间的符合均匀分布的一个样本值
print(np.random.rand())

# 生成0-1之间的符合均匀分布的，size为2x4的一组样本值
print(np.random.rand(2,4))
```

    0.9979568356514348
    [[0.17730386 0.48566315 0.09793621 0.01278852]
     [0.15596577 0.64693952 0.52916144 0.4368679 ]]
    


```python
# 生成[1,5)之间的符合均匀分布的size为2x4的一组整数值
np.random.randint(low=1,high=5,size=(2,4))
```




    array([[3, 3, 3, 3],
           [4, 4, 3, 4]])




```python
# 生成[1,5)之间的符合均匀分布的size为2x4的一组样本值
np.random.uniform(low=1,high=5,size=(2,4))
```




    array([[2.13243924, 2.57371109, 1.05353739, 2.92320726],
           [1.36604152, 4.10251762, 4.33901933, 1.20899162]])



## 数组的索引和切片
一维、二维、三维的数组如何索引？
* 一维：直接进行索引,切片（类似于Python列表的索引切片）
* 二维：对象[:,:] -- 先行后列
* 三维：对象[:,:,:] -- 对应三个维度

### 一维举例


```python
a1 = np.arange(0,10)
a1
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
# 索引：想要获取下标为2的元素
a1[2]
```




    2




```python
# 切片：想要获取下标为2到5的元素
a1[2:6]
```




    array([2, 3, 4, 5])



### 二维举例


```python
# 生成一个3x8列的二维数组
a2 = np.arange(0,24).reshape((3,8))
a2
```




    array([[ 0,  1,  2,  3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12, 13, 14, 15],
           [16, 17, 18, 19, 20, 21, 22, 23]])




```python
# 索引：想要获取第1行的第7个元素
a2[0,6]
```




    6




```python
# 切片：想要所有行的第0-6列元素
a2[:,0:6]
```




    array([[ 0,  1,  2,  3,  4,  5],
           [ 8,  9, 10, 11, 12, 13],
           [16, 17, 18, 19, 20, 21]])



### 三维举例


```python
# 生成一个2x3x4的三维数组
a3 = np.arange(0,24).reshape(2,3,4)
a3
```




    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])




```python
# 索引：想要获取第一个维度的第0行的第0个元素
a3[0,0,0]
```




    0




```python
# 切片：想要获取第1个维度的前两行的第0至2列
a3[0,:2,:3]
```




    array([[0, 1, 2],
           [4, 5, 6]])



## 数组形状修改
* **ndarray.reshape(shape, order)**
    * reshape是将数组转换成指定的形状，然后返回转换后的结果，对于原数组的形状是不会发生改变的
* **ndarray.resize(new_shape)**
    * resize是将数组转换成指定的形状，会直接修改数组本身。并不会返回任何值。
* **ndarray.T**
    * 数组的转置


```python
# 新建一个3x4的数组
a = np.random.randint(0,10,size=(3,4))
a
```




    array([[9, 8, 1, 5],
           [9, 0, 1, 8],
           [5, 8, 1, 7]])




```python
# 使用reshape进行修改数组形状
a1 = a.reshape((2,6))
print("转换前的数组:{}".format(a))
print("转换后的数组:{}".format(a1))
```

    转换前的数组:[[9 8 1 5]
     [9 0 1 8]
     [5 8 1 7]]
    转换后的数组:[[9 8 1 5 9 0]
     [1 8 5 8 1 7]]
    


```python
# 使用resize进行修改数组形状,直接修改原数组a,没有返回值
a.resize((2,6))
print(a)
print(a.resize((2,6)))
```

    [[9 8 1 5 9 0]
     [1 8 5 8 1 7]]
    None
    


```python
# 使用.T进行数组的转置
print(a)
print(a.T)
```

    [[9 8 1 5 9 0]
     [1 8 5 8 1 7]]
    [[9 1]
     [8 8]
     [1 5]
     [5 8]
     [9 1]
     [0 7]]
    

## 数组类型修改
* **ndarray.astype(type)**
    * 修改成指定数组类型
* **ndarray.tostring([order])**
    * 构建一个包含ndarray的原始字节数据的字节字符串


```python
# 新建一个数组,检查其dtype
a = np.arange(0,24).reshape(3,8)
print(a)
print(a.dtype)
```

    [[ 0  1  2  3  4  5  6  7]
     [ 8  9 10 11 12 13 14 15]
     [16 17 18 19 20 21 22 23]]
    




    dtype('int32')




```python
# 更改a的dtype，改为int8
a1 = a.astype(np.int8)
a1
```




    array([[ 0,  1,  2,  3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12, 13, 14, 15],
           [16, 17, 18, 19, 20, 21, 22, 23]], dtype=int8)




```python
# 构造包含数组中原始数据字节的Python字节
a.tostring()
```




    b'\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00\x05\x00\x00\x00\x06\x00\x00\x00\x07\x00\x00\x00\x08\x00\x00\x00\t\x00\x00\x00\n\x00\x00\x00\x0b\x00\x00\x00\x0c\x00\x00\x00\r\x00\x00\x00\x0e\x00\x00\x00\x0f\x00\x00\x00\x10\x00\x00\x00\x11\x00\x00\x00\x12\x00\x00\x00\x13\x00\x00\x00\x14\x00\x00\x00\x15\x00\x00\x00\x16\x00\x00\x00\x17\x00\x00\x00'



## 数组去重
* np.unique()


```python
# 新建一个由重复元素的数组
temp = np.array([[1, 2, 3, 4],[3, 4, 5, 6]])

# 检查数组里不重复的元素有哪些
np.unique(temp)
```




    array([1, 2, 3, 4, 5, 6])



## 数组间的计算（广播机制）

### 广播原则

### 数组与数的计算

### 数组与数组的计算

### 不同数组的组合

### 数组的切割


```python

```

# ndarray运算

## 逻辑运算

## 通用判断函数

## np.where（三元运算符）

## 统计运算

## 数组排序




```python

```


```python

```
