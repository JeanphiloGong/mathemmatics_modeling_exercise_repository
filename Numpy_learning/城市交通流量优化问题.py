# 任务要求：

# 给定一个包含N个路口的数据集，每个路口的坐标由x和y值表示。你可以使用Numpy创建一个N×2的数组来表示这些坐标。
# 使用Numpy的argsort函数，沿着两个轴（x轴和y轴）分别找到每个路口的最近邻路口。最终你需要得到一个N×2的数组，其中第i行表示第i个路口的最近邻路口的索引。
# 使用Python和Numpy编写一个函数，接受一个路口坐标数组作为输入，并返回最近邻路口的索引数组。
# 验证你的函数是否正确工作，并对结果进行可视化展示。你可以使用matplotlib等库来实现可视化。

(本示例暂无结果，大体方向可以使用np.argparttion函数进行解决）
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Urban Traffic Flow Optimization Problem


# In[2]:


#首先导入Numpy进行数据分析和matplotlib库进行结果可视化和验证
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn; seaborn.set()  # 设置绘图风格




# In[3]:


# 模拟交通状况数据，每个路口坐标使用x和y表示
rand = np.random.RandomState(23)
td = rand.rand(50,2) #td(traffic data)

# 打印路口坐标点
print(td)

# 观察一下点的分布
plt.scatter(td[:,0], td[:,1], s=15);

# 现在我们标出各个路口的索引，方便我们之后进行选择
for i, point in enumerate(td):
    plt.annotate(str(i), (point[0], point[1]), textcoords="offset points", xytext=(0, 5), ha ='center')

# 显示图形
plt.show()


# In[4]:


# 通过两坐标的平方差之和得到两点之间的距离
dist_sq = np.sum((td[:,np.newaxis,:] - td[np.newaxis,:,:]) ** 2,axis=-1)


# In[5]:


# 可以使用dist_sq.diagonal()进行检验，对角线的值是点到本身的距离，值应该为零，
# dist_sq.diagonal()


# In[6]:


# 现在我们通过np.argpartition()函数找出距离所选点的K（本题为最近，即为1）个最近邻
K = 1
nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)


# In[7]:


# 现在先将每个点和他最近的点进行可视化连接进行大体观察
plt.scatter(td[:,0], td[:,1], s=15)

K = 1
# 为每条线段设置不同的颜色
colors = plt.cm.Set1(range(50))

for i in range(td.shape[0]):
    for j in nearest_partition[i, :K+1]:
        # 画一条从td[i]到td[j]的线段
        # 使用zip方法实现
        plt.plot(*zip(td[j],td[i]),color = colors[i % 50])
        


# In[8]:


# 现在我们接受一个坐标点数据的输入
in_data = np.array(input("输入你的坐标数据:[ , ]"))

# 确保in_data是一个一维数组
if in_data.ndim == 0:
    in_data = np.expand_dims(in_data, axis=0)

# 使用np.where()方法找到数据并返回索引
indices = np.where((td[:,0] == in_data[0]))

# 如果目标点存在于数组中，返回其最近点的坐标，否则打印未找到该点的消息
if len(indices[0]) > 0:
    
    # 得到最近点的索引
    nearest_dot_index = nearest_partition[indices[0]]
    nearest_dot_index = np.squeeze(nearest_dot_index)  # 将0维数组转换为1维数组
    
    # 从数据集中找到这个点
    target_dot = td[nearest_dot_index]
    
    # 返回该点的坐标
    print("距离该点最近的点的坐标为：",target_dot)
else:
    print("没有找到该坐标点")






