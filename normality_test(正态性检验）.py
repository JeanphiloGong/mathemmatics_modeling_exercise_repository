#!/usr/bin/env python
# coding: utf-8

# In[2]:


# task 
# 假设你是一位汽车制造商的质量控制工程师。
# 你想要确定一种特定型号的汽车发动机的燃油效率是否服从正态分布。
# 你收集了10辆这种型号汽车的燃油效率数据（单位：升/百公里），数据如下：
# 15.2, 14.9, 16.5, 16.3, 15.7, 15.8, 15.9, 16.1, 15.4, 16.0


# In[3]:


# 使用Numpy导入数据
import  numpy as np
# 使用SciPy中的stats模块执行正态性检验
import scipy.stats as stats


# In[21]:


# 导入数据，建立一个数组(10辆汽车的燃油效率数据)
data = np.array([15.2, 14.9, 16.5, 16.3, 15.7, 15.8, 15.9, 16.1, 15.4, 16.0])

# 使用Shapiro-Wilk检验
shapiro_test = stats.shapiro(data)
print("Shapiro-Wilk检验结果： ")
print("统计量（W)          :",shapiro_test.statistic)
print("p值                 : ",shapiro_test.pvalue)


# In[24]:


# 使用Kolmogorov-Smirnov检验
ks_test = stats.kstest(data,'norm')
print(data)
print("\nKolmogorov-Smirnov检验结果：")
print("统计量（D）                 :", ks_test.statistic)
print("p值                        :", ks_test.pvalue)


# In[ ]:




