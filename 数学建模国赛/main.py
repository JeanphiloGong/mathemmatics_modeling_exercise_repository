# 导入需要的库
import pandas as pd
import numpy as np
from solution import Solution

# 创建实例
sol = Solution()

# 读取数据
thes = pd.read_csv('data/basis.csv', encoding='gbk')
comp = pd.read_csv('data/composition.csv', encoding='gbk')
unca = pd.read_csv('data/uncategorized.csv', encoding='gbk')
new_comp = pd.read_csv('n_data/new_comp.csv', encoding='gbk')

# (第一问)
# 检查数据的有效性
sol.effectiveness(comp)

# 设置数据不省略
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# 对化学成分进行统计学分析
# 填充缺失值
new_comp.fillna(0, inplace=True)

# 对数据进行分组并排序
group_comp = new_comp.groupby(['类型','表面风化'])
sorted_df = new_comp.sort_values(by=['类型', '表面风化'])

# 保存数据
sorted_df.to_csv('data/group_comp.csv', index=False)

# 概括
summary = group_comp.describe()
summary.to_csv('data/summary.csv')

# 均值
group_comp.fillna(0, inplace=True)
mean = group_comp.mean()
mean.loc['铅钡（风化-无风化）'] = mean.iloc[1, :] - mean.iloc[0, :]
mean.loc['高钾（风化-无风化）'] = mean.iloc[3, :] - mean.iloc[2, :]
mean.sum(axis=1)

# 检查数据的相关性并绘图
sol.correlation(thes, '类型', '颜色')

# 检查多元相关性
sol.cross_table(thes, '表面风化', '纹饰', '类型')

# 观察化学成分的分类趋势
# 导入趋势数据
trend = pd.read_csv('data/trend_composition.csv', encoding='gbk')

# 进行绘图
df = trend[trend['类型'] == '铅钡'].iloc[:, 1:16]
sol.trend1(df, '表面风化', '铅钡')

# 对数据进行预测 
# 读取数据
pred_comp = pd.read_csv('data/group_comp.csv')
pred_comp.fillna(0, inplace=True)

# 选取风化后的铅钡和高钾的数据
w_hp = pred_comp[(pred_comp['表面风化'] == '风化') & (pred_comp['类型'] == '高钾')]
w_lb = pred_comp[(pred_comp['表面风化'] == '风化') & (pred_comp['类型'] == '铅钡')]

# 选择预测标准
# 查看均值
avg = group_comp.mean()
# 计算预测标准
hp_standard = avg.iloc[3, :] - avg.iloc[2, :]
lb_standard = avg.iloc[1, :] - avg.iloc[0, :]

# 进行预测
w_hp.iloc[:, 1:15] = w_hp.iloc[:, 1:15] - hp_standard
w_lb.iloc[:, 1:15] = w_lb.iloc[:, 1:15] - lb_standard

# 修改标签
w_hp['表面风化'] = w_hp['表面风化'].replace('风化', '风化预测点')
w_lb['表面风化'] = w_lb['表面风化'].replace('风化', '风化预测点')

# 将新增数据添加到原数据集中进行预测
pre_data = pd.concat([trend, w_hp, w_lb], axis=0)

# 进行绘图预测
df = pre_data[pre_data['类型'] == '高钾'].iloc[:, 1:16]
sol.trend1(df, '表面风化', '高钾预测')

# 第二问
# 统计分类规律
# 导入数据
cate_data = pd.read_csv('data/group_comp.csv')
cate_data['表面风化'] = cate_data['表面风化'].replace('风化', '风化点')

# 添加分类标签
cate_data['表面风化'] = cate_data['表面风化'] + "-" + cate_data['类型']

# 进行分布图绘图
df = cate_data[cate_data['表面风化'].str.contains('风化')].iloc[:, 1:16]
sol.trend1(df, '表面风化', '总体')

# 根据上方分布图挑选优质数据绘制箱形图
sol.box_plot_trend1(df, '表面风化', '总体')

# 进行聚类分析
type = '高钾'
weathering = '风化'
# 导入趋势数据
trend = pd.read_csv('data/trend_composition.csv', encoding='gbk')

# 进行分布图绘图进行观察
df = cate_data[cate_data['表面风化'].str.contains('风化点')].iloc[:, 1:16]
sol.trend1(df, '表面风化', '总体')

# 选中数据
df = trend[(trend['类型'] == type) & (trend['表面风化'] == weathering)].iloc[:, :15]
df.fillna(0, inplace=True)
# 丢弃非数字列
df_numeric = df.drop('文物采样点', axis=1)

# 使用肘部法则确定k值
sol.k_mean(df, df_numeric,  '二氧化硅(SiO2)')

# 进行二元预测
sol.two_features(df, df_numeric, '二氧化硅(SiO2)', '氧化铅(PbO)', type, weathering, cate=[0, 1, 2, 3, 4, 5], k=4)

# 选择三个特征进行聚类
sol.three_features(df, '二氧化硅(SiO2)','氧化钠(Na2O)', '氧化钾(K2O)')

#第三问
# 进行预测未知数据
weathering = '无风化'
# 导入我们需要预测的数据
un_cate = pd.read_csv('data/uncategorized.csv', encoding='gbk')
# 进行预测
sol.prediction(cate_data, weathering, un_cate)

# 第四问
# 进行化学样品之间的相关性分析
# 导入趋势数据
trend = pd.read_csv('data/trend_composition.csv', encoding='gbk')
# 进行相关性分析并绘图
sol.corr_comp(trend, '高钾', '无风化')


