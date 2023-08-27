# 导入需要的模块
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Solution():
    """存储模型中函数的类"""

    # 计算每一行的数据是否有效
    def effectiveness(self, data):
        uneffectiveness = []
        for n in range(data.index.size):
            total = data.iloc[n, 1:].sum()
            if total >= 80:
                print(f"{data.iloc[n, 0]}采样点数据为：{total},该数据合格")
            else:
                uneffectiveness.append(data.iloc[n, 0])
                print(f"{data.iloc[n, 0]}采样点数据为：{total},该数据不合格")
        return uneffectiveness

    # 对数据相关性进行分析
    def correlation(self, setname, name1, name2):
        crosstab = pd.crosstab(setname[name1], setname[name2])
        sns.heatmap(crosstab, annot=True, cmap='YlGnBu')
         # 添加标题
        plt.title(f"{name1}与{name2}之间的相关性")
        plt.savefig(f"figures/{name1}与{name2}之间的相关性.png")
        plt.show()

    # 绘制2对1的相关性分析交叉表图
    def cross_table(self, df, name1, name2, name3):
        cross_table = pd.crosstab(index=df[name1], columns=[df[name2], df[name3]])
        print(cross_table)
        plt.figure(figsize=(10, 6))
        sns.heatmap(cross_table, annot=True, cmap='YlGnBu', cbar=True, linewidths=0.5)
        plt.title(f'{name2}、{name3}与{name1}的交叉表')
        plt.savefig(f'figures/{name2}、{name3}与{name1}的交叉表')
        plt.show()
        
    # 绘制散点图
    def trend1(self, df, name1, name2): # 鲜艳风格
        unique_values = df[name1].unique()
        
        # 自动生成颜色映射
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_mapping = {val: default_colors[i % len(default_colors)] for i, val in
                         enumerate(unique_values)}
        
        # 自动生成标记映射
        markers = ['o', 's', '^', 'x', 'p', '*', 'v', '<', '>', '1', '2', '3', '4']
        marker_mapping = {val: markers[i % len(markers)] for i, val in enumerate(unique_values)}
        
        plt.figure(figsize=(12, 8))
        
        labels_added = set()
        for column in df.columns:
            if column not in [name1, name2]:
                for value in unique_values:
                    subset = df[df[name1] == value].dropna(subset=[column])
                    label = value if value not in labels_added else ""    
                    plt.scatter([column] * len(subset), subset[column], 
                                color=color_mapping[value], 
                                marker=marker_mapping[value],
                                label=label, alpha=0.75)
                    labels_added.add(value)
            
        plt.legend()
        plt.title(f'各种物质的分类 vs 数值({name2})')
        plt.xlabel('物质的分类')
        plt.ylabel('数值')
        plt.xticks(rotation=45)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f'figures/各种物质的分类 vs 数值({name2})')
        plt.show()
        
    # 绘制箱形图
    def box_plot_trend1(self, df, name1, name2):
        unique_values = df[name1].unique()
        
        # 自动生成颜色映射
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_mapping = {val: default_colors[i % len(default_colors)] for i, val in
                         enumerate(unique_values)}
        
        # 定义用于散点图的标记
        markers = ['o', 's', '^', 'x', 'p', '*', 'v', '<', '>', '1', '2', '3', '4']
        marker_mapping = {val: markers[i % len(markers)] for i, val in enumerate(unique_values)}
        
        plt.figure(figsize=(12, 8))
        
        columns_to_plot = [col for col in df.columns if col not in [name1]]
        
        # 绘制箱型图和散点图
        for i, column in enumerate(columns_to_plot):
            data_to_plot = [df[df[name1] == value][column].dropna() for value in unique_values]
            
            # 使用 position 参数将箱型图放在正确的位置上
            positions = [i + (j * 0.15) for j in range(len(unique_values))]
            boxplots = plt.boxplot(data_to_plot, positions=positions, patch_artist=True, 
            widths=0.20)
            
            # 为每个箱型图设置颜色
            for patch, value in zip(boxplots['boxes'], unique_values):
                patch.set_facecolor(color_mapping[value])
            
            # 增加散点图
            for j, value in enumerate(unique_values):
                subset = df[df[name1] == value].dropna(subset=[column])
                y_values = subset[column]
                jitter = 0.15 * (np.random.rand(len(y_values)) - 0.5)
                plt.scatter([positions[j]] * len(y_values) + jitter, y_values, 
                            color=color_mapping[value], 
                            marker=marker_mapping[value], alpha=0.6)
    
            # 增加标签
            if i == 0:
                for j, value in enumerate(unique_values):
                    plt.plot([], color=color_mapping[value], label=value, 
                           marker=marker_mapping[value], markersize=10, linestyle='None')
                    
        # 更新x轴的标签
        plt.xticks([i + (len(unique_values)-1) * 0.15 / 2 for i in range(len(columns_to_plot))], 
                   columns_to_plot)
    
        plt.legend()
        plt.title(f'各种物质的分布规律箱形图-{name2}')
        plt.xlabel('物质的分类')
        plt.ylabel('数值')
        plt.xticks(rotation=45)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f'figures/各种物质的分布规律箱形图-{name2}')
        plt.show()
        
    # 进行二元聚类划分
    def two_features(self, df, df_numeric, name1, name2, type, weathering, cate=[0, 1, 2], k=3):
        # 选择一个合适的k值
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        df['Cluster'] = kmeans.fit_predict(df_numeric)
        
        # 将聚类结果可视化
        plt.scatter(df[name1], df[name2], c=df['Cluster'])
        plt.title(f'{weathering}-{type}类')
        plt.xlabel(name1)
        plt.ylabel(name2)
        plt.colorbar(ticks=cate)
        plt.savefig(f'figures/{weathering}{type}聚类划分图像({name1}与{name2})')
        plt.show()

    # 进行三元聚类
    def three_features(self, df, name1, name2, name3):
        # 选择三个特征进行聚类
        features = df[[name1, name2, name3]]
        
        # 使用KMeans进行聚类
        kmeans = KMeans(n_clusters=3)
        df['Cluster'] = kmeans.fit_predict(features)
        
        # 可视化聚类结果
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df[name1], df[name2], df[name3], c=df['Cluster'])
        
        ax.set_xlabel(f'{name1}')
        ax.set_ylabel(f'{name2}')
        ax.set_zlabel(f'{name3}')
        
        plt.show()
    # 使用肘部法则确定k值
    def k_mean(self, df, df_numeric, name1):
        count = len(df[name1])
        wcss = []
        for i in range(1, count): 
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, 
               random_state=0)
            kmeans.fit(df_numeric)
            wcss.append(kmeans.inertia_)
        
        plt.plot(range(1, count), wcss)
        plt.title('Elbow Method For Optimal k')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

    # 进行预测未知数据（第三问）
    def prediction(self, cate_data, weathering, un_cate):
        # 划分数据集
        X = cate_data[cate_data['表面风化'].str.contains(weathering)].iloc[:, 1:15]
        y = cate_data[cate_data['表面风化'].str.contains(weathering)].iloc[:, 15]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # 数据缩放
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # 使用支持向量机（SVM）进行预测
        # 训练模型
        clf_svm = SVC(kernel='linear')
        clf_svm.fit(X_train, y_train)
        
        # 验证模型
        y_pred = clf_svm.predict(X_test)
        
        # 计算敏感性
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        print(f"Accuracy: {accuracy:.2f}")
        
        # 进模型效率可视化
        unique_labels = sorted(list(set(y_test)))  # 获取数据集中所有唯一的类别标签并排序它们
        
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=unique_labels, 
                    yticklabels=unique_labels)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f"{weathering}模拟结果混淆矩阵")
        plt.savefig(f"figures/{weathering}模拟结果混淆矩阵")
        plt.show()
        
        # 存储精确度等文件
        df = pd.DataFrame(report).transpose()
        df.to_csv(f'data/{weathering}classification_report.csv')
        
        # 导入我们需要预测的数据
        un_cate = pd.read_csv('data/uncategorized.csv', encoding='gbk')
        un_cate.fillna(0, inplace=True)
        un_cate['表面风化'] = un_cate['表面风化'].replace('风化', '风化点')
        pred = un_cate[un_cate['表面风化'].str.contains(weathering)]
        
        # 进行预测
        cate_pred = clf_svm.predict(pred.iloc[:, 2:])
        pred['预测'] = cate_pred
        pred.to_csv(f'data/{weathering}预测.csv')

    # 化学成分之间的相关性分析(第四问)
    # 进行化学样品之间的相关性分析
    def corr_comp(self, data, type, weathering):
        df = data[(data['类型'] == type) & (data['表面风化'] == weathering)].iloc[:, :15]
        df.fillna(0, inplace=True)
        # 丢弃非数字列
        df_numeric = df.drop('文物采样点', axis=1)
        # 识别全零列
        cols_to_drop = df_numeric.columns[(df_numeric == 0).all()]
        
        # 识别全部NaN的列 (如果你先前使用fillna(0)填充了NaN，这步可能就不需要了)
        cols_to_drop_nan = df_numeric.columns[df_numeric.isna().all()]
        
        # 移除这些列
        df_numeric.drop(columns=cols_to_drop.union(cols_to_drop_nan), inplace=True)
        # 计算相关矩阵
        correlation_matrix = df_numeric.corr()
        
        # 使用seaborn绘制热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f"{weathering}-{type}相关性分析热图")
        plt.savefig(f'figures/{weathering}-{type}相关性分析')
        plt.show()