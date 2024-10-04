import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
data=pd.read_csv("D:/dataset/train.csv")
# 数据处理
# 缩小数据范围
data=data.query("x<2.3&x>2&y<1.5&y>1.0")# query方法用于过滤数据，选择一个小范围
# 处理时间特征
time_value=pd.to_datetime(data['time'],unit='s')# 单位为秒
date=pd.DatetimeIndex(time_value)# 转换为时间索引
data['day']=date.day
data['weekday']=date.weekday
data['hour']=date.hour
# 过滤掉签到次数少的地点
place_count=data.groupby('place_id').count()# 统计每个地点签到数量
data_final=data[data["place_id"].isin(place_count[place_count>3].index.values)]
# 筛选特征值和目标值
x=data_final[["x","y","accuracy","day","weekday","hour"]]
y=data_final["place_id"]

# 数据集划分
x_train,x_test,y_train,y_test=train_test_split(x,y)

# 标准化
transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

# KNN算法预估器
estimator=KNeighborsClassifier()

# 加入网格搜索与交叉验证
# 参数准备
param_dict={"n_neighbors":[3,5,7,9,11]}
estimator=GridSearchCV(estimator,param_grid=param_dict,cv=3)
estimator.fit(x_train,y_train)

# 5.模型评估
# 方法1：直接对比真实值和预测值
y_pridict = estimator.predict(x_test)
print(f"预测结果：{y_pridict}")
print(f"直接对比真实值和预测值：{y_test == y_pridict}")
# 方法2：计算准确率
score = estimator.score(x_test, y_test)
print(f"准确率为：{score}")
print(f"最佳参数：{estimator.best_params_}")
print(f"最佳结果：{estimator.best_score_}")
print(f"最佳估计器：{estimator.best_estimator_}")
print(f"交叉验证结果：{estimator.cv_results_}")

