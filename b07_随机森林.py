from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import seaborn as sns
import matplotlib.pyplot as plt
california=fetch_california_housing()
x=california.data
y=california.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
# 使用随机森林进行回归
rf=RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x_train,y_train)
y_predict=rf.predict(x_test)
# 输出性能指标
mse=mean_squared_error(y_test,y_predict)# 均方误差
mae=mean_absolute_error(y_test,y_predict)# 平均绝对误差
r2=r2_score(y_test,y_predict)# R方值
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared Score: {r2:.2f}")

# 绘制预测值与实际值的散点图
plt.scatter(y_test,y_predict,alpha=0.5)
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.title("Actual vs Predicted Values (Random Forest Regression)")
plt.show()

# 绘制特征重要性
feature_importances=rf.feature_importances_# 特征重要性
features=california.feature_names
sns.barplot(x=feature_importances,y=features)# 柱状图
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance im Random Forest Regression")
plt.show()




