import pandas as pd
from sklearn.decomposition import PCA
def reduce_dimension():# 预测用户购买商品种类
    # 1.获取数据
    order_products=pd.read_csv("D:/instacart/order_products__prior.csv")
    products=pd.read_csv("D:/instacart/products.csv")
    orders=pd.read_csv("D:/instacart/orders.csv")
    aisles=pd.read_csv("D:/instacart/aisles.csv")
    # 2.合并表
    # 合并aisles和products
    table1=pd.merge(aisles,products,on=["aisle_id","aisle_id"])
    # 合并table1和order_products
    table2=pd.merge(table1,order_products,on=["product_id","product_id"])
    # 合并table2和orders
    table3=pd.merge(table2,orders,on=["order_id","order_id"])
    # 3.找到user_id和aisle_id的关系
    table=pd.crosstab(table3["user_id"],table3["aisle_id"])# 透视表
    # 4.PCA降维
    transfer=PCA(n_components=0.95)
    data=transfer.fit_transform(table)
    print(data,data.shape)

    return None
if __name__ == '__main__':
    reduce_dimension()