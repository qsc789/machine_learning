from sklearn.preprocessing import StandardScaler
import pandas as pd
def stand_demo():
    """
    标准化
    :return:
    """
    data=pd.read_csv("D:/dating.txt")
    data=data.iloc[:,:3]
    transfer=StandardScaler()
    data_new=transfer.fit_transform(data)
    print(data_new)

    return None
if __name__ == '__main__':
    stand_demo()