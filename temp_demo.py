import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def ShowInfo() :
    print(dataset.info())
    print(dataset.describe())
    dataset.plot(x='MaxTemp',y='MinTemp',style="o")
    plt.title('Min-Max temp')
    plt.xlabel('min temp')
    plt.ylabel('max temp')
    plt.show()
def ShowModel() :
    plt.scatter(x_test,y_test)
    plt.scatter(x_test,y_pred,color='yellow')
    plt.plot(x_test,y_pred,color='red',linewidth=2)
    plt.show()
def ShowModelQ() :
    df1.plot(kind="bar",figsize=(16,10))
    #df1.plot(kind="line",figsize=(16,10))
    plt.show()
#pandas อ่าน csv file
dataset=pd.read_csv("Weather.csv")

#train_test_split
x=dataset["MinTemp"].values.reshape(-1,1)
y=dataset["MaxTemp"].values.reshape(-1,1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#training model
model=LinearRegression()
model.fit(x_train,y_train)

#testing model
y_pred=model.predict(x_test)

#compare TrueData & PredData 20 record
df=pd.DataFrame({'Actually':y_test.flatten(),'Predicted':y_pred.flatten()}) #key dataframe ต้องเป็น 1 มิติ
df1=df.head(20)
print("MAE = ",metrics.mean_absolute_error(y_test,y_pred))
print("MSE = ",metrics.mean_squared_error(y_test,y_pred))
print("RMSE = ",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

#หา r-squared 0-1 ยิ่งเยอะยิ่งดี
print("Score = ",metrics.r2_score(y_test,y_pred))