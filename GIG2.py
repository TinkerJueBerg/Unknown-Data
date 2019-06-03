import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle 
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV
from sklearn.decomposition import PCA
from sklearn import preprocessing
#from sklearn.grid_search import GridSearchCV 
from sklearn.linear_model import LinearRegression,LassoCV,RidgeCV,ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#np.set_printoptions(threshold=np.inf) #全部输出 
pca = PCA(n_components=20)


df = pd.read_excel('data.xlsx')
train = df




X_train = train.iloc[:,1:]
X_train = X_train.drop('AA',axis=1)
Y_train = train.iloc[:,0]

def cleandata(X_train):    
    X_train.drop(['G','I','J','Q','AD','AC','R','V','N'], axis=1, inplace=True)       
cleandata(X_train)



"""
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(23, 23))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True,
        annot=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(X_train)
"""
#print(Y_train)
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
#print(X_train_minmax)
xo_train,xo_test, yo_train, yo_test =  train_test_split(
        X_train,Y_train,test_size=0.1, random_state=0) 

model1 = linear_model.ElasticNetCV(alphas=[0.01,0.1,1], 
                                   l1_ratio=[0.001,0.01,0.1,0.3],  
                                   max_iter=10000)
#model1 = linear_model.ElasticNet(l1_ratio=0.7,alpha=0.02)
model2 = Lasso(alpha=0.1)
model3 = RidgeCV(alphas=[0.1,0.3,0.5,0.7])


model1.fit(xo_train, yo_train)
model2.fit(xo_train, yo_train)
model3.fit(xo_train, yo_train)

#print('系数矩阵:\n',model.coef_)6
#print('线性回归模型:\n',model)
predicted1 = model1.predict(xo_test)
predicted2 = model2.predict(xo_test)
predicted3 = model3.predict(xo_test)
n = np.arange(13)
n = n.reshape(13,1)

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.plot(n,predicted1)
#plt.plot(n,predicted2)
plt.plot(n,predicted3)
plt.plot(n,yo_test)
plt.sca(ax1)
plt.show()

predicted1 = predicted1.reshape(-1,1)
predicted2 = predicted2.reshape(-1,1)
predicted3 = predicted3.reshape(-1,1)
yo_test = np.array(yo_test)
yo_test = yo_test.reshape(-1,1)
#print(yo_test)

"""
for i in range(len(n)):
    #print("测试输入：",xo_test)
    print("第n个:",i)
    print("预测值：",predicted1[i])
   # print("预测值：",predicted2[i])
   # print("预测值：",predicted3[i])
    print("验证：",yo_test[i])
"""
predicted =( predicted1 +  predicted3) / 2

loss_sum = 0
loss = []
for i in range(len(n)):
    a = (yo_test[i] - predicted[i])
    loss.append(a)
    loss_sum = abs(loss[i]) + loss_sum
    #print(loss_sum)
loss_function = loss_sum / len(n)
print("loss_mean=",loss_function)