#X1
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import spearmanr,pearsonr,kendalltau
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('New_v1.csv')
X = dataset.iloc[:, 2:17].values
median_srocc = np.zeros(3)
median_pcc = np.zeros(3)
median_kcc = np.zeros(3)
median_rmse = np.zeros(3)

y = dataset.iloc[:, 41].values
rmse_X1=np.zeros(100)
srocc_X1=np.zeros(100)
pcc_X1=np.zeros(100)
kcc_X1=np.zeros(100)
y_pred_overall=[]
y_test_overall=[]

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = i)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    y_train = y_train.reshape(len(y_train),1)
    y_test = y_test.reshape(len(y_test),1)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)
    X_test = sc_X.transform(X_test)
    y_test = sc_y.transform(y_test)
    y_train = y_train.reshape(len(y_train))
    y_test = y_test.reshape(len(y_test))
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    srocc_X1[i],p = spearmanr(y_test, y_pred)
    pcc_X1[i],p = pearsonr(y_test, y_pred)
    kcc_X1[i],p = kendalltau(y_test, y_pred)
    y_test = y_test.reshape(len(y_test),1)
    y_pred = y_pred.reshape(len(y_pred),1)
    rmse_X1[i] = np.sqrt(((sc_y.inverse_transform(y_pred)- sc_y.inverse_transform(y_test)) ** 2).mean())

median_srocc[0] = np.median(srocc_X1)
median_pcc[0] = np.median(pcc_X1)
median_kcc[0] = np.median(kcc_X1)
median_rmse[0] = np.median(rmse_X1)
print('SROCC score for X1 is = ', np.median(srocc_X1))
print('RMSE score for X1 is = ', np.median(rmse_X1))
print('PCC score for X1 is = ', np.median(pcc_X1))
print('KCC score for X1 is = ', np.median(kcc_X1))
y_pred_overall = pd.DataFrame(y_pred_overall)
y_test_overall = pd.DataFrame(y_test_overall)

y_pred_overall = pd.concat([y_pred_overall,pd.DataFrame(y_pred)],axis=1)
y_test_overall = pd.concat([y_test_overall,pd.DataFrame(y_test)],axis=1)


# Y1
y = dataset.iloc[:, 42].values
rmse_Y1=np.zeros(100)
srocc_Y1=np.zeros(100)
pcc_Y1=np.zeros(100)
kcc_Y1=np.zeros(100)

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = i)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    y_train = y_train.reshape(len(y_train),1)
    y_test = y_test.reshape(len(y_test),1)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)
    X_test = sc_X.transform(X_test)
    y_test = sc_y.transform(y_test)
    y_train = y_train.reshape(len(y_train))
    y_test = y_test.reshape(len(y_test))
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    srocc_Y1[i],p = spearmanr(y_test, y_pred)
    pcc_Y1[i],p = pearsonr(y_test, y_pred)
    kcc_Y1[i],p = kendalltau(y_test, y_pred)
    y_test = y_test.reshape(len(y_test),1)
    y_pred = y_pred.reshape(len(y_pred),1)
    rmse_Y1[i] = np.sqrt(((sc_y.inverse_transform(y_pred)- sc_y.inverse_transform(y_test)) ** 2).mean())

median_srocc[1] = np.median(srocc_Y1)
median_pcc[1] = np.median(pcc_Y1)
median_kcc[1] = np.median(kcc_Y1)
median_rmse[1] = np.median(rmse_Y1)
print('SROCC score for Y1 is = ', np.median(srocc_Y1))
print('RMSE score for Y1 is = ', np.median(rmse_Y1))
print('PCC score for Y1 is = ', np.median(pcc_Y1))
print('KCC score for Y1 is = ', np.median(kcc_Y1))
y_pred_overall = pd.concat([y_pred_overall,pd.DataFrame(y_pred)],axis=1)
y_test_overall = pd.concat([y_test_overall,pd.DataFrame(y_test)],axis=1)

# Z1
y = dataset.iloc[:, 43].values
rmse_Z1=np.zeros(100)
srocc_Z1=np.zeros(100)
pcc_Z1=np.zeros(100)
kcc_Z1=np.zeros(100)

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = i)
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    y_train = y_train.reshape(len(y_train),1)
    y_test = y_test.reshape(len(y_test),1)
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train)
    X_test = sc_X.transform(X_test)
    y_test = sc_y.transform(y_test)
    y_train = y_train.reshape(len(y_train))
    y_test = y_test.reshape(len(y_test))
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    
    srocc_Z1[i],p = spearmanr(y_test, y_pred)
    pcc_Z1[i],p = pearsonr(y_test, y_pred)
    kcc_Z1[i],p = kendalltau(y_test, y_pred)
    y_test = y_test.reshape(len(y_test),1)
    y_pred = y_pred.reshape(len(y_pred),1)
    rmse_Z1[i] = np.sqrt(((sc_y.inverse_transform(y_pred)- sc_y.inverse_transform(y_test)) ** 2).mean())


print('SROCC score for Z1 is = ', np.median(srocc_Z1))
print('RMSE score for Z1 is = ', np.median(rmse_Z1))
print('PCC score for Z1 is = ', np.median(pcc_Z1))
print('KCC score for Z1 is = ', np.median(kcc_Z1))
y_pred_overall = pd.concat([y_pred_overall,pd.DataFrame(y_pred)],axis=1)
y_test_overall = pd.concat([y_test_overall,pd.DataFrame(y_test)],axis=1)




