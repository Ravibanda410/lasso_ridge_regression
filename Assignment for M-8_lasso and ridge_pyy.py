# Multilinear Regression with Regularization using L1 and L2 norm
## sol for 1st question
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

startup = pd.read_csv("C:/RAVI/Data science/Assignments/Module 8 Lasso-Ridge/Datasets/50_Startups.csv")
startup.head()
startup.columns

dummys = pd.get_dummies(startup['State'])
startup = pd.concat([startup, dummys], axis = 1)
startup= startup.drop(["State"],axis=1)
startup.columns = ["RD", "admin", "market", "profit", "california", "florida", "newyork"]

##Rearranging the startup dataset
startup = startup.iloc[:, [4, 0, 1, 2, 3, 5, 6]]
startup.columns
startup.head()

startup.columns = ["profit", "RD", "admin", "market", "california", "florida", "newyork"]

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

## Hear alpha is nothing but lambda,we creating parameter's grid
parameters = {'alpha': [1e-15, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1, 0, 4, 7 ,10, 15, 20, 24, 28, 30]}

## Creating the model using lasso regression
lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 12)
lasso_reg.fit(startup.iloc[:, 1:], startup.profit)

## Finding best parameter
lasso_reg.best_params_
## Best parameter = 30
lasso_reg.best_score_
## best score = -9.94814600217921
 
## Predict the Y (hat)
lasso_pred = lasso_reg.predict(startup.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(startup.iloc[:, 1:], startup.profit)
## Adjusted R^2 = 0.9507495695766348

# RMSE
np.sqrt(np.mean((lasso_pred - startup.profit)**2))
## RMSE = 8855.023064068575


# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1, 0, 4, 7 ,10, 15, 20, 24, 28, 30]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 7)
ridge_reg.fit(startup.iloc[:, 1:], startup.profit)

## Finding parameters
ridge_reg.best_params_
### Best parameter = 1e-15
ridge_reg.best_score_
## Best score = 1

## Predict Y(hat)
ridge_pred = ridge_reg.predict(startup.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(startup.iloc[:, 1:], startup.profit)
## Adjusted R^2 = 1

# RMSE
np.sqrt(np.mean((ridge_pred - startup.profit)**2))
## RMSE = 2.7910574051884875e-16


# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1, 0, 4, 7 ,10, 15, 20, 24, 28, 30]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 7)
enet_reg.fit(startup.iloc[:, 1:], startup.profit)

## Finding best parameter
enet_reg.best_params_
##  Best parameter =  0.0001
enet_reg.best_score_
## Beast score = -2.4498702937278105e-07

## pridecting Y (hat)
enet_pred = enet_reg.predict(startup.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(startup.iloc[:, 1:], startup.profit)
## Adjusted R^2 = -1.838040444915859e-07

# RMSE
np.sqrt(np.mean((enet_pred - startup.profit)**2))
## RMSE = 0.00042872373912764136

###\/\/\/\/\/\/\/\/\//\/\\/\/\//\/////\/\/\//\/\/\/\/\/\/\/\/\/#######
######################################################################
#######################################################################

######computer ########
## sol for 1st question
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

computer = pd.read_csv("C:/RAVI/Data science/Assignments/Module 8 Lasso-Ridge/Datasets/Computer_Data.csv")
computer.columns
computer= computer.drop(["Unnamed: 0"],axis=1) ## deliting unnamed column

## Creating dummy veriable
computer = pd.get_dummies(computer, columns=['cd', 'multi', 'premium'])
computer.columns

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
lasso = Lasso()

## Hear alpha is nothing but lambda,we creating parameter's grid
parameters = {'alpha': [1e-21, 1e-19, 1e-17, 1e-25, 1e-15, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1, 0, 4, 7 ,10, 15, 20, 24, 28, 30]}

## Creating the model using lasso regression
lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 15)
lasso_reg.fit(computer.iloc[:, 1:], computer.price)

## Finding best parameter
lasso_reg.best_params_
## Best parameter(alpha) = 1
lasso_reg.best_score_
## best score = 0.7491102413453794
 
## Predict the Y (hat)
lasso_pred = lasso_reg.predict(computer.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(computer.iloc[:, 1:], computer.price)
## Adjusted R^2 = 0.7754809397124085

# RMSE
np.sqrt(np.mean((lasso_pred - computer.price)**2))
## RMSE = 8855.023064068575


# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
ridge = Ridge()

## Hear alpha is nothing but lambda,we creating parameter's grid
parameters = {'alpha': [1e-21, 1e-19, 1e-17, 1e-25, 1e-15, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1, 0, 4, 7 ,10, 15, 20, 24, 28, 30]}

## Creating the model using Ridge regression
ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 15)
ridge_reg.fit(computer.iloc[:, 1:], computer.price)

## Finding parameters
ridge_reg.best_params_
### Best parameter = 30
ridge_reg.best_score_
## Best score = 0.7492482058224219

## Predict Y(hat)
ridge_pred = ridge_reg.predict(computer.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(computer.iloc[:, 1:], computer.price)
## Adjusted R^2 = 0.775509992633267

# RMSE
np.sqrt(np.mean((ridge_pred - computer.price)**2))
## RMSE = 275.16511650338737


# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-21, 1e-19, 1e-17, 1e-25, 1e-15, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1, 0, 4, 7 ,10, 15, 20, 24, 28, 30]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 7)
enet_reg.fit(computer.iloc[:, 1:], computer.price)

## Finding best parameter
enet_reg.best_params_
##  Best parameter =  0.01
enet_reg.best_score_
## Beast score = -91129.64647770918

## pridecting Y (hat)
enet_pred = enet_reg.predict(computer.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(computer.iloc[:, 1:], computer.price)
## Adjusted R^2 = -75717.71198948809

# RMSE
np.sqrt(np.mean((enet_pred - computer.price)**2))
## RMSE = 275.16851562176976

######################################################################################
######################################################################################
##########################################################################################
## sol for 3rd question Toyota

Toyota = pd.read_csv("C:/RAVI/Data science/Assignments/Module 8 Lasso-Ridge/Datasets/ToyotaCorolla.csv")
Toyota.describe
Toyota = Toyota.drop(["Model", "Fuel_Type", "Id", "Color", "Mfg_Month", "Mfg_Year", "Met_Color", "Color", "Automatic", "Cylinders", "Mfr_Guarantee", "BOVAG_Guarantee", "Guarantee_Period", "ABS", "Airbag_1", "Airbag_2", "Airco", "Automatic_airco", "Boardcomputer", "CD_Player", "Central_Lock", "Powered_Windows", "Power_Steering", "Radio", "Mistlamps", "Sport_Model", "Backseat_Divider", "Metallic_Rim", "Radio_cassette", "Tow_Bar"], axis=1)

Toyota.columns = ["price", "age", "km", "hp", "cc", "doors", "gears", "Q_tax", "weight"]

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()
## Hear alpha is nothing but lambda,we creating parameter's grid
parameters = {'alpha': [1e-15, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1, 0, 4, 7 ,10, 15, 20, 24, 28, 30]}

## Creating the model using lasso regression
lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 12)
lasso_reg.fit(Toyota.iloc[:, 1:], Toyota.price)

## finding best parameter(least error)
lasso_reg.best_params_
lasso_reg.best_score_

## Predicting Y(hat)
lasso_pred = lasso_reg.predict(Toyota.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(Toyota.iloc[:, 1:], Toyota.price)
## we got best R^2 = 0.8637

# RMSE
np.sqrt(np.mean((lasso_pred - Toyota.price)**2))
## RMSE = 1338.4354

# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

## Hear alpha is nothing but lambda,we creating parameter's grid
parameters = {'alpha': [1e-15, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1, 0, 4, 7 ,10, 15, 20, 24, 28, 30]}

## Creating the model using Ridge regression for Toyota dataset
ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(Toyota.iloc[:, 1:],Toyota.price)

## finding best parameter(least error)
ridge_reg.best_params_
## Best parameter (Alpha = 1e-15)

ridge_reg.best_score_
## Best score 0.03184293294435396

## predicting Y(hat)
ridge_pred = ridge_reg.predict(Toyota.iloc[:, 1:])


# Adjusted r-square#
ridge_reg.score(Toyota.iloc[:, 1:], Toyota.price)
## R^2 = 0.8636291221805601

# RMSE
np.sqrt(np.mean((ridge_pred - Toyota.price)**2))
## RMSE = 1338.914557786828


# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

## Hear alpha is nothing but lambda,we creating parameter's grid
parameters = {'alpha': [1e-15, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1, 0, 4, 7 ,10, 15, 20, 24, 28, 30]}

## creating model using ElasticNet Regression
enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(Toyota.iloc[:, 1:], Toyota.price)

## finding best parameter(least error)
enet_reg.best_params_
## alpha: 4

enet_reg.best_score_
## Best score = -3689253.5806358187

## predicting Y(hat)
enet_pred = enet_reg.predict(Toyota.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(Toyota.iloc[:, 1:], Toyota.price)
##-1802618.1871171794

# RMSE
np.sqrt(np.mean((enet_pred - Toyota.price)**2))
## RMSE = 1342.6161726707958
