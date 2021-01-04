## Startup ###

library(readr)
startup <- read_csv(choose.files())
View(startup)
startup <- startup[c(5,1,2,3,4)]

colnames(startup) <- c("profit", "RD", "admin", "market", "state")
colnames(startup)
View(startup)
attach(startup)

## creating the array without profit
startup_x <- model.matrix(profit ~ ., data = startup)[,-1]
View(startup_x)

## creating profit vector
startup_y <- profit
View(startup_y)

## Creating Grid with 50 random values
grid_s <- 5^seq(10, -10, length = 50)
View(grid_s)

## LASSO Regression (aLPHA = 1)
install.packages("glmnet")
library(glmnet)

## Building model using lasso
model_ls <- glmnet(startup_x, startup_y, alpha = 1, lambda = grid_s)
summary(model_ls)

## cross validating  the model for finding least error
cv_ls <- cv.glmnet(startup_x, startup_y, alpha = 1, lambda = grid_s)
plot(cv_ls)
leasterror_ls <- cv_ls$lambda.min
leasterror_ls

## Finding R^2 value
pred_ls <- predict(model_ls, s = leasterror_ls, newx = startup_x)
sse_ls <- sum((pred_ls - startup_y)^2)
sst_ls <- sum((startup_y - mean(startup_y))^2)
rsquared <- 1-sse_ls/sst_ls
rsquared

## Finding Residuals
predict(model_ls, s = leasterror_ls, type="coefficients", newx = startup_x)


#########\/\/\//\/\/\/\//\/////\\/\/\/##########

## RIDGE REgression
# Creating model using Ridge Regression (Alpha = 0)

library(glmnet)

## Creating model using Ridge reg
model_rs <- glmnet(startup_x, startup_y, alpha = 0, lambda = grid_s)
summary(model_rs)

## Cross validating the Ridge model for finding least error
cv_rs <- cv.glmnet(startup_x, startup_y, alpha = 0, lambda = grid_s)
summary(cv_rs)
plot(cv_rs)
leasterror_rs <- cv_rs$lambda.min
leasterror_rs

## predicting the values for finding R^2
pred_rs <- predict(model_rs, s = leasterror_rs, newx = startup_x)

## finding R^2
sse_r <- sum((pred_rs-startup_y)^2)
sst_r <- sum((startup_y - mean(startup_y))^2)
rsquared <- 1-sse_r/sst_r
rsquared

## Finding coefficients
predict(model_rs, s = leasterror_rs, type="coefficients", newx = startup_x)


############################################################################################
##################################################################################################
#########################################################################################
#computer

# Load the Data set - computer.csv
library(readr)
computer <- read.csv(choose.files())
View(computer)

## removing X veriable
computer <- computer[,-1]
View(computer)

install.packages("glmnet")
library(glmnet)

## Creating the array without y
computer_x <- model.matrix(price ~ ., data = computer)[,-1]
View(computer_x)

## Creating the vectors with y
computer_y <- computer$price
View(computer_y)

## Creating the grid with 70 random values
computer_g <- 12^seq(10, -5, length = 100)
computer_g


## Lasso Regression (lasso = 1)
## creating the model using lasso
model_lc <- glmnet(computer_x, computer_y, alpha = 1, lambda = computer_g)
summary(model_lc)

## crossvalidating the model for finding least error
cv_lc <- cv.glmnet(computer_x, computer_y, alpha = 1, lambda = computer_g)
summary(cv_lc)
plot(cv_lc)
leasterror_lc <- cv_lc$lambda.min
leasterror_lc

## Finding R^2 value
pred_lc <- predict(model_lc, s = leasterror_lc, newx = computer_x)
sse_lc <- sum((pred_lc - computer_y)^2)
sst_lc <- sum((computer_y - mean(computer_y))^2)
rsquared <- 1-sse_lc/sst_lc
rsquared

## Finding Residuals
predict(model_lc, s = leasterror_lc, type="coefficients", newx = computer_x)


### Ridge Regression ####
model_rc <- glmnet(computer_x, computer_y, alpha = 0, lambda = computer_g)
summary(model_ridge)

##cross validation for finding least error
cv_rc <- cv.glmnet(computer_x, computer_y, alpha = 0, lambda = computer_g)
plot(cv_rc)
leasterror_rc <- cv_rc$lambda.min
leasterror_rc

## Predict for finding R^2
pred_rc <- predict(model_rc, s = leasterror_rc, newx = computer_x)
sse_rc <- sum((pred_rc - computer_y)^2)
sst_rc <- sum((computer_y - mean(computer_y))^2)
rsquared <- 1-sse_rc/sst_rc
rsquared

## Finding coefficients
predict(model_rc, s = leasterror_rc, type="coefficients", newx = x)


############################################################################################
##################################################################################################
#########################################################################################
###Tayoto


library(readr)
Tayoto <- read.csv(choose.files())
Tayoto <- Tayoto[, c('Price','Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight')]
colnames(Tayoto) <- c("price", "age", "km", "hp", "cc", "doors", "gears", "Q_tax", "weight")
View(Tayoto)

## Creating array withot price
Tayoto_x <- model.matrix(price ~ ., data = Tayoto)[,-1]
View(Tayoto_x)

## Creating the vector for price
Tayoto_y <- Tayoto$price
View(Tayoto_y)

## Creating the grid using 150 of random values
grid_T <- 10^seq(10, -5, length = 150)
grid_T

## Lasso Regression (aLPHA = 1)
install.packages("glmnet")
library(glmnet)

## Creating the model using lasso regression
model_lasso_T <- glmnet(Tayoto_x, Tayoto_y, alpha = 1, lambda = grid_T)
summary(model_lasso_T)

## cross validation the model for finding the leasterror
cv_fit_T <- cv.glmnet(Tayoto_x, Tayoto_y, alpha = 1, lambda = grid_T)
plot(cv_fit_T)
leasterror <- cv_fit_T$lambda.min
leasterror

## PREDICTING y(Hat)
yH_T <- predict(model_lasso_T, s = leasterror, newx = Tayoto_x)
yH_T
## Finding R^2 value
sse_l <- sum((yH_T - Tayoto_y)^2)
sst_l <- sum((Tayoto_y - mean(Tayoto_y))^2)
rsquar <- 1-sse_l/sst_l
rsquar

## Finding Coefficiants
Tayota_cl <- predict(model_lasso_T, s = leasterror, type="coefficients", newx = Tayoto_x)
Tayota_cl


#### RIDGE Regression (Alpha = 0)

## Creating the model REDGE regression
library(glmnet)
model_r_Tayot <- glmnet(Tayoto_x, Tayoto_y, alpha = 0, lambda = grid_T)
summary(model_r_Tayot)

## Cross validation the Redge nmodel
cv_r_Toyota <- cv.glmnet(Tayoto_x, Tayoto_y, alpha = 0, lambda = grid_T)
summary(cv_r_Toyota)
plot(cv_r_Toyota)

## Least error
leasterror_r <- cv_r_Toyota$lambda.min
leasterror_r

## Predicting y(hat)
y_h_r <- predict(model_r_Tayot, s = leasterror_r, newx = Tayoto_x)
y_h_r

## Finding R^2
sse_r <- sum((y_h_r - Tayoto_y)^2)
sst_r <- sum((Tayoto_y-mean(Tayoto_y))^2)
rsquar_t <- 1-sse_r/sst_r
rsquar_t

## Finding Coefficiants
Toyot_cr <- predict(model_r_Tayot, s = leasterror_r, type = "coefficients", newx = Tayoto_x)
Toyot_cr
