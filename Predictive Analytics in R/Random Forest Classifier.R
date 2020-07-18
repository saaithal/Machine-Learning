library(MASS)
library(dplyr)
library(caret) # confusion matrix
require(randomForest)
library(Metrics)
library(caTools)

#Functions
accuracy <- function(x)
{
  #pass a table x
  return((x[1]+x[4])/(x[1]+x[2]+x[3]+x[4]))
}


recall <- function(x)
{
  #pass a table x
  return((x[1])/(x[1]+x[2]))
}



data(package="MASS")
boston<-Boston
dim(boston)
names(boston)

#boxplot
quantile(boston$medv)
boxplot(boston$medv)

hist(boston$medv)
boston <- boston %>% mutate(high = ifelse(medv >= 21.2, "Yes", "No") )
boston$high <- as.factor(boston$high)


# relevel - gives confusion matrix and evaluation metric as per this level
boston$high <- relevel(boston$high, "Yes")


# set seed
set.seed(101)   # set seed to ensure you always have same random numbers generated
sample = sample.split(boston,SplitRatio = 0.65)
train =subset(boston,sample ==TRUE) # creates a training dataset named train1 with rows which are marked as TRUE
test=subset(boston, sample==FALSE)
rownames(train) <- NULL #reset row index
rownames(test) <- NULL #reset row index



#model
rf.boston = randomForest(high~.-medv, data = train,ntree = 700,replace = TRUE, importance = TRUE)
rf.boston # Mean of squared residuals and % Var explained are on OOB samples

# Tune the only parameter - mtry

oob.err = double(13)
test.err = double(13)

# There are 13 variables, so let's have mtry range from 1 to 13
#fix no. of trees to 350

for(mtry in 1:13){
  fit = randomForest(high~.-medv, data =train, mtry=mtry, ntree = 700)
  oob.err[mtry] = accuracy(table(train$high,fit$predicted)) #based on confusion matrix of OOB sample
  pred = predict(fit, test) #predict
  test.err[mtry] = accuracy(table(test$high, pred)) # accuracy based on test dataset
}

#plot
matplot(1:mtry, cbind(test.err, oob.err), pch = 23, col = c("red", "blue"), type = "b", ylab="Accuracy")
legend("right", legend = c("OOB", "Test"), pch = 23, col = c("red", "blue"))


# remodel
rf.boston = randomForest(high~.-medv, data =train, mtry=7, ntree = 700,replace = TRUE, importance = TRUE)
rf.boston # Mean of squared residuals and % Var explained are on OOB samples

#predict
pred = predict(rf.boston, test) #predict
accuracy(table(test$high, pred))  #recall of test

