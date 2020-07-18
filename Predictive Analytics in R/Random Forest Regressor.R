library(MASS)
data(package="MASS")
boston<-Boston
dim(boston)
names(boston)
require(randomForest)

# set seed
set.seed(101)   # set seed to ensure you always have same random numbers generated
sample = sample.split(boston,SplitRatio = 0.65)
train =subset(boston,sample ==TRUE) # creates a training dataset named train1 with rows which are marked as TRUE
test=subset(boston, sample==FALSE)
rownames(train) <- NULL #reset row index
rownames(test) <- NULL #reset row index



#model
rf.boston = randomForest(medv~., data = train)
rf.boston # Mean of squared residuals and % Var explained are on OOB samples

# Tune the only parameter - mtry

oob.err = double(13)
test.err = double(13)

# There are 13 variables, so let's have mtry range from 1 to 13
#fix no. of trees to 350

for(mtry in 1:13){
  fit = randomForest(medv~., data =train, mtry=mtry, ntree = 350)
  oob.err[mtry] = fit$mse[350] #mse of OOB sample
  pred = predict(fit, test) #predict
  test.err[mtry] = with(test, mean( (medv-pred)^2 )) #MSE of test
}

#plot
matplot(1:mtry, cbind(test.err, oob.err), pch = 23, col = c("red", "blue"), type = "b", ylab="Mean Squared Error")
legend("topright", legend = c("OOB", "Test"), pch = 23, col = c("red", "blue"))


# remodel
rf.boston = randomForest(medv~., data = train, mtry = 3)
rf.boston # Mean of squared residuals and % Var explained are on OOB samples

#predict
pred = predict(rf.boston, test) #predict
with(test, mean( (medv-pred)^2 )) #MSE of test

