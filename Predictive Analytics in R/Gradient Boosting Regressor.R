require(gbm)
library(MASS)
data(package="MASS")
boston<-Boston
dim(boston)
names(boston)

# set seed
set.seed(101)   # set seed to ensure you always have same random numbers generated
sample = sample.split(boston,SplitRatio = 0.65)
train =subset(boston,sample ==TRUE) # creates a training dataset named train1 with rows which are marked as TRUE
test=subset(boston, sample==FALSE)
rownames(train) <- NULL #reset row index
rownames(test) <- NULL #reset row index

#model
boost.boston = gbm(medv~., data = train, distribution = "gaussian", n.trees = 10000, shrinkage = 0.01, interaction.depth = 4)
summary(boost.boston)


#plot
plot(boost.boston,i="lstat")
plot(boost.boston,i="rm")


#pred
n.trees = seq(from = 100, to = 10000, by = 100)
predmat = predict(boost.boston, newdata = test, n.trees = n.trees)
dim(predmat)

#evaluation
boost.err = with(test, apply( (predmat - medv)^2, 2, mean) )
plot(n.trees, boost.err, pch = 23, ylab = "Mean Squared Error", xlab = "# Trees", main = "Boosting Test Error")
abline(h = min(boost.err), col = "red")
