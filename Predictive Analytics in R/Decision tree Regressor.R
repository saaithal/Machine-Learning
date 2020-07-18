library(ISLR) #sample dataset
library(dplyr) #data manipulation
require(caTools) #train test split
library(Metrics)


data(package="ISLR")

carseats<-Carseats

require(tree)
names(carseats)

# check what value to use to create target binary variable
hist(carseats$Sales)


#fit model
tree.carseats = tree(Sales ~., data=carseats)
summary(tree.carseats)

# tree diagram with label
plot(tree.carseats)
text(tree.carseats, pretty = 0)

#detailed summary
tree.carseats


# tree prunning



# TRAIN TEST SPLIT

# set seed
set.seed(123)   # set seed to ensure you always have same random numbers generated
sample = sample.split(carseats,SplitRatio = 0.75)
train =subset(carseats,sample ==TRUE) # creates a training dataset named train1 with rows which are marked as TRUE
test=subset(carseats, sample==FALSE)

rownames(train) <- NULL #reset row index
rownames(test) <- NULL #reset row index


#fit model again
tree.carseats = tree(Sales ~.,data=train)

plot(tree.carseats)
text(tree.carseats, pretty=0)

summary(tree.carseats)


# Predict
tree.pred = predict(tree.carseats, test)


#Evaluation metric


#scores
mape(test$Sales, tree.pred)


# CV
cv.carseats = cv.tree(tree.carseats, FUN = prune.tree)
cv.carseats
#plot
plot(cv.carseats) #see where the misclass is low


#CV again using plot learning
prune.carseats = prune.tree(tree.carseats, best = 10)
plot(prune.carseats)
text(prune.carseats, pretty=0)

#Predict again
tree.pred = predict(prune.carseats, test)

#Evaluation
#scores
mape(test$Sales, tree.pred)

