library(ISLR) #sample dataset
library(dplyr) #data manipulation
require(caTools) #train test split

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


precision <- function(x)
{
  #pass a table x
  return((x[1])/(x[1]+x[3]))
}


f1_score <- function(x)
{
  #pass a table x
  P = precision(x)
  
  R = recall(x)
  
  return((2*P*R)/(P+R))
}


data(package="ISLR")

carseats<-Carseats

require(tree)
names(carseats)

# check what value to use to create target binary variable
hist(carseats$Sales)
# High <- ifelse(carseats$Sales<=8, "No", "Yes")

# create binary target variable
carseats <- carseats %>% mutate(High = ifelse(Sales<=8, "No", "Yes") )

# convert target variable to factor
carseats$High <- as.factor(carseats$High)

#fit model
tree.carseats = tree(High ~. -Sales, data=carseats)
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
tree.carseats = tree(High ~. -Sales, data=train)

plot(tree.carseats)
text(tree.carseats, pretty=0)

summary(tree.carseats)


# Predict
tree.pred = predict(tree.carseats, test, type="class")


#Evaluation metric

#confusion matrix
with(test, table(tree.pred, High))

#scores
accuracy(table(tree.pred, test$High))
precision(table(tree.pred, test$High))
recall(table(tree.pred, test$High))
f1_score(table(tree.pred, test$High))


# CV
cv.carseats = cv.tree(tree.carseats, FUN = prune.misclass)
cv.carseats
#plot
plot(cv.carseats) #see where the misclass is low


#CV again using plot learning
prune.carseats = prune.misclass(tree.carseats, best = 12)
plot(prune.carseats)
text(prune.carseats, pretty=0)

#Predict again
tree.pred = predict(prune.carseats, test, type="class")

#Evaluation
#Confusion Matrix
with(test, table(tree.pred, High))

#scores
accuracy(table(tree.pred, test$High))
precision(table(tree.pred, test$High))
recall(table(tree.pred, test$High))
f1_score(table(tree.pred, test$High))

