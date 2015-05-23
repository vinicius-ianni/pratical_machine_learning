library(caret)
library(doMC)
set.seed(65874)
registerDoMC(4) # register 4 cores.

train <- read.csv("pml-training.csv")
test <- read.csv("pml-testing.csv")

sanit <- unique(unlist(lapply(train, function(x) which(x == "#DIV/0!"))))
train <- train[-sanit,]
train <- train[,c(-1,-2,-3,-4,-5,-6)]
train <- modifyList(train, lapply(train[,sapply(train[,-154], is.factor)], as.numeric))
train <- train[,sapply(train, function(x)!all(is.na(x)))]
train <- train[,sapply(train, function(x)!all(x==1))]

trainer <- createDataPartition(train$classe, p = 0.7, list = FALSE)
training <- train[trainer,]
testing <- train[-trainer,]
ctrltiny <- trainControl(method = "repeatedcv",repeats = 4, number = 2, allowParallel=TRUE)

knnmd <- train(classe ~., data=training, method='knn', trControl = ctrltiny, preProcess=c("pca"))
plot(knnmd)
confus <- confusionMatrix(predict(knnmd, newdata = training), training$classe)
print(confus$table)
print(confus$overall)
confos <- confusionMatrix(predict(knnmd, newdata = testing), testing$classe)
print(confos)

prev <- predict(knnmd, newdata = test)
print(prev)