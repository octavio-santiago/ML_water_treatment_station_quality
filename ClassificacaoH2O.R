library(h2o)
library(plyr)
h2o.init()
library(dplyr)
library(purrr)
library(caret)
library(plotROC)
library(MESS)

mydata <- read.csv("dfPreTratado2.csv", header=TRUE, sep=",")

#remove na
mydata <- mydata[complete.cases(mydata), ]

#combine os datasets
#results <- data.frame("resultado" = ifelse(mydata$Efluente.Final > 250,1,0))
#results <- as.h2o(results)
colnames(mydata)[colnames(mydata)=="Desenquadrou"] <- "resultado"

df <- as.h2o(mydata)
#df <- h2o.cbind(df, results)
# convert response column to a factor
df["resultado"] <- as.factor(df["resultado"])



predictors <- c('Eq.D.1','phAeracao','OD',
                'Efluente.D.1','VLodoSD30','phEntrada.D.1',
                'phNeutralizacao.D.1','phEqualizacao.D.1','phAeracao.D.1','OD.D.1','DiasDescarte'
                )

response <- "resultado"


# split into train and validation sets
mydata.split <- h2o.splitFrame(data = df,ratios = 0.80, seed = 1234)
train <- mydata.split[[1]]
valid <- mydata.split[[2]]

#modelo GLM
df_glm <- h2o.glm(family= "binomial", x= predictors, y=response, training_frame=train,lambda = 0.261)

h2o.confusionMatrix(df_glm)
h2o.performance(df_glm, newdata = valid, train = FALSE, valid = TRUE,
                xval = FALSE)
h2o.scoreHistory(df_glm)
h2o.varimp(df_glm)


pred = predict(df_glm,valid)
finalPrediction <- as.data.frame(h2o.cbind(valid,pred))

# otimização
cutoff <- seq(0.258,0.263,0.0005)
result_valid <- as.data.frame(valid)$resultado
specif_test <- map_dbl(cutoff,function(x){
  y_exp <- as.data.frame(predict(h2o.glm(family= "binomial", x= predictors, y=response, training_frame=train,lambda=x),valid))$predict %>%
    factor(levels = levels(result_valid))
  specificity(y_exp,result_valid)
})
plot(cutoff,specif_test,type="b")
best_cutoff <- cutoff[which.max(specif_test)]
#lambda = 0.322790335222
# lambda = 0.261


#autoML
automl <- h2o.automl(x = predictors, y = response,
                     training_frame = train,
                     max_models = 20,
                     seed = 1)

pred <- predict(automl@leader,valid)
finalPrediction <- as.data.frame(h2o.cbind(valid,pred))

#guessing
df_train <- as.data.frame(train)
df_test <- as.data.frame(valid)
y_guess <- sample(c(0,1),length(df_train$OD),replace=TRUE) %>% factor(levels = levels(df_train$resultado))
mean(y_guess == df_train$resultado)

#exploratory
df_train %>% group_by(resultado) %>% summarize(mean(Eq.D.1),sd(Eq.D.1))
df_train %>% group_by(resultado) %>% summarize(mean(phEntrada),sd(phEntrada))
df_train %>% group_by(resultado) %>% summarize(mean(OD),sd(OD))

y_exp <- ifelse(df_train$phEntrada<6.75-3.05,1,0) %>% factor(levels = levels(df_train$resultado))
mean(y_exp == df_train$resultado)

cutoff <- seq(0.5,8)
accuracy <- map_dbl(cutoff,function(x){
  y_exp <- ifelse(df_train$phEntrada<x,1,0) %>%
    factor(levels = levels(df_train$resultado))
  mean(y_exp == df_train$resultado)
})
plot(cutoff,accuracy,type="b")
best_cutoff <- cutoff[which.max(accuracy)]
best_cutoff

y_exp <- ifelse(df_train$phEntrada<best_cutoff,1,0) %>% factor(levels = levels(df_train$resultado))
mean(y_exp == df_train$resultado)
table(predicted = y_exp,actual=df_train$resultado)

#teste
y_exp <- ifelse(df_test$phEntrada<best_cutoff,1,0) %>% factor(levels = levels(df_test$resultado))
mean(y_exp == df_test$resultado)
table(predicted = y_exp,actual=df_test$resultado)

df_test %>% mutate(y_exp = y_exp) %>% 
  group_by(resultado) %>%
  summarize(accuracy = mean(y_exp == resultado ))

prev <- mean(df_test$resultado == "0")

#confusion matrix
confusionMatrix(data=y_exp,reference=df_test$resultado)

#f1
cutoff <- seq(0.5,8)
F1 <- map_dbl(cutoff,function(x){
  y_exp <- ifelse(df_train$phEntrada<x,1,0) %>%
    factor(levels = levels(df_train$resultado))
  F_meas(data=y_exp,reference=df_train$resultado)
})
plot(cutoff,F1,type="b")
best_cutoff <- cutoff[which.max(F1)]
best_cutoff

#ROC
cutoffs <- c(0,seq(0.5,8),14)
height_cutoff <- map_df(cutoffs,function(x){
  y_exp <- ifelse(df_train$phEntrada<x,1,0) %>%
    factor(levels = levels(df_train$resultado))
  list(method="height cutoff", 
       FPR = 1-specificity(y_exp,df_train$resultado), 
       TPR = sensitivity(y_exp,df_train$resultado))
  
})
TPR = sensitivity(y_exp,df_train$resultado)
FPR = 1-specificity(y_exp,df_train$resultado)
plot(height_cutoff$FPR,height_cutoff$TPR,type="b")
auc(height_cutoff$FPR,height_cutoff$TPR)
