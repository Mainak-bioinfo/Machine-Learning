require(devtools)
install_github("WandeRum/multiROC")
require(multiROC)


# Libraries
library(keras)
library(mlbench) 
library(dplyr)
library(magrittr)

#data
data("iris")
data <- iris
str(data)



#partition
# Partition
set.seed(1234)
total_number <- nrow(data)
train_idx <- sample(2, nrow(data), replace = T, prob = c(0.7, 0.3))
train_df <- data[train_idx==1,1:5]
test_df <- data[train_idx==2,1:5]

train_df_depp <- data[train_idx==1,1:4] #split for deeplearning
test_df_depp <- data[train_idx==2,1:4 ]#split for deeplearning
train_df_depp_target <- data[train_idx==1,5]#split for deeplearning
test_df_depp_target <- data[train_idx==2,5]#split for deeplearning

#RANDOMFOREST
rf_res <- randomForest::randomForest(Species~., data = train_df, ntree = 100)
rf_pred <- predict(rf_res, test_df, type = 'prob') 
rf_pred <- data.frame(rf_pred)
colnames(rf_pred) <- paste(colnames(rf_pred), "_pred_RF")


#MULTINOMAIL LR
mn_res <- nnet::multinom(Species ~., data = train_df)
mn_pred <- predict(mn_res, test_df, type = 'prob')
mn_pred <- data.frame(mn_pred)
colnames(mn_pred) <- paste(colnames(mn_pred), "_pred_MN")

#SVM
svm_res <- e1071::svm(Species ~., data = train_df,kernel="linear")
svm_pred <- predict(mn_res, test_df, type = 'prob')
svm_pred <- data.frame(svm_pred)
colnames(svm_pred) <- paste(colnames(svm_pred), "_pred_SVM")

#deeplearning
#matrix

train_df_depp <- as.matrix(train_df_depp)
dimnames(train_df_depp) <- NULL
str(train_df_depp)
test_df_depp <- as.matrix(test_df_depp)
dimnames(test_df_depp) <- NULL
str(test_df_depp)


#hot encoding
train_Labels<-model.matrix(~0+train_df_depp_target)
attr(train_Labels, "dimnames")[[2]] <- levels(iris$Species) #rename the column
test_Labels<-model.matrix(~0+test_df_depp_target)
attr(test_Labels, "dimnames")[[2]] <- levels(iris$Species) #rename the column

# Create Model
model <- keras_model_sequential()
model %>% 
  layer_dense(units = 8, activation = 'relu', input_shape = c(4)) %>%
  layer_dense(units = 3,activation = "softmax")

# Compile

model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = list('accuracy')
)

# Fit Model
mymodel <- model %>%
  fit(train_df_depp,
      train_Labels,
      epochs = 50,
      batch_size = 32,
      validation_split = 0.2)
plot(mymodel)

# Evaluate
model %>% evaluate(test_df_depp,test_Labels)
deep_pred <- model %>% predict(test_df_depp,type = 'prob')
colnames(deep_pred)<-paste(colnames(test_Labels),"_pred_Deeplearn")

# Prediction & confusion matrix - test data
prob <- model %>%
  predict_proba(test_df_depp)

pred <- model %>%
  predict_classes(test_df_depp)
table1 <- table(Predicted = pred, Actual = test_df_depp_target)

cbind(prob, pred, test_df_depp_target)


#merge true label and pred label
true_label <- dummies::dummy(test_df$Species, sep = ".")
true_label <- data.frame(true_label)
colnames(true_label) <- gsub(".*?\\.", "", colnames(true_label))
colnames(true_label) <- paste(colnames(true_label), "_true")
final_df <- cbind(true_label, rf_pred, mn_pred,svm_pred,deep_pred)

#multi roc and multi pr
roc_res <- multi_roc(final_df, force_diag=T)
pr_res <- multi_pr(final_df, force_diag=T)


#plot
plot_roc_df <- plot_roc_data(roc_res)
plot_pr_df <- plot_pr_data(pr_res)

require(ggplot2)
ggplot(plot_roc_df, aes(x = 1-Specificity, y=Sensitivity)) +
  geom_path(aes(color = Group, linetype=Method), size=1.5) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), 
               colour='grey', linetype = 'dotdash') +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), 
        legend.justification=c(1, 0), legend.position=c(.95, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=0.5, 
                                         linetype="solid", colour ="black"))

ggplot(plot_pr_df, aes(x=Recall, y=Precision)) + 
  geom_path(aes(color = Group, linetype=Method), size=1.5) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), 
        legend.justification=c(1, 0), legend.position=c(.95, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=0.5, 
                                         linetype="solid", colour ="black"))

save.image(file = "multiroc.RData")
load


#confusion matrix
plotConfusionMatrix <- function(model, norm = "none"){
  
  cm <- confusionMatrix(model, norm = norm)
  
  conf_matrix <- matrix(cm$table, ncol = length(unique(model$trainingData$.outcome)))
  
  nr <- nrow(conf_matrix)
  
  M <- t(conf_matrix)[, nr:1]
  Mv <- as.vector(M)
  colnames(M) <- colnames(cm$table)[nr:1]
  rownames(M) <- colnames(cm$table)
  
  g <- ggplot2::ggplot(reshape2::melt(M), ggplot2::aes_string(x='Var1', y='Var2', fill='value')) + ggplot2::geom_raster() +
    ggplot2::scale_fill_gradient2(low='blue', high='red') + ggplot2::xlab("True") + ggplot2::ylab("Predicted") +
    ggplot2::theme(axis.text.x=ggplot2::element_text(angle=45,hjust=1,vjust=1)) + 
    ggplot2::geom_text(aes(label = round(Mv,2)), vjust = 1)  
  
  return(g)
  
}
