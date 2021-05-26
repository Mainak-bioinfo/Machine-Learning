#require(devtools)
#install_github("WandeRum/multiROC")
# Libraries
library(keras)
library(e1071)
library(randomForest)
library(mlbench) 
library(dplyr)
library(magrittr)
library(caret)
library(kernlab)
library(multiROC)
library(ggplot2)  
library(gridExtra)   
library(grid)


#data
data("iris")
data <- iris
str(data)

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
rf_pred_conf <- predict(rf_res, test_df) ## for confusion matrix
rf_pred <- data.frame(rf_pred)
colnames(rf_pred) <- paste(colnames(rf_pred), "_pred_RF")

confusionMatrix(test_df$Species,rf_pred_conf)


#SVM
svm_res <- e1071::svm(Species~., data = train_df,kernel="linear",probability=TRUE)
svm_pred <- predict(svm_res, test_df, type="prob",probability=TRUE)
svm_pred<-attr(svm_pred, "probabilities")
svm_pred_conf <- predict(svm_res,test_df) ## for confusion matrix
svm_pred <- data.frame(svm_pred)
colnames(svm_pred) <- paste(colnames(svm_pred), "_pred_SVM")

confusionMatrix(test_df$Species,svm_pred_conf)

#NB
nb_res <- e1071::naiveBayes(Species~., data = train_df)
nb_pred <- predict(nb_res, test_df, type='raw')
nb_pred_conf <- predict(nb_res,test_df) ## for confusion matrix
nb_pred <- data.frame(nb_pred)
rownames(nb_pred)<-rownames(svm_pred)
colnames(nb_pred) <- paste(colnames(nb_pred), "_pred_NB")

confusionMatrix(test_df$Species,nb_pred_conf)

#deeplearning
#matrix

train_df_depp <- as.matrix(train_df_depp)
dimnames(train_df_depp) <- NULL
str(train_df_depp)
test_df_depp <- as.matrix(test_df_depp)
dimnames(test_df_depp) <- NULL
str(test_df_depp)


#hot encoding
train_Labels<-model.matrix(~0+train_df_depp_target) # binary convesion of all the labels
attr(train_Labels, "dimnames")[[2]] <- levels(iris$Species) #rename the column
test_Labels<-model.matrix(~0+test_df_depp_target) # binary convesion of all the labels
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
      epochs = 10,
      batch_size = 32,
      validation_split = 0.2)
plot(mymodel)

# Evaluate
model %>% evaluate(test_df_depp,test_Labels)
deep_pred <- model %>% predict(test_df_depp,type = 'prob')

deep_pred <- format(round(deep_pred, 2), nsamll = 4)
deep_pred_conf <- data.frame( "setosa"=deep_pred[,1], "versicolor"=deep_pred[,2], "virginica"=deep_pred[,3],
                              "predicted" = ifelse(max.col(deep_pred[ ,1:3])==1, "setosa",
                                                   ifelse(max.col(deep_pred[ ,1:3])==2, "versicolor","virginica")))

deep_pred_conf<-cbind(deep_pred_conf,test_df_depp_target)

deep_pred_conf <- predict(deep_pred_conf, test_df_depp_target) ## for confusion matrix
colnames(deep_pred)<-paste(colnames(test_Labels),"_pred_Deeplearn")


#preparing for multi ROC curves 
#merging true label and pred label
true_label <- dummies::dummy(test_df$Species, sep = ".")
true_label <- data.frame(true_label)
colnames(true_label) <- gsub(".*?\\.", "", colnames(true_label))
colnames(true_label) <- paste(colnames(true_label), "_true")
final_df <- cbind(true_label,rf_pred,nb_pred,svm_pred,deep_pred)


head(final_df)
#multi roc
roc_res <- multi_roc(final_df, force_diag=T)

#plot
plot_roc_df <- plot_roc_data(roc_res)
plot_roc_df<-plot_roc_df%>%
  filter(!Group %in% c("Macro","Micro")) ## remove no needed rows

ggplot(plot_roc_df, aes(x = 1-Specificity, y=Sensitivity)) +
  geom_path(aes(color = Group, linetype=Method), size=1) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), 
               colour='grey', linetype = 'dotdash') +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), 
        legend.justification=c(1, 0), legend.position=c(.95, .05),
        legend.title=element_blank(), 
        legend.background = element_rect(fill=NULL, size=0.5, 
                                         linetype="solid", colour ="black")) #visualization of comparative analysis of algorithm and also label based on ROC


###############CONFUSION MATRIX##################################       
#creating a confusion matrix
#randomforest start#
confusion_rf<-confusionMatrix(test_df$Species,rf_pred_conf)

# construct confusion matrix as data frame
confusion_rf_d <- as.data.frame(confusion_rf$table)

# confusion matrix statistics as data.frame and round up
confusion_rf_st <-data.frame(round(confusion_rf$overall,2))

# confusion matrix plotting is ON

confusion_rf_d_p <-  ggplot(data = confusion_rf_d, aes(x = Prediction , y =  Reference, fill = Freq))+
  geom_tile() +
  geom_text(aes(label = paste("",Freq)), color = 'black', size = 6) +
  theme_light() +
  guides(fill=FALSE)+scale_fill_gradient(low="white",high = "blue")

# plotting the stats
confusion_rf_st_p <-  tableGrob(confusion_rf_st)

# add confusion matrix plotting and stats together
grid.arrange(confusion_rf_d_p, confusion_rf_st_p,nrow = 1, ncol = 2, 
             top=textGrob("Confusion Matrix and stats for Randomforest",gp=gpar(fontsize=12,font=1)))
#randomforest end#


#svm start#

confusion_svm<-confusionMatrix(test_df$Species,svm_pred_conf)

# construct confusion matrix as data frame
confusion_svm_d <- as.data.frame(confusion_svm$table)

# confusion matrix statistics as data.frame and round up
confusion_svm_st <-data.frame(round(confusion_svm$overall,2))

# confusion matrix plotting is ON

confusion_svm_d_p <-  ggplot(data = confusion_svm_d, aes(x = Prediction , y =  Reference, fill = Freq))+
  geom_tile() +
  geom_text(aes(label = paste("",Freq)), color = 'black', size = 6) +
  theme_light() +
  guides(fill=FALSE)+scale_fill_gradient(low="white",high = "blue")

# plotting the stats
confusion_svm_st_p <-  tableGrob(confusion_svm_st)

# add confusion matrix plotting and stats together
grid.arrange(confusion_svm_d_p, confusion_svm_st_p,nrow = 1, ncol = 2, 
             top=textGrob("Confusion Matrix and stats for SVM",gp=gpar(fontsize=12,font=1)))
#svm end#

#naive bayes start#

confusion_nb<-confusionMatrix(test_df$Species,nb_pred_conf)

# construct confusion matrix as data frame
confusion_nb_d <- as.data.frame(confusion_nb$table)

# confusion matrix statistics as data.frame and round up
confusion_nb_st <-data.frame(round(confusion_nb$overall,2))

# confusion matrix plotting is ON
confusion_nb_d_p <-  ggplot(data = confusion_nb_d, aes(x = Prediction , y =  Reference, fill = Freq))+
  geom_tile() +
  geom_text(aes(label = paste("",Freq)), color = 'black', size = 6) +
  theme_light() +
  guides(fill=FALSE)+scale_fill_gradient(low="white",high = "blue")

# plotting the stats
confusion_nb_st_p <-  tableGrob(confusion_nb_st)

# add confusion matrix plotting and stats together
grid.arrange(confusion_nb_d_p, confusion_nb_st_p,nrow = 1, ncol = 2, 
             top=textGrob("Confusion Matrix and stats for Randomforest",gp=gpar(fontsize=12,font=1)))

#naives bayes end#

#deep learning start#

confusion_deep<-confusionMatrix(deep_pred_conf$predicted, deep_pred_conf$test_df_depp_target)

# construct confusion matrix as data frame
confusion_deep_d <- as.data.frame(confusion_deep$table)

# confusion matrix statistics as data.frame and round up
confusion_deep_st <-data.frame(round(confusion_deep$overall,2))

# confusion matrix plotting is ON
confusion_deep_d_p <-  ggplot(data = confusion_deep_d, aes(x = Prediction , y =  Reference, fill = Freq))+
  geom_tile() +
  geom_text(aes(label = paste("",Freq)), color = 'black', size = 6) +
  theme_light() +
  guides(fill=FALSE)+scale_fill_gradient(low="white",high = "blue")

# plotting the stats
confusion_deep_st_p <-  tableGrob(confusion_deep_st)

# add confusion matrix plotting and stats together
grid.arrange(confusion_deep_d_p, confusion_deep_st_p,nrow = 1, ncol = 2, 
             top=textGrob("Confusion Matrix and stats for Deep",gp=gpar(fontsize=12,font=1)))
#deep learning end#
