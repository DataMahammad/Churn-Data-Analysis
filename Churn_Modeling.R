library(tidyverse)
library(scorecard)
library(data.table)
library(skimr)
library(inspectdf)
library(caret)
library(glue)
library(highcharter)
library(h2o)
library(dplyr)


dataset <- fread("Churn_Modelling.csv")

dataset %>% glimpse()


dataset %>% inspect_na()

dataset$Exited <- dataset$Exited %>% as.character()
dataset$RowNumber <- dataset$RowNumber %>% as.character()
dataset$HasCrCard <- dataset$HasCrCard %>% as.character()
dataset$IsActiveMember <- dataset$IsActiveMember %>% as.character()
dataset$CustomerId <- dataset$CustomerId %>% as.character()


dataset %>% glimpse()

df.num <- dataset %>% select_if(is.numeric)
df.chr <- dataset %>% mutate_if(is.character,as.factor) %>% 
          select_if(is.factor)

df.chr %>% glimpse()

#Outliers
for_vars <- c()

num_vars <- df.num %>% names()

df.num <- df.num %>% as.data.frame()

num_vars %>% length()

for (i in 1:length(num_vars) ){
  Outliers <- boxplot(df.num[[num_vars[i]]],plot=F)$out
  if(length(Outliers)>0){
    for_vars[i] <- num_vars[i]
  }  

}
for_vars <- for_vars %>% as.data.frame() %>% drop_na() %>% pull(.)

for_vars %>% length() 

df.num <- df.num %>% as.matrix()


for (i in for_vars) {
  Outliers <- boxplot(df.num[,i],plot = F)$out
  mean <- mean(df.num[,i],na.rm =T)
  
  o3 <- ifelse(Outliers>mean,Outliers,NA) %>% as.data.frame() %>%  drop_na() %>% pull(.)
  o1 <- ifelse(Outliers<mean,Outliers,NA) %>% as.data.frame() %>%  drop_na() %>% pull(.)
  
  val3 <- quantile(df.num[,i],0.75,na.rm = T) + 1.5*IQR(df.num[,i],na.rm=T)
  val1 <- quantile(df.num[,i],0.25,na.rm = T) - 1.5*IQR(df.num[,i],na.rm=T)
  
  df.num[df.num[,i] %in% o3,i] <- val3
  df.num[df.num[,i] %in% o1,i] <- val1
}

boxplot(df.num[,"CreditScore"])
boxplot(df.num[,"Age"])
boxplot(df.num[,"NumOfProducts"])
boxplot(df.num[,"Tenure"])
boxplot(df.num[,"Balance"])
boxplot(df.num[,"EstimatedSalary"])


#One Hote Encoding
df.chr <- df.chr %>% select(Exited,everything())
df.chr <- df.chr %>% select(-Surname) #Bir işə də yaramır :D
names(df.chr)
df.chr %>% glimpse()


ohe <- dummyVars(" ~ .", data = df.chr[,c("Geography","Gender")]) %>% 
  predict(newdata = df.chr[,c("Geography","Gender")]) %>% 
  as.data.frame()


df <- cbind(df.chr[-1],ohe,df.num)

df %>% dim()

#---------------------------------------------MODELING---------------------------------
#INFO VALUE
iv <- df %>% iv(y='Exited') %>% mutate(info_value=round(info_value,3)) %>% 
  arrange(desc(info_value))

ivars <- iv %>% filter(info_value>0.02) %>% select(variable) %>% pull(.)


df.iv <- df %>% select(Exited,ivars)

names(df.iv)

#WOE BINNING

bins <- df.iv %>% woebin("Exited")

bins %>% glimpse()

#bins$Age %>% woebin_plot()


df_list <- df.iv %>% 
  split_df("Exited",ratio = 0.8,seed=123)

df_list %>% glimpse()

train_woe <- df_list$train %>% woebin_ply(bins)
test_woe <- df_list$test %>% woebin_ply(bins)

names <- train_woe %>% names() %>% gsub("_woe","",.)                   
names(train_woe) <- names              ; names(test_woe) <- names

#GLM

target <- "Exited"
features <- train_woe %>% select(-Exited) %>% names()

f <- as.formula(paste(target,paste(features,collapse = " + "),sep = " ~ "))
glm <- glm(f,data = train_woe,family = "binomial")

coef_na <- attributes(alias(glm)$Complete)$dimnames[[1]]
features <- features[!features %in% coef_na]

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = train_woe, family = "binomial")

glm %>% vif() #No need to treat VIF


library(rJava)

Sys.setenv(JAVA_HOME= "C:\\Program Files\\Java\\jre1.8.0_271")
Sys.getenv("JAVA_HOME")


h2o.init(nthreads = -1, max_mem_size = '2g', ip = "127.0.0.1", port = 54321)

train_h2o <- train_woe %>% select(target,features) %>% as.h2o()
test_h2o <- test_woe %>% select(target,features) %>% as.h2o()

model <- h2o.glm(
  x = features, y = target, family = "binomial", 
  training_frame = train_h2o, validation_frame = test_h2o,
  nfolds = 10, seed = 123, remove_collinear_columns = T,
  balance_classes = T, lambda = 0, compute_p_values = T)

model@model$coefficients_table %>%
  as.data.frame() %>%
  select(names,p_value) %>%
  mutate(p_value = round(p_value,3)) %>% 
  arrange(desc(p_value))
#No need to treat p-values as all of them are lower than 0.05

model@model$coefficients %>%
  as.data.frame() %>%
  mutate(names = rownames(model@model$coefficients %>% as.data.frame())) %>%
  `colnames<-`(c('coefficients','names')) %>%
  select(names,coefficients)


h2o.varimp(model) %>% as.data.frame() %>% .[.$percentage != 0,] %>%
  select(variable, percentage) %>%
  hchart("pie", hcaes(x = variable, y = percentage)) %>%
  hc_colors(colors = 'orange') %>%
  hc_xAxis(visible=T) %>%
  hc_yAxis(visible=T)

#PREDICTION & CONFUSION MATRICES
pred <- model %>% h2o.predict(newdata = test_h2o) %>% 
  as.data.frame() %>% select(p1,predict)

model %>% h2o.performance(newdata = test_h2o) %>%
  h2o.find_threshold_by_max_metric('f1')

eva <- perf_eva(
  pred = pred %>% pull(p1),
  label = df_list$test$Exited %>% as.character() %>% as.numeric(),
  binomial_metric = c("auc","gini"),
  show_plot = "roc")

eva$confusion_matrix$dat #NNUUUUUUUUUUUUUUUULLLLLLLLL://///////////


# Check overfitting ----
model %>%
  h2o.auc(train = T,
          valid = T,
          xval = T) %>%
  as_tibble() %>%
  round(2) %>%
  mutate(data = c('train','test','cross_val')) %>%
  mutate(gini = 2*value-1) %>%
  select(data,auc=value,gini)
