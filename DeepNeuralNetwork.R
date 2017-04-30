# Import the log4r package.
library('log4r')
# Import Deep learning packages
library(reticulate)
reticulate::py_available()
reticulate::use_python("/Users/sdeshmukh1/anaconda/bin/python")
reticulate::py_config()
reticulate::import("keras.models")
library(kerasR)
library(ROCR)

dataset_dir = "/Users/sdeshmukh1/Desktop/Kshitija/Keras_assignment/datasets_v1/"
general_dir = "/Users/sdeshmukh1/Desktop/Kshitija/Keras_assignment/Submission/Keras_Deep_Learner/"

# Create a new logger object with create.logger().
logger <- create.logger()
# Set the logger's file output: currently only allows flat files.
logfile(logger) <- file.path(paste0(general_dir, 'base.log', collapse = ""))
# Set the current level of the logger.
level(logger) <- "INFO"

set.seed(1)
dataset_list = list.files(path = dataset_dir)

group_relu = dataset_list[c(11,31,34)]
group_softsign = dataset_list[c(16,17,25,32,49)]
group_tanh = dataset_list[setdiff(seq(1,52),c(11,16,17,25,31,32,34,49))]
extra_layer = dataset_list[c(10,18,19,24,31,33,34)]

act_per_grp = data.frame("dataset" = dataset_list, "activation" = c(""),  "final" = c(""), stringsAsFactors=F)
act_per_grp = within(act_per_grp, activation[which(dataset %in% group_relu)] <- 'relu')
act_per_grp = within(act_per_grp, activation[which(dataset %in% group_softsign)] <- 'softsign')
act_per_grp = within(act_per_grp, activation[which(dataset %in% group_tanh)] <- 'tanh')

accuracy_metric = data.frame()
failed = NULL
for(dataset in dataset_list[24]){
  train = read.csv(file = paste0(dataset_dir, dataset, "/train0.csv", collapse = ""))
  test = read.csv(file = paste0(dataset_dir, dataset, "/test0.csv", collapse = ""))
  
  X_train = train[, 1:(ncol(train)-1)]
  X_test = test[, 1:(ncol(test)-1)]
  
  Y_train = train[, ncol(train)]
  Y_test = test[, ncol(test)]
  
  X_train = data.matrix(X_train)
  X_test = data.matrix(X_test)
  
  ################# Neural Network #############
  # create model
  layers = 6
  mod = Sequential()
  
  unit = nrow(X_train)/layers
  mod$add(Dense(unit, input_shape = dim(X_train)[2]))
  mod$add(BatchNormalization())
  mod$add(Activation(act_per_grp[dataset == act_per_grp$dataset, "activation"]))
  
  unit = max(c(unit/2,1)) 
  mod$add(Dense(unit))
  mod$add(BatchNormalization())
  mod$add(Activation(act_per_grp[dataset == act_per_grp$dataset, "activation"]))
  
  mod$add(Dense(unit))
  mod$add(BatchNormalization())
  mod$add(Activation(act_per_grp[dataset == act_per_grp$dataset, "activation"]))
  
  mod$add(BatchNormalization())
  mod$add(Dense(unit))
  mod$add(Activation(act_per_grp[dataset == act_per_grp$dataset, "activation"]))
  
  unit = max(c(unit/2,1)) 
  mod$add(Dense(unit))
  mod$add(BatchNormalization())
  mod$add(Activation(act_per_grp[dataset == act_per_grp$dataset, "activation"]))
  
  if(dataset %in% extra_layer){
    mod$add(Dense(unit))
    mod$add(BatchNormalization())
    mod$add(Activation(act_per_grp[dataset == act_per_grp$dataset, "activation"]))
  }

  mod$add(Dense(1))
  mod$add(Activation('sigmoid'))
  
  # Compile model
  keras_compile(mod, loss='binary_crossentropy', optimizer=Adam(), metrics='binary_accuracy')
  #callbacks
  callbacks <- list(CSVLogger(paste0(general_dir, "tmp.csv", collapse = "")),
                    EarlyStopping(patience = 5 ),
                    ReduceLROnPlateau()
  )
  # Fit the model
  result = try(keras_fit(mod, X_train, Y_train,  batch_size = 20, epochs = 200, callbacks =  callbacks, verbose = 1,  validation_split = 0.1), silent = TRUE)
  if(class(result) == 'try-error'){
    error(logger, paste('Error occured while processing dataset: ', dataset))
    failed = c(failed, dataset)
    accuracy_metric = rbind(accuracy_metric,c(0,1,0,1))
    next
  }
  #predict
  pred_nn  = try(keras_predict(mod, X_test, batch_size = 32, verbose = 1), silent = TRUE)
  if(class(pred_nn) == 'try-error'){
    error(logger, paste('Error occured while processing dataset: ', dataset))
    failed = c(failed, dataset)
    accuracy_metric = rbind(accuracy_metric,c(0,1,0,1))
    next
  }
  pr = prediction(pred_nn, Y_test)
  prf = performance(pr, measure = 'tpr', x.measure = 'fpr')
  plot(prf, main= paste("ROC for ", dataset))
  #calculate accuracy of the model
  tmp_file = read.csv(file = paste0(general_dir, "tmp.csv", collapse = ""))
  info(logger, paste('Accuracy for dataset ', dataset, "is ", tmp_file$binary_accuracy[nrow(tmp_file)]))
  
  accuracy_metric = rbind(accuracy_metric,c(tmp_file$loss[nrow(tmp_file)],tmp_file$binary_accuracy[nrow(tmp_file)],tmp_file$val_loss[nrow(tmp_file)],tmp_file$val_binary_accuracy[nrow(tmp_file)]))
}

accuracy_metric$dataset = dataset_list[24]
colnames(accuracy_metric) <- c("loss", "binary_accuracy", "val_loss", "val_binary_accuracy", "dataset")
#save accuracy metric
write.csv(x = accuracy_metric, file = paste0(general_dir, "accuracy_metric.csv", collapse = ""))
