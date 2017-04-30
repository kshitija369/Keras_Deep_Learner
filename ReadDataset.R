
setwd("/Users/sdeshmukh1/Desktop/Kshitija/Keras_assignment/datasets_v1/")

list_files = list.files(path = ".", full.names = FALSE)
count = 0
for(dataset in list_files[1]){
  count = count + 1
  print(paste0("Preparing Train & Test data for ", dataset))
  #Read actual data 
  x_df = read.table(file = paste0(dataset, "/data", collapse = ""),header = FALSE, sep = " ", fill = TRUE)
  #Read train labels
  labels_df = read.table(file = paste0(dataset, "/random_class.0", collapse = ""),header = FALSE, sep = " ")
  #Read true labels
  labelsT_df = read.table(file = paste0(dataset, "/trueclass", collapse = ""),header = FALSE, sep = " ")
  
  #function to merge train labels and actual data
  mergeXnDF <- function(y_lables, row_num){
    z = c(x_df[row_num + 1, ], y_lables)
    return(z)
  }
  #prepare train data
  final_df = mapply(FUN = mergeXnDF, labels_df$V1, labels_df$V2, SIMPLIFY = TRUE)
  final_df = t(final_df)
  final_df = final_df[, colSums(is.na(final_df)) != nrow(final_df)]
  final_df[is.na(final_df)] <- 0
  
  #write train data
  rownames(final_df) = NULL
  colnames(final_df) = NULL
  write.csv(x = final_df,file = paste0(dataset, "/train0.csv", collapse = ""), row.names = FALSE, col.names = FALSE)
  
  #prepare test data
  test_rows = setdiff(x = seq(from = 0, to = nrow(x_df) - 1), y = labels_df$V2);
  
  #function to merge train labels and actual data
  mergeXnDF_test <- function(test_row){
    z = c(x_df[test_row + 1, ], labelsT_df[which(labelsT_df$V2 == test_row), "V1"])
    return(z)
  }
  #prepare train data
  test_df= sapply(FUN = mergeXnDF_test, test_rows)
  test_df = t(test_df)
  test_df = test_df[, colSums(is.na(test_df)) != nrow(test_df)]
  test_df[is.na(test_df)] <- 0
  
  #write test data
  rownames(test_df) = NULL
  colnames(test_df) = NULL
  write.csv(x = test_df,file = paste0(dataset, "/test0.csv", collapse = ""), row.names = FALSE, col.names = FALSE)
  print(paste0(count, " of ", length(list_files), " completed."))
}


