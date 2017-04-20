
setwd("/Users/sdeshmukh1/Desktop/Kshitija/Keras_assignment/datasets_v1/")

list_files = list.files(path = ".", full.names = FALSE)
count = 0
for(dataset in list_files){
  count = count + 1
  train_file = paste0(dataset, "/train0.csv", collapse = "")
  test_file = paste0(dataset, "/test0.csv", collapse = "")
  
  system(paste('python /Users/sdeshmukh1/Desktop/Kshitija/Keras_assignment/SequentialModel.py', train_file, ' ', test_file))
}