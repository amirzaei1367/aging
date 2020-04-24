# Aging

predicting the age of a circuit when its features are provided in an input file.

### How it works: ###
Just set the path for root directory **root_path**, and set the name of input file **in_file** and a file name for storing statistics of the used model **res_file**, and a file name for storing the predictions **out_file**.  

### Possible Improvement: ###
* Right now my focus was on **StackingCVRegressor**, it can be improved in order to consider only the best regressors.
* Maybe a better preprocessing improves the performance. As is now features have been normalized and zero features have also been removed.

