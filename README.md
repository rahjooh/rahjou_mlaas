ML_Service is a general framework to run Machine Learning Pipelines. Tme main ideas are as follows:
 
 * Using a portable json oriented format to define the pipeline and to store results
 * Using a unified framework for processing data-frames called `coldata` (column based data). A coldata is a dict of arrays each having the same number of rows. Each stage of the pipeline transforms this data (preferably by just adding a new column or changing the order of rows to guarantee immutability).
 
 Stages usually have `input` and `output` column names.
 Curretn stages are as follows:
 * **Impute:** this stage generates a new column named `output` and stores values of `input` column where all the `nan` values are replaced by a suitable value.
 * **Quantize:** this stage quantize a column with float data type to suitable values (possibiably using a target variable and decision tree to find the most informative quantization for the variable)
 * **Onehot:** onehot encoding of a discrete random variable
 * **Sort:** lexical sort of arrays based on the list of `input` columns 
 * **Split:** the data to `k` subsets with given `ratios`. As all the processes should be modeled as adding a new column, this process just adds an output column containing the split number for each row. 
 * **Learner:**  first `fit`s a model on the training set predict `target` from the `input` columns. Then `transform`s and produce the prediction for each row in the `output` column.
 * **CustomAugment:** applies a function given by its fully quaifies name on two or more `input` columns and store the result in `output`.
 This allows to easily use all the `numpy` operators, e.g. `add`, `subtract`, `and` , `or`, `not` and ... 
 * **CustomFunction:** is like the previous one but has not output and just outputs a single result which is stored in the json.
 * **CustomStage** (most general custom processing): applying a function which gets whole coldata and some parameters and change the coldata as it wants. See the `seqproc.freq_in_batch` as an example.

 
 **Note:** The trainig set for all the stages which have a `fit` method can be specified by a boolean column representing the indicator for the training set.


# Summary of Files:

* main.py (main file - not so clean currently)
* schemas.py (object oriented structure of pipeline stages)
* evaluation_functions.py (sample evaluation methods)
* classifiers.py (sample classifiers)
* seqproc.py (some general utils to do sequential processes with high speed)
* Init and Test.ipynb (developement code)


# Basic functionality:
After running the ML_service.py it will try to connect to the MongoDB on the server it is running on and look for a database named 'feature-engineering'.
This databse has all the information needed for the ML_service to execute the jobs. When running, ML_Service always read jobs from this dataset and execute the new ones.
The two important configuration collections in the dataset are '_learning_jobs' and '_settings'

The '_settings' collection contains the basic settings for the service. In partcular it defines how the auto_job_creation works (see the related section for this functionality).
The '_learning_jobs' collection contains the definition of each learning job, along its current status, results and other information regarding that job.



# Simple Use-Cases:
1. Investigating the quality of new features for search_ad (SA): The simplest usage is to create a collection starting with "SA#" which contains a column named 'requestId' and a number of categorical (string) and numerical (number) features. The documents in this collection can be nested (which in this case the name of each field can be accessed by a dot-separated address).
The MLService automatically create learning jobs for such collections which can be seen after a few seconds (if the service is not busy) in the _learning_jobs collection.
The template of this automatically created job can be  configured in the _settings collection in 'auto_job_creation' field.
2. Running a job with custom settings. The prefered way is to use the step 1 to have a basic job definition, then copy the created doc to configure it the way you like and make a new document based on your needs.
3. Adding new classifiers and evaluation functions: the classifiers and evaluation functions  specified in the pipeline are class/functions in the 'path' of the MLService.py with a specific signature. Please see the classifiers.py for sample classifiers and evaluation_functions for sample evaluation metrics. You can also use third party codes installed through pip as long as they match the signature (e.g. scikit classifiers).


# auto_job_creation:
