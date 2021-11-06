ML_Service is a general framework to run Machine Learning Pipelines in a portable json oriented format.

# Summary of Files:

main.py (main file - not so clean currently)
schemas.py (object oriented structure of pipeline stages)
evaluation_functions.py (sample evaluation methods)
classifiers.py (sample classifiers)
seqproc.py (some general utils to do sequential processes with high speed)
Init and Test.ipynb (developement code)


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

# Current Limitations:
1. Single Threaded
2. Not easy to introduce completely custom pipeline tasks
3. MLService.py is requires major factorization as it is developed from a collection of various jupyter codes.