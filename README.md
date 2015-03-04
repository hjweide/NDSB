# National Data Science Bowl (NDSB)

This is my code for the National Data Science Bowl recently hosted by Kaggle.

## Getting the Data
To run this code, first get the training data from:
https://www.kaggle.com/c/datasciencebowl/download/train.zip

Then get the test data from:
https://www.kaggle.com/c/datasciencebowl/download/test.zip

## Resizing the Images
Now give the path to the train directory and a path to the output directory
to `resize.py`, and it will generate the resized images in the output
directory while maintaining the same directory structure.  Do the same for
the test data if you wish to make a submission that can be submitted to 
Kaggle for evaluation.

## Generating the Training Data
Next, you need to run `gen_data.py` with the directory containing the resized
images as input.  This will generate numpy matrices containing the data, as
well as other files that will be used to generate the final Kaggle submission
file.

## Training a Learning Model
Finally, you can run one of several network files, named as `netd.py`, where d
indicates the number of the network.

## Useful utilities
Some utilities are also provided.  Of these, `ensemble.py` is the most important.
You can give multiple network files to `ensemble.py` and ensemble will run each
predictions using each of the provided networks.  The results are then averaged.
Either the training, validation, or test data may be presented to ensemble.py,
but the expected loss can only be displayed for training or validation.

Another useful utility is `visualize.py`.  You may provide it with a network file,
and it will display the misclassified images in your training set in an
easy-to-interpret plot.  Useful information such as the actual class, predicted
class, predicted probabilities of the actual and predicted classes is displayed.
Additionally, random images from both the actual and the predicted class are
displayed for a visual comparison between the classes.  This helps to provide
context if you are trying to find out why a network is confusing two classes.
