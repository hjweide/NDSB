# National Data Science Bowl (NDSB)

This is my code for the National Data Science Bowl recently hosted by Kaggle.  My final solution scored 0.755580 on the public leaderboard and 0.759122 on the private leaderboard, giving me an overall finish of 103rd place out of 1049 contestants.

To achieve this score, I created an ensemble of my five best models.  These are included in the repository and are the networks numbered 10, 13, 19, 21, and 24.  A short description of each network is given at the end of the file.

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
Next, you need to run `gen_data.py train` with the directory containing the resized
images set as input in the file.  This will generate numpy matrices containing the data, as
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

## Summary of Networks
* net10 was my first real successful network.  It achieved a validation score of 0.9816 after 215 epochs at 115.8 seconds per epoch.  After test-time augmentation, it achieved a validation score of 0.8412, and a test score of 0.821988 on the public leaderboard.  An interesting point about net10 is that it had very large 7x7 filters in the first convolutional layer.

* net13 was inspired from net10 but uses much smaller convolutional filters.  It achieved a validation score of 0.894 after 148 epochs at 209 seconds per epoch.  After test-time augmentation, its validation score was 0.836.

* net19 was trained on much less aggressive data augmentation and thus converged much faster.  It did not perform as well alone (validation error of 0.844 after test-time augmentation), but brought good variation to the ensemble.

* net21 was simply net13 with fine-tuning.  To do this, I trained net13 with data augmentation disabled and a very low learning rate for a very small number of epochs.  This allowed me to remove the need for test-time augmentation from net21.  Once again, the improvement over net13 was negligible, but it helped to reduce the variance and thereby improve the performance of the ensemble.

* net24 was another variation on net13 using only fixed rotations of 30 degrees during data augmentation, as well as no feature pooling in the lower convolutional layers.  This was the final improvement to the variance required to achieve my best score with the ensemble.
