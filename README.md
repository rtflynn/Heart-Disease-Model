Currently:  Very simple network.  Input (size 19) --> Layer 1 (size 100) --> Layer 2 (size 100) --> A single neuron ('target') .

After running through the training set 200 times, its performance on the test set is :

specificity :  0.8636363636363636

sensitivity :  0.8205128205128205



(New record:  specificity :  0.85 ,   sensitivity : 0.91,  accuracy : 0.885.  Achieved by using categorical_hinge as loss function)

Interesting observation:  I've looked through a few kaggle kernels, and the best accuracy anyone seems to be able to get is ~88.5%.  There's a concept called 'Bayes error' which captures the idea that there is some error intrinsic to a problem, i.e. if the Bayes error of this problem happened to be 11.5%, then that would mean no classifier will ever do better than 88.5% unless it's overfitting (which is bad).  The fact that so many other classifiers get to 88.5% using random forests, decision trees, SVM, logistic regression, etc, makes me want to conclude that this is as good as we can do here.

A bit more on Bayes error here:  The reason there's this 11.5% unavoidable error has to do with the fact that we only have 19 data categories from which to draw a conclusion.  If we had more, the Bayes error would likely be smaller (and certainly wouldn't be larger).



By the way, I think the data set has a problem.  The 'target' variable is supposed to be 0 if no heart disease, and 1 if heart disease, but I'm relatively (?) certain they reversed these values.  Check out the heat map.  Forget 'target' for a minute and look at the age column.  This looks like we'd expect, for example there's a 0.3 in cholesterol meaning we have a positive correlation between age and cholesterol.  We have -0.4 for max heart rate which again makes sense.

 ![Heatmap](/Images/heatmap.png)

'target' is negatively correlated with age, cholesterol, being male....   But I don't know anything about number of blood vessels, levels of pain, etc.  What do you think?
