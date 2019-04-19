Currently:  Very simple network.  Input (size 19) --> Layer 1 (size 100) --> Layer 2 (size 100) --> A single neuron ('target') .

After running through the training set 200 times, its performance on the test set is :

specificity :  0.8636363636363636
sensitivity :  0.8205128205128205



By the way, I think the data set has a problem.  The 'target' variable is supposed to be 0 if no heart disease, and 1 if heart disease, but I'm relatively (?) certain they reversed these values.  Check out the heat map.  Forget 'target' for a minute and look at the age column.  This looks like we'd expect, for example there's a 0.3 in cholesterol meaning we have a positive correlation between age and cholesterol.  We have -0.4 for max heart rate which again makes sense.

 ![Heatmap](/Images/heatmap.png)

'target' is negatively correlated with age, cholesterol, being male....   But I don't know anything about number of blood vessels, levels of pain, etc.  What do you think?
