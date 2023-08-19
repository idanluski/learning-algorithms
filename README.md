# learning-algorithms

**KNN**\
• Implement in knn.py according to the template given.\
• Constructor – The constructor receives a single argument named k, which dictates the
number of neighbors considered.\
• A “fit” method.\
o The method receives a NumPy array of samples to be used as train data (called
x_train in the template file). In the array, each row is a data sample whose
dimensions equal the number of columns.\
o The method receives a second 1-d NumPy array of labels per training sample to
be used as train data (called y_train in the template file).\
o It doesn’t return anything.\
• A “predict” method.\
o The method receives a NumPy array of samples whose class is to be predicted. In
the array, each row is a data sample whose dimensions equal the number of
columns.\
o The method returns a 1-d NumPy array of labels, one per training sample.\
• The underlying data type which will be used both for features and labels will be integers.\
**K-Means**\
• Implement in k_means.py according to the template given.\
• Constructor – The constructor receives an argument named k, which dictates the
number of clusters considered, and max_iter which dictates the maximal number of
iterations done during the fit stage.\
• An “initialize” method.\
o As you might have read, the k-means algorithm performance depends on the
initialization of the model. Even though it is customary to use random
initialization, this would make it hard for us to grade your work.\
o Therefore, you are required to implement an “initialize” method, which will
accept a single argument. The argument is a 2-d NumPy array. The number of
rows is the same as the number of clusters, and each row will represent the
initial coordinates of some centroid. You are required to use this initialization for
the clusters.\
o Our test code will call this method before the “fit” method to provide some
initialization for clusters.\
• A “fit” method.\
o The method receives a NumPy array of samples and uses it as train data (called
x_train in the template file). In the array, each row is a data sample whose
dimensions equal the number of columns. The method performs the k-means
algorithm on the train data to find centroids and train the model.\
o The method returns a dictionary. Its keys are clusters IDs that you assign during
training (unique integers of your choice). The values are 1-d NumPy arrays, each
of which represents a centroid.\
• A “predict” method.\
o The method receives a NumPy array of samples to be assigned to clusters. Each
row is a data sample whose dimensions equal the number of columns.\
o The method returns a 1-d NumPy array of cluster IDs. Each ID correlates to the
cluster to which the sample was assigned. Cluster IDs are the same ones
returned during training.\
o Note that the centroids are calculated only during training, and aren’t changed
by the predict method.\
• A “wcss” method – The method will return the WCSS of the model, as calculated during
the “fit” stage. During grading, it won’t be called before the “fit” method is called.\
• The underlying data type which will be used both for features is floats.\
