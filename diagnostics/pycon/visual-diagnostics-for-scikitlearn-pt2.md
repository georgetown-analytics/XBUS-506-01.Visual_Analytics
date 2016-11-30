# Visual Diagnostics for More Informed Machine Learning, Part 2: Demystifying Model Selection

_Note: Before starting Part 2, be sure to read Part 1!_

When it comes to machine learning, ultimately the most important picture to have is the big picture. Discussions of (i.e. arguments about) machine learning are usually about which model is the best. Whether it's logistic regression, random forests, Bayesian methods, support vector machines, or neural nets, everyone seems to have their favorite! Unfortunately these discussions tend to truncate the challenges of machine learning into a single problem, which is a particularly problematic misrepresentation for people who are just getting started with machine learning. Sure, picking a good model is important, but it's certainly not enough (and it's debatable whether a model can actually be 'good' devoid of the context of the domain, the hypothesis, the shape of the data, the intended application, etc, but we'll leave that to another post).

In this post we'll discuss model selection in the context of the big picture, which I'll present in terms of the 'model selection triple', and we'll explore a set of visual tools for navigating the triple.

## The Model Selection Triple

Producing a fitted model that is well-suited to the data, predictive, and also performant is critically dependent on feature selection and tuning as well as model selection. Kumar et al. refer to this trio of steps as the [model selection triple](http://pages.cs.wisc.edu/~arun/vision/SIGMODRecord15.pdf). As they explain, &ldquo;Model selection is iterative and exploratory because the space of [model selection triples] is usually infinite, and it is generally impossible for analysts to know a priori which [combination] will yield satisfactory accuracy and/or insights.&rdquo; In other words, this is the part that makes machine learning _hard_. The process is complex, iterative, and disjointed, often with many missteps and restarts along the way. And yet these iterations are central to the science of machine learning &mdash; optimization is not about limiting those iterations (e.g. helping you pick the best model on the first try every time), but about facilitating them. For that reason, let's begin with the visualization I think is the most important of all: a view of the workflow that I use to put together all of the steps and visual diagnostics described throughout Parts 1, 2, and 3 of this post.

As shown in the diagram below, I begin with data stored on disk and take a first pass through feature analysis using histograms, scatterplots, parallel coordinates and other visual tools. My analysis of the features often leads back to the data, where I take another pass through to normalize, scale, extract, or otherwise wrangle the attributes. After more feature analysis has confirmed I'm on the right track, I identify the category of machine learning models best suited to my features and problem space, often experimenting with fit-predict on multiple models. I iterate between evaluation and tuning using a combination of numeric and visual tools like ROC curves, residual plots, heat maps and validation curves. Finally, the best model is stored back to disk for later use.

![Model Selection Triple Workflow](figures/model_triple_workflow.png)

In Part 1 of this post, we covered the feature analysis tools, and we'll explore evaluation and tuning later in Part 3. Here in Part 2, we'll be focusing on the decision-making process that goes into choosing the set of algorithms to use for a given dataset. As in Part 1, we'll be using three different datasets from the UCI Machine Learning Repository &mdash; one about room occupancy, one about credit card default, and one about concrete compressive strength. Using those three datasets, we'll explore a range of models, some of which will work better than others, and none of which will work equally well for all three. This is normal, which is why the goal is not to get good enough at machine learning that we can pick the best model on the first try every time, but to adopt a process that will facilitate exploration and iteration. Because we want to be doing _informed_ machine learning, we do want some kind of process, and the flowcharts and maps we'll explore below can serve as a guide.

## Getting Started with Model Selection
Those who have used Scikit-Learn before will no doubt already be familiar with the [Choosing the Right Estimator](http://scikit-learn.org/stable/tutorial/machine_learning_map/) flow chart. This diagram is handy for those who are just getting started, as it models a(n albeit simplified) decision-making process for selecting the machine learning algorithm that is best suited to one's dataset.

![Scikit-Learn: Choosing the Right Estimator](figures/scikitlearncheatsheet.png)

Let's try it together. First we are asked whether we have more than 50 samples &hellip;

```python
print len(occupancy) # 8,143
print len(credit)    # 30,000
print len(concrete)  # 1,030
```

&hellip; which we do for each of our three datasets. Next we're asked if we're predicting a category. For the occupancy and credit datasets, the answer is yes: for occupancy, we are predicting whether a room is occupied (0 for no, 1 for yes), and for credit, we are predicting whether the credit card holder defaulted on their payment (0 for no, 1 for yes). For the concrete dataset, the labels for the strength of the concrete are continuous, so we are predicting a quantity, not a category. Therefore, we will be looking for a classifier for our occupancy and credit datasets, and for a regressor for our concrete dataset.

Since both of our categorical datasets have fewer than 100,000 instances, we are prompted to start with `sklearn.svm.LinearSVC` (which will map the data to a higher dimensional feature space), or failing that, `sklearn.neighbors.KNeighborsClassifier` (which will assign instances to the class most common among its k nearest neighbors). In our feature exploration of the occupancy dataset, you'll remember that the different attributes were not all on the same scale, so in addition to the other steps, we import `scale` so that we can standardize all the features before we run fit-predict:

```python
from sklearn.preprocessing import scale

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

def classify(attributes, targets, model):
    # Split data into 'test' and 'train' for cross validation
    splits = cv.train_test_split(attributes, targets, test_size=0.2)
    X_train, X_test, y_train, y_test = splits

    model.fit(X_train, y_train)
    y_true = y_test
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_true, y_pred))

# Divide data frame into features and labels
features = occupancy[['temp', 'humid', 'light', 'co2', 'hratio']]
labels   = occupancy['occupied']

# Scale the features
stdfeatures = scale(features)

classify(stdfeatures, labels, LinearSVC())
classify(stdfeatures, labels, KNeighborsClassifier())
```

Ok, let's use the same `classify` function to model the credit default dataset next. As you'll remember from our visual exploration of the features, while there are two classes in this dataset, there are very few cases of default, meaning we should be prepared to see some manifestations of class imbalance in our classifier.

```python
features = credit[[
    'limit', 'sex', 'edu', 'married', 'age', 'apr_delay', 'may_delay',
    'jun_delay', 'jul_delay', 'aug_delay', 'sep_delay', 'apr_bill', 'may_bill',
    'jun_bill', 'jul_bill', 'aug_bill', 'sep_bill', 'apr_pay', 'may_pay',
    'jun_pay', 'jul_pay', 'aug_pay', 'sep_pay'
]]
labels   = credit['default']

stdfeatures = scale(features)

classify(stdfeatures, labels, LinearSVC())
classify(stdfeatures, labels, KNeighborsClassifier())
```

Meanwhile for our concrete dataset, we must determine whether we think all of the features are important, or only a few of them. If we decide to keep all the features as is, the chart suggests using `sklearn.linear_model.RidgeRegression` (which will identify features that are less predictive and ensure they have less influence in the model) or possibly `sklearn.svm.SVR` with a linear kernel (which is similar to the LinearSVC classifier). If we guess that some of the features are not important, we might decide instead to choose `sklearn.linear_model.Lasso` (which will drop out any features that aren't predictive) or `sklearn.linear_model.ElasticNet` (which will try to find a happy medium between the Lasso and Ridge methods, taking the linear combination of their L1 and L2 penalties).

Let's try a few because, why not?

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

def regress(attributes, targets, model):
    splits = cv.train_test_split(attributes, targets, test_size=0.2)
    X_train, X_test, y_train, y_test = splits

    model.fit(X_train, y_train)
    y_true = y_test
    y_pred = model.predict(X_test)
    print('Mean squared error = {:0.3f}'.format(mse(y_true, y_pred)))
    print('R2 score = {:0.3f}'.format(r2_score(y_true, y_pred)))

features = concrete[[
    'cement', 'slag', 'ash', 'water', 'splast', 'coarse', 'fine', 'age'
]]
labels   = concrete['strength']

regress(features, labels, Ridge())
regress(features, labels, Lasso())
regress(features, labels, ElasticNet())
```

As illustrated in the code above, the Scikit-Learn API allows us to rapidly deploy as many models as we want. This is an incredibly powerful feature of the Scikit-Learn library that cannot be understated. After all, being able to iterate and experiment is why we all got into science in the first place, right? As we learned in the beginning of this post, the workflows we use to navigate the model selection triple tend to be highly non-linear, and iteration and experimentation are particularly key to model selection.

## Visualizing Models at Work

The Scikit-Learn flowchart is useful because it offers us a map, but it doesn't offer much in the way of transparency about how the various models are functioning. For that kind of insight, there are two images that have become somewhat canonical in the Scikit-Learn community: the [classifier comparison](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) and [cluster comparison](http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html) plots.

Because unsupervised learning is done without the benefit of a ground truth to inform us when we have labeled data properly, this plot of small multiples is a useful way to compare different clustering algorithms across different  datasets:    

![Cluster comparison plot](figures/clustercompare_DDL.png)

Similarly, the classifier comparison plot below is a helpful visual comparison of the performance of nine different classifiers across three different toy datasets:     

![Classifier comparison plot](figures/classifiercompare_DDL.png)

Generally these images are used just to demonstrate the substantial differences in the performance of various models across different datasets. Unfortunately, the datasets used are synthetic, and while it would be exciting to be able to operationalize the code as a visual tool for model selection and exploration, the curse of dimensionality will pose problems for most real-world datasets. Nonetheless, I find it helpful to able to picture the behavior of different models in the same dataspace. For example, in image above we can see the difference in way that `KNeighborsClassifier` and `LinearSVC` divide the data up, which provides some useful insight into how the two algorithms are operating with our room occupancy and credit card default datasets.

## Model Families
Visualizations and flow diagrams of the model selection process like Scikit-Learn's &ldquo;Choosing the Right Estimator&rdquo; can be helpful, especially when you're just getting started out with machine learning. But what do you do when you've exhausted all those options? There are a lot more models available in Scikit-Learn, and the estimator flowchart only barely scratches the surface. It is possible to use an exhaustive approach to essentially test the [entire catalog of Scikit-Learn models](http://scikit-learn.org/stable/modules/classes.html) in order to find the one that works the best on your dataset. But if our goal is to be more _informed_ machine learning practitioners, we care not only about whether our models are working, but also about _why_ they are or are not working. For that matter, we still haven't addressed what we mean when we say a model is 'working'. The outputs of our `classify` and `regress` functions above only give us a very small picture of what's happening, and can mask a lot of problems. We'll discuss more about model evaluation in Part 3.  

For now, we're looking for a more systematic way to experiment with different kinds of models once we've exhausted the options proposed to us through the Scikit-Learn flow chart. To do that, let's first take a step back and reconsider what we mean when we say 'model'. As Hadley Wickham [points out](http://had.co.nz/stat645/model-vis.pdf), the word 'model' is an overloaded term because we use it to mean at least three different things:

 - the model family (e.g. linear model, nearest neighbors, support vector machine, trees, Bayes, ensemble methods)    
 - the model form (e.g. `sklearn.linear_model.Ridge()`, `sklearn.linear_model.Lasso()`, `sklearn.linear_model.ElasticNet`)    
 - the fitted model (e.g. the result of `Ridge().fit(X_train, y_train)`)    

As Wickham explains, _model family_ is largely determined by the problem space, whereas _model form_ is chosen through experimentation and statistical testing (or sometimes based on the preference of the practitioner), and a _fitted model_ is generated through a combination of human parameter tuning and machine computation.

For our purposes, experimentation within the model forms (and as we'll discuss later, [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_optimization)), is probably the place where we can expect to get the most return on our investment. Model form specifies how our features are related within the framework of the model family. In the context of our concrete model, the forms of Ridge, Lasso, and ElasticNet specify that the 'strength' variable is the target, whereas cement content, slag, ash, water, superplasticity, coarseness, fineness, and age are the predictors. They also specify how those predictors are related to each other and to the target.

One tool I like for model exploration is Dr. Saed Sayad's interactive ['data mining map'](http://www.saedsayad.com/data_mining_map.htm), because it is much more comprehensive than the Scikit-Learn flow chart and integrates a lot of the ideas of model family and form. Moreover, in addition to predictive methods, Sayad's map includes a separate section on statistical methods for explaining the past.

Below is an iteration on a genealogical chart that we've been exploring at District Dat Labs. It aims to present the same broad coverage of predictive methods as Sayad's (representing some, like reinforcement learning, not represented in original), while integrating the [Scikit-Learn model classes](http://scikit-learn.org/stable/modules/classes.html) that correspond to the model forms. Color and hierarchy designate the model forms and model families:

![Model selection genealogical chart](figures/ml_map_v4.png)

While our map isn't comprehensive (and more importantly, not yet interactive), we envision it becoming the intermediate version of the Scikit-Learn estimator flow chart, a visual selection tool that can be integrated seamlessly into the  model selection triple workflow.

## Conclusion

In Part 1 we began our visual journey at the feature analysis and feature selection phases of the model selection triple, and in Part 2, we've moved to the model selection phase. It is worth repeating that for many machine learning practitioners, the traversal of the phases is iterative and non-linear. Within the model family that is appropriate to your problem space, it is useful (and easy) to explore multiple model forms, though I do recommend using the visualizations, as well as the other tools provided in this post (like the `classify` and `regress` wrapper functions) to experiment in as strategic a way as possible.

By comparing and contrasting the performance of different models on a single dataset, and by doing this in a repeatable way over time with a number of datasets, we can begin building intuition around the model forms likely to outperform their siblings in the model family.  In Part 3, we'll move into the next phases of the model selection triple, exploring a suite of visual tools for evaluating fitted models and for tuning their parameters to improve their performance.

Read Part 3 next: Coming soon!

### Resources and Helpful Links

- [Visualizing Statistical Models: Removing the Blindfold by Hadley Wickham et al.](http://had.co.nz/stat645/model-vis.pdf)
- [Model Selection Management Systems by Arun Kumar et al.](http://pages.cs.wisc.edu/~arun/vision/)
- [A Visual Introduction to Machine Learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
- [The Scikit-Learn Algorithm Cheatsheet](http://scikit-learn.org/stable/tutorial/machine_learning_map/)
- [Visualizing Machine Learning Thresholds](http://blog.insightdatalabs.com/visualizing-classifier-thresholds/)     
- [ML Demos](http://mldemos.epfl.ch/)
- [Plotting SVM Classifiers](http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html#example-svm-plot-iris-py)
- [Introduction to ROC Analysis](https://ccrma.stanford.edu/workshops/mir2009/references/ROCintro.pdf)    
- [Visualizing Representations](http://colah.github.io/posts/2015-01-Visualizing-Representations/)    
- [Accurately Measuring Model Prediction Error](http://scott.fortmann-roe.com/docs/MeasuringError.html)    
- [Understanding the Bias-Variance     Tradeoff](http://scott.fortmann-roe.com/docs/BiasVariance.html)
