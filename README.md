# **Logistic Regression** 

A logistic regression with both L1 and L2 regularization will be implemented. The class LogisticRegression will have the following API:

* `__init__(alpha, tol, max_iter, theta_init, penalty, lambd)`
* `sigmoid(x)`
* `compute_cost(theta, X, y)`
* `compute_gradient(theta, X, y)`
* `has_converged(theta_old, theta_new)`
* `fit(X, y)`
* `predict_proba(X)`
* `predict(X)`


### **Sigmoid Function**

The sigmoid function $\sigma(x)$ is mathematically defined as follows.

<img src="https://latex.codecogs.com/svg.image?\sigma(x)&space;=&space;\frac{1}{1\&space;&plus;\&space;\text{exp}(-x)}" title="\sigma(x) = \frac{1}{1\ +\ \text{exp}(-x)}" />


### **Cost Function**

The cost function computes the scalar cost for a given $\theta$ vector.

<img src="https://latex.codecogs.com/svg.image?\mathcal{L}({\theta})&space;=&space;-\sum_{i&space;=1}^N&space;[&space;y_i\log(h_{{\theta}}({x}_i))&space;&plus;&space;(1&space;-&space;y_i)\log(1&space;-&space;h_{{\theta}}({x}_i))]" title="\mathcal{L}({\theta}) = -\sum_{i =1}^N [ y_i\log(h_{{\theta}}({x}_i)) + (1 - y_i)\log(1 - h_{{\theta}}({x}_i))]" />

where

<img src="https://latex.codecogs.com/svg.image?h_{\theta}(x_{i})&space;=&space;\sigma(\theta^{T}x_{i})" title="h_{\theta}(x_{i}) = \sigma(\theta^{T}x_{i})" />


L1 Regularization Loss:

<img src="https://latex.codecogs.com/svg.image?\mathcal{L1}({\theta})&space;=&space;\mathcal{L}({\theta})&space;&plus;&space;\lambda&space;\sum_{j&space;=&space;1}^D&space;&space;|{\theta}_j|" title="\mathcal{L1}({\theta}) = \mathcal{L}({\theta}) + \lambda \sum_{j = 1}^D |{\theta}_j|" />

L2 Regularization Loss:

<img src="https://latex.codecogs.com/svg.image?\mathcal{L2}({\theta})&space;=&space;\mathcal{L}({\theta})&space;&plus;&space;\lambda&space;\sum_{j&space;=&space;1}^D&space;&space;{\theta}_j^2&space;" title="\mathcal{L2}({\theta}) = \mathcal{L}({\theta}) + \lambda \sum_{j = 1}^D {\theta}_j^2 " />
