## Classification 

To attempt classification, one method is to use linear regression and map all predictions greater than 0.5 as a 1 and all less than 0.5 as a 0. However, this method doesn't work well because classification is not actually a linear function.

The classification problem is just like the regression problem, except that the values we now want to predict take on only a small number of discrete values. For now, we will focus on the **binary classification** **problem** in which y can take on only two values, 0 and 1. (Most of what we say here will also generalize to the multiple-class case.) For instance, if we are trying to build a spam classifier for email, then x*(*i*) may be some features of a piece of email, and y may be 1 if it is a piece of spam mail, and 0 otherwise. Hence, y∈{0,1}. 0 is also called the negative class, and 1 the positive class, and they are sometimes also denoted by the symbols “-” and “+.” Given x*(*i*), the corresponding y^{(i)} is also called the label for the training example. 



## Logistic Hypothesis Representation

We could approach the classification problem ignoring the fact that y is discrete-valued, and use our old linear regression algorithm to try to predict y given x. However, it is easy to construct examples where this method performs very poorly. Intuitively, it also doesn’t make sense for *hθ*(*x*) to take values larger than 1 or smaller than 0 when we know that y ∈ {0, 1}. To fix this, let’s change the form for our hypotheses *hθ(x)* to satisfy 0≤*hθ*(*x*)≤1. This is accomplished by plugging θ**T**x into the Logistic Function.

Our new form uses the "**Sigmoid Function**" also called the "Logistic Function":
$$
\begin{align*}& h_\theta (x) = g ( \theta^T x ) \newline \newline& z = \theta^T x \newline& g(z) = \dfrac{1}{1 + e^{-z}}\end{align*}
$$
The following image shows us what the sigmoid function looks like: 

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/1WFqZHntEead-BJkoDOYOw_2413fbec8ff9fa1f19aaf78265b8a33b_Logistic_function.png?expiry=1638316800000&hmac=_rkWeSMpEpkP_a_7cax2k3k61eUXnpAC3QVq3yC0IXc)



The function g(z), shown here, maps any real number to the (0, 1) interval, making it useful for transforming an arbitrary-valued function into a function better suited for classification.

*hθ*(*x*) will give us the **probability** that our output is 1. For example, h**θ*(*x*)=0.7 gives us a probability of 70% that our output is 1. Our probability that our prediction is 0 is just the complement of our probability that it is 1 (e.g. if probability that it is 1 is 70%, then the probability that it is 0 is 30%).
$$
\begin{align*}& h_\theta(x) = P(y=1 | x ; \theta) = 1 - P(y=0 | x ; \theta) \newline& P(y = 0 | x;\theta) + P(y = 1 | x ; \theta) = 1\end{align*}
$$


## Decision Boundary 

In order to get our discrete 0 or 1 classification, we can translate the output of the hypothesis function as follows:
$$
\begin{align*}& h_\theta(x) \geq 0.5 \rightarrow y = 1 \newline& h_\theta(x) < 0.5 \rightarrow y = 0 \newline\end{align*}
$$
The way our logistic function g behaves is that when its input is greater than or equal to zero, its output is greater than or equal to 0.5:
$$
\begin{align*}& g(z) \geq 0.5 \newline& when \; z \geq 0\end{align*}
$$


Remember.
$$
\begin{align*}z=0, e^{0}=1 \Rightarrow g(z)=1/2\newline z \to \infty, e^{-\infty} \to 0 \Rightarrow g(z)=1 \newline z \to -\infty, e^{\infty}\to \infty \Rightarrow g(z)=0 \end{align*}
$$


So if our input to g is θ**T**X*, then that means:
$$
\begin{align*}& h_\theta(x) = g(\theta^T x) \geq 0.5 \newline& when \; \theta^T x \geq 0\end{align*}
$$


From these statements we can now say:
$$
\begin{align*}& \theta^T x \geq 0 \Rightarrow y = 1 \newline& \theta^T x < 0 \Rightarrow y = 0 \newline\end{align*}
$$


The **decision boundary** is the line that separates the area where y = 0 and where y = 1. It is created by our hypothesis function.

**Example**:
$$
\begin{align*}& \theta = \begin{bmatrix}5 \newline -1 \newline 0\end{bmatrix} \newline & y = 1 \; if \; 5 + (-1) x_1 + 0 x_2 \geq 0 \newline & 5 - x_1 \geq 0 \newline & - x_1 \geq -5 \newline& x_1 \leq 5 \newline \end{align*}
$$


In this case, our decision boundary is a straight vertical line placed on the graph where x_1 = 5*x*1=5, and everything to the left of that denotes y = 1, while everything to the right denotes y = 0.

![image-20211129194955184](https://i.loli.net/2021/11/29/8S5aLXj7Q6AN1F9.png)

Again, the input to the sigmoid function g(z) (e.g. X*θ**T**X*) doesn't need to be linear, and could be a function that describes a circle (e.g. z*=*θ_0+θ_1*x*_1^2+*θ*2x_2^2) or any shape to fit our data.



## Cost Function

We cannot use the same cost function that we use for linear regression because the Logistic Function will cause the output to be **wavy**, causing many local optima. In other words, it will not be a **convex** function.

Instead, our cost function for logistic regression looks like:
$$
\begin{align*}& \mathrm{Cost}(h_\theta(x),y) = 0 \text{ if } h_\theta(x) = y \newline & \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{ if } y = 0 \; \mathrm{and} \; h_\theta(x) \rightarrow 1 \newline & \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{ if } y = 1 \; \mathrm{and} \; h_\theta(x) \rightarrow 0 \newline \end{align*}
$$
When y = 1, we get the following plot for 	J*(*θ*)  	vs 	h*θ*(*x):

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Q9sX8nnxEeamDApmnD43Fw_1cb67ecfac77b134606532f5caf98ee4_Logistic_regression_cost_function_positive_class.png?expiry=1638403200000&hmac=CaYXAu5GooLHfBqgjJx3wNkbf3d0tuHtFFBG_Jm-ilM)

Similarly, when y = 0, we get the following plot for 	J*(*θ*) vs 	h*θ*(*x):

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/Ut7vvXnxEead-BJkoDOYOw_f719f2858d78dd66d80c5ec0d8e6b3fa_Logistic_regression_cost_function_negative_class.png?expiry=1638403200000&hmac=Hg7-lcHYYDF-Fviqo0HPDlj6Y_Dv0qRQN6UQuUc-cnY)
$$
\begin{align*}& \mathrm{Cost}(h_\theta(x),y) = 0 \text{ if } h_\theta(x) = y \newline & \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{ if } y = 0 \; \mathrm{and} \; h_\theta(x) \rightarrow 1 \newline & \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{ if } y = 1 \; \mathrm{and} \; h_\theta(x) \rightarrow 0 \newline \end{align*}
$$
If our correct answer 'y' is 0, then the cost function will be 0 if our hypothesis function also outputs 0. If our hypothesis approaches 1, then the cost function will approach infinity.

If our correct answer 'y' is 1, then the cost function will be 0 if our hypothesis function outputs 1. If our hypothesis approaches 0, then the cost function will approach infinity.

Note that writing the cost function in this way guarantees that J(θ) is convex for logistic regression.



## Simplified Cost Function and Gradient Descent 

**Note:** [6:53 - the gradient descent equation should have a 1/m factor]

We can compress our cost function's two conditional cases into one case:
$$
\operatorname{Cost}\left(h_{\theta}(x), y\right)=-y \log \left(h_{\theta}(x)\right)-(1-y) \log \left(1-h_{\theta}(x)\right)
$$
We can fully write out our entire cost function as follows:
$$
J(\theta)=-\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \log \left(h_{\theta}\left(x^{(i)}\right)\right)+\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]
$$
A vectorized implementation is:
$$
\begin{align*} & h = g(X\theta)\newline & J(\theta) = \frac{1}{m} \cdot \left(-y^{T}\log(h)-(1-y)^{T}\log(1-h)\right) \end{align*}
$$

### **Gradient Descent**

Remember that the general form of gradient descent is:
$$
\begin{align*}& Repeat \; \lbrace \newline & \; \theta_j := \theta_j - \alpha \dfrac{\partial}{\partial \theta_j}J(\theta) \newline & \rbrace\end{align*}
$$


We can work out the derivative part using calculus to get:
$$
\begin{align*} & Repeat \; \lbrace \newline & \; \theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \newline & \rbrace \end{align*}
$$


Notice that this algorithm is identical to the one we used in linear regression. We still have to simultaneously update all values in theta.

A vectorized implementation is:
$$
\theta:=\theta-\frac{\alpha}{m} X^{T}(g(X \theta)-\vec{y})
$$

## Advanced Optimization

**Note:** [7:35 - '100' should be 100 instead. The value provided should be an integer and not a character string.]

- **"Conjugate gradient"**
- **"BFGS"**
- **"L-BFGS"** 

are more **sophisticated**, **faster** ways to optimize θ that can be used instead of gradient descent. We suggest that you should not write these more sophisticated algorithms yourself (unless you are an expert in numerical computing) but **use the libraries in matlab  instead**, as they're already tested and highly optimized. Octave provides them.

We first need to provide a function that evaluates the following two functions for a given input value θ:
$$
\begin{align*} & J(\theta) \newline & \dfrac{\partial}{\partial \theta_j}J(\theta)\end{align*}
$$


- We can write a single function that returns both of these:

```matlab
function [jVal, gradient] = costFunction(theta) 
jVal = [code to compute J(theta)]; 
gradient = [code to compute derivative of J(theta)];
end
```

Then we can use octave's "fminunc()" optimization algorithm along with the "optimset()" function that creates an object containing the options we want to send to "fminunc()". (Note: the value for MaxIter should be an integer, not a character string - errata in the video at 7:30)

```
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);  
[optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

We give to the function "fminunc()" our cost function, our initial vector of theta values, and the "options" object that we created beforehand.

### [fminunc](https://www.mathworks.com/help/optim/ug/fminunc.html)

Find minimum of unconstrained multivariable function

Finds the minimum of a problem specified by 
$$
\min _{x} f(x)
$$
where *f*(*x*) is a function that returns a scalar, *x* is a vector or a matrix; see [Matrix Arguments](https://www.mathworks.com/help/optim/ug/matrix-arguments.html).

- we try to solve a problem of `J(θ) =  (θ1 - 5)^2 + (θ2 - 5)^2` in gradient decent:

```matlab
% write a single function to compute the cost.
function [f, g] = costFunction(theta)
	f = sum((theta - 5).^2);	% cost
	g = 2*(theta - 5);			% gradient
end

% set options
options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);

% use 
theta0 = zeros(2,1);  
x = fminunc(@costFunction,theta0,options)
```

> output

```
Local minimum found.
Optimization completed because the size of the gradient is less than
the value of the optimality tolerance.
x =
     5
     5
```



## Multiclass Classification: One-vs-all

Now we will approach the classification of data when we have more than two categories. Instead of y = {0,1} we will expand our definition so that y = {0,1...n}.

Since y = {0,1...n}, we divide our problem into n+1 (+1 because the index starts at 0) binary classification problems; in each one, we predict the probability that 'y' is a member of one of our classes.
$$
\begin{align*}& y \in \lbrace0, 1 ... n\rbrace \newline& h_\theta^{(0)}(x) = P(y = 0 | x ; \theta) \newline& h_\theta^{(1)}(x) = P(y = 1 | x ; \theta) \newline& \cdots \newline& h_\theta^{(n)}(x) = P(y = n | x ; \theta) \newline& \mathrm{prediction} = \max_i( h_\theta ^{(i)}(x) )\newline\end{align*}
$$
We are basically choosing one class and then lumping all the others into a single second class. We do this repeatedly, applying binary logistic regression to each case, and then use the hypothesis that returned the highest value as our prediction.

The following image shows how one could classify 3 classes: 

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/cqmPjanSEeawbAp5ByfpEg_299fcfbd527b6b5a7440825628339c54_Screenshot-2016-11-13-10.52.29.png?expiry=1638489600000&hmac=i6BBUsfkQTifcW-2-qCJyGy80dNeHBtqwtEFO9sImt0)

**To summarize:** 

Train a logistic regression classifier h*θ*(*x*) for each class to predict the probability that y = i. To make a prediction on a new x, pick the class that maximizes  *hθ*(*x*)



