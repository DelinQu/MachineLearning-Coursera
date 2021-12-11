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
We can fully write out our **entire cost function** as follows:
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

```zsh
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



## The Problem of Overfitting

Consider the problem of predicting y from x ∈ R. The leftmost figure below shows the result of fitting a y = θ_0 + θ_1x*θ*0+*θ*1*x* to a dataset. We see that the data doesn’t really lie on straight line, and so the fit is not very good. 

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/0cOOdKsMEeaCrQqTpeD5ng_2a806eb8d988461f716f4799915ab779_Screenshot-2016-11-15-00.23.30.png?expiry=1639180800000&hmac=HYM_puZtBjEdynILECeE5Vo0rtTUlC8WU3dcFoHEksE)

Instead, if we had added an extra feature x^2 , and fit y = theta_0 + theta_1x , then we obtain a slightly better fit to the data (See middle figure). Naively, it might seem that the more features we add, the better. However, there is also a danger in adding too many features: The rightmost figure is the result of fitting a 5^{th}5*t**h* order polynomial y*=∑*j*=05*	θj xj. We see that even though the fitted curve passes through the data perfectly, we would not expect this to be a very good predictor of, say, housing prices (y) for different living areas (x). Without formally defining what these terms mean, we’ll say the figure on the left shows an instance of **underfitting**—in which the data clearly shows structure not captured by the model—and the figure on the right is an example of **overfitting**.

Underfitting, or high bias, is when the form of our hypothesis function h maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. At the other extreme, overfitting, or high variance, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

This terminology is applied to both linear and logistic regression. There are two main options to address the issue of overfitting:

**1）Reduce the number of features:**

- Manually select which features to keep.
- Use a **model selection algorithm** (studied later in the course).

**2）Regularization**

- Keep all the features, but reduce the **magnitude of parameters** θj.
- Regularization works well when we have a lot of slightly useful features.



## Cost Function

**Note:** [5:18 - There is a typo. It should be \sum_{j=1}^{n} \theta _j ^2∑*j*=1*n**θ**j*2 instead of \sum_{i=1}^{n} \theta _j ^2∑*i*=1*n**θ**j*2]

If we have overfitting from our hypothesis function, we can reduce the weight that some of the terms in our function carry by increasing their cost.

Say we wanted to make the following function more quadratic:
$$
\theta_0 + \theta_1x+\theta_2x^2 + \theta_3x^4
$$
We'll want to eliminate the influence of \theta_3x^3*θ*3*x*3 and \theta_4x^4*θ*4*x*4 . Without actually getting rid of these features or changing the form of our hypothesis, we can instead modify our **cost function**:
$$
min_{\theta}\frac{1}{2m}\sum_{i=1}^{m}(h_\theta (x^{(i)}-y^{(i)})^2+1000\theta_3^2+1000\theta_4^2 
$$


We've added two extra terms at the end to inflate the cost of \theta_3*θ*3 and \theta_4*θ*4. Now, in order for the cost function to get close to zero, we will have to reduce the values of \theta_3*θ*3 and \theta_4*θ*4 to near zero. This will in turn greatly reduce the values of \theta_θ*3*x*3 and  *θ*4*x*4 in our hypothesis function. As a result, we see that the new hypothesis (depicted by the pink curve) looks like a quadratic function but fits the data better due to the extra small terms θ*3*x*3 and  *θ*4*x*4.

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/j0X9h6tUEeawbAp5ByfpEg_ea3e85af4056c56fa704547770da65a6_Screenshot-2016-11-15-08.53.32.png?expiry=1639180800000&hmac=AB65Qy2gXVzV2VSE_oqNZycWBYvk4dzXcdF6JNgAOfQ)

We could also regularize all of our theta parameters in a single summation as:
$$
\min _{\theta} \frac{1}{2 m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}+\lambda \sum_{j=1}^{n} \theta_{j}^{2}
$$


The λ, or lambda, is the **regularization parameter**. It determines how much the costs of our theta parameters are inflated. 

Using the above cost function with the extra summation, we can smooth the output of our hypothesis function to reduce overfitting. If lambda is chosen to be too large, it may smooth out the function too much and cause underfitting. Hence, what would happen if \lambda = 0*λ*=0 or is too small ?



## Regularized Linear Regression

**Note:** [8:43 - It is said that X is non-invertible if m \leq≤ n. The correct statement should be that X is non-invertible if m < n, and may be non-invertible if m = n.

We can apply regularization to both linear regression and logistic regression. We will approach linear regression first.

### Gradient Descent

We will modify our gradient descent function to separate out \theta_0*θ*0 from the rest of the parameters because we do not want to penalize:
$$
\begin{align*} & \text{Repeat}\ \lbrace \newline & \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline & \ \ \ \ \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace\newline & \rbrace \end{align*}
$$


The term \frac{\lambda}{m}\theta_j*m**λ**θ**j* performs our regularization. With some manipulation our update rule can also be represented as:
$$
\theta_{j}:=\theta_{j}\left(1-\alpha \frac{\lambda}{m}\right)-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}
$$


The first term in the above equation, 1 - \alpha\frac{\lambda}{m}1−*α**m**λ* will always be less than 1. Intuitively you can see it as reducing the value of \theta_j*θ**j* by some amount on every update. Notice that the second term is now exactly the same as it was before.

### **Normal Equation**

Now let's approach regularization using the alternate method of the non-iterative normal equation.

To add in regularization, the equation is the same as our original, except that we add another term inside the parentheses:
$$
\begin{align*}& \theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty \newline& \text{where}\ \ L = \begin{bmatrix} 0 & & & & \newline & 1 & & & \newline & & 1 & & \newline & & & \ddots & \newline & & & & 1 \newline\end{bmatrix}\end{align*}
$$
L is a matrix with 0 at the top left and 1's down the diagonal, with 0's everywhere else. It should have dimension (n+1)×(n+1). Intuitively, this is the identity matrix (though we are not including x_0*x*0), multiplied with a single real number λ.

Recall that if m < n, then X**T**X s non-invertible. However, when we add the term λ⋅L, then X^TX + λ⋅L becomes invertible.
