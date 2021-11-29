## Multiple Features

**Note:** [7:25 - \theta^T*θ**T* is a 1 by (n+1) matrix and not an (n+1) by 1 matrix]

Linear regression with multiple variables is also known as "multivariate linear regression".

We now introduce notation for equations where we can have any number of input variables.

*x*(*i*)*j**x*(*i*)*m**n*=value of feature *j* in the *i**t**h* training example=the input (features) of the *i**t**h* training example=the number of training examples=the number of features

The multivariable form of the hypothesis function accommodating these multiple features is as follows:

h_\theta (x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n*h**θ*(*x*)=*θ*0+*θ*1*x*1+*θ*2*x*2+*θ*3*x*3+⋯+*θ**n**x**n*

In order to develop intuition about this function, we can think about \theta_0*θ*0 as the basic price of a house, \theta_1*θ*1 as the price per square meter, \theta_2*θ*2 as the price per floor, etc. x_1*x*1 will be the number of square meters in the house, x_2*x*2 the number of floors, etc.

Using the definition of matrix multiplication, our multivariable hypothesis function can be concisely represented as:
$$
h*θ(x)=[θ_0 θ_1...θ_n][x_0x_1...x_n]=θ^T*x
$$
This is a vectorization of our hypothesis function for one training example; see the lessons on vectorization to learn more.

Remark: Note that for convenience reasons in this course we assume x_{0}^{(i)} =1 \text{ for } (i\in { 1,\dots, m } )*x*0(*i*)=1 for (*i*∈1,…,*m*). This allows us to do matrix operations with theta and x. Hence making the two vectors '\theta*θ*' and x^{(i)}*x*(*i*) match each other element-wise (that is, have the same number of elements: n+1).] 

## Gradient Descent For Multiple Variables

The gradient descent equation itself is generally the same form; we just have to repeat it for our 'n' features:
$$
\begin{matrix}

repeat \quad until \quad convergence:\{ \\ 
θ_j:=θ_j−α\frac{1}{m}\sum_{i=1}^{m}(h_θ(x^{(i)})−y^{(i)})x^{(i)}_j \qquad for j := 0...n
\\ \}

\end{matrix}
$$
The following image compares gradient descent with one variable to gradient descent with multiple variables: 

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/MYm8uqafEeaZoQ7hPZtKqg_c974c2e2953662e9578b38c7b04591ed_Screenshot-2016-11-09-09.07.04.png?expiry=1637539200000&hmac=rxn9WNSDYb13ag_EXo9D6Gotvny60Z7KeSihZrJbxhY)



## Gradient Descent in Practice I Feature Scaling

**Note:** [6:20 - The average size of a house is 1000 but 100 is accidentally written instead]

We can speed up gradient descent by having each of our input values in roughly the same range. This is because θ will descend quickly on small ranges and slowly on large ranges, and so will oscillate inefficiently down to the optimum when the variables are very uneven.

The way to prevent this is to modify the ranges of our input variables so that they are all roughly the same. Ideally:
$$
−1 ≤ x_{(i)} ≤ 1
$$
or
$$
−0.5 ≤ x_{(i)}≤ 0.5
$$
These aren't exact requirements; we are only trying to speed things up. The goal is to get all input variables into roughly one of these ranges, give or take a few.

Two techniques to help with this are **feature scaling** and **mean normalization**. Feature scaling involves dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1. Mean normalization involves subtracting the average value for an input variable from the values for that input variable resulting in a new average value for the input variable of just zero. To implement both of these techniques, adjust your input values as shown in this formula:
$$
x_i := \dfrac{x_i - \mu_i}{s_i}
$$


Where μ_i is the **average** of all the values for feature (i) and s_i is the range of values (max - min), or s_i is the standard deviation.

Note that dividing by the range, or dividing by the standard deviation, give different results. The quizzes in this course use range - the programming exercises use standard deviation.

For example, if x_i represents housing prices with a range of 100 to 2000  and a mean value of 1000, then, 
$$
x_i := \dfrac{price-1000}{1900}
$$

## Gradient Descent in Practice II Learning Rate

**Note:** [5:20 - the x -axis label in the right graph should be \theta*θ* rather than No. of iterations ]

**Debugging gradient descent.** Make a plot with *number of iterations* on the x-axis. Now plot the cost function, J(θ) over the number of iterations of gradient descent. If J(θ) ever increases, then you probably need to decrease α.

**Automatic convergence test.** Declare convergence if J(θ) decreases by less than E in one iteration, where E is some small value such as 10^{−3}10−3. However in practice it's difficult to choose this threshold value.

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/FEfS3aajEea3qApInhZCFg_6be025f7ad145eb0974b244a7f5b3f59_Screenshot-2016-11-09-09.35.59.png?expiry=1637625600000&hmac=zRFJ0LX6zJb39OLYSksgAEjPDQ1LuOTBI8haJ-vikKc)

It has been proven that if learning rate α is sufficiently small, then J(θ) will decrease on every iteration.

![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/rC2jGKgvEeamBAoLccicqA_ec9e40a58588382f5b6df60637b69470_Screenshot-2016-11-11-08.55.21.png?expiry=1637625600000&hmac=y_yXGq8MVl_wXQuCDLOavtLXxpj6F8cjTqfyBakQVfo)

To summarize:

 If α* is too small: slow convergence. 

 If *α* is too large: ￼may not decrease on every iteration and thus may not converge.



## Features and Polynomial Regression

We can improve our features and the form of our hypothesis function in a couple different ways.

We can **combine** multiple features into one. For example, we can combine x*1 and *x*2 into a new feature *x*3 by taking *x*1⋅*x*2.

### **Polynomial Regression**

Our hypothesis function need not be linear (a straight line) if that does not fit the data well.

We can **change the behavior or curve** of our hypothesis function by making it a quadratic, cubic or square root function (or any other form).

For example, if our hypothesis function is *h*θ*(*x*)=*θ*0+*θ*1*x*1 then we can create additional features based on *x*1, to get the quadratic function  
$$
h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2
$$
 or the cubic function 
$$
h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2 + \theta_3 x_1^3
$$
In the cubic version, we have created new features *x*2 and *x*3 where x_2 = x_1^2x and x_3 = x_1^3x 

To make it a square root function, we could do: 
$$
h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 \sqrt{x_1}
$$
One important thing to keep in mind is, if you choose your features this way then feature scaling becomes very important.

eg. if x_1 has range 1 - 1000 then range of x_1^2 becomes 1 - 1000000 and that of x_1^3 becomes 1 - 1000000000



## Normal Equation

**Note:** [8:00 to 8:44 - The design matrix X (in the bottom right side of the slide) given in the example should have elements x with subscript 1 and superscripts varying from 1 to m because for all m training sets there are only 2 features x_0*x*0 and x_1*x*1. 12:56 - The X matrix is m by (n+1) and NOT n by n. ]

Gradient descent gives one way of minimizing J. Let’s discuss a second way of doing so, this time performing the minimization explicitly and without resorting to an iterative algorithm. In the "Normal Equation" method, we will minimize J by explicitly taking its derivatives with respect to the θj ’s, and setting them to zero. This allows us to find the optimum theta without iteration. The normal equation formula is given below: 
$$
\theta = (X^T X)^{-1}X^T
$$


![img](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/dykma6dwEea3qApInhZCFg_333df5f11086fee19c4fb81bc34d5125_Screenshot-2016-11-10-10.06.16.png?expiry=1637798400000&hmac=S5B36n-ZSeYtQHiM8YHJine4UxDI6PMiPixZcdaLDAg)

There is **no need** to do feature scaling with the normal equation.

The following is a comparison of gradient descent and the normal equation:

| Gradient Descent                 | Normal Equation                            |
| :------------------------------- | :----------------------------------------- |
| Need to choose alpha             | No need to choose alpha                    |
| Needs many iterations            | No need to iterate                         |
| O (k n^2)                        | O (n^3), need to calculate inverse of X^TX |
| Works well when n is large       | Slow if n is very large                    |
| if n < 10^6 use gradient descent |                                            |

With the normal equation, computing the inversion has complexity \mathcal{O}(n^3)O(*n*3). So if we have a very large number of features, the normal equation will be slow. In practice, when n exceeds 10,000 it might be a good time to go from a normal solution to an iterative process.

## Normal Equation Noninvertibility

When implementing the normal equation in octave we want to use the 'pinv' function rather than 'inv.' The 'pinv' function will give you a value of \theta*θ* even if X^TX is not **invertible.** 

If X^TX is **noninvertible,** the common causes might be having :

- **Redundant features**, where two features are very closely related (i.e. they are linearly dependent)
- Too many features (e.g. m ≤ n). In this case, delete some features or use "**regularization**" **(正则化)**，(to be explained in a later lesson).

Solutions to the above problems include deleting a feature that is linearly dependent with another or deleting one or more features when there are too many features.



## Vectorial implementation
