# Udacity Deep Learning Nanodegree

## General

Forum: https://discussions.udacity.com/c/nd101-Project-1/
Slack: https://nd101.slack.com/messages/announcements/

## Term 1

### Additional Resources

#### Books

- https://www.manning.com/books/grokking-deep-learning
- http://neuralnetworksanddeeplearning.com/
- http://www.deeplearningbook.org/

#### Study material

- https://www.khanacademy.org/math/linear-algebra
- https://www.khanacademy.org/math/multivariable-calculus
  - https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivatives/v/partial-derivatives-introduction

#### Videos

- Gradient descent: https://youtu.be/eikJboPQDT0
- Partial derivatives: https://youtu.be/1CMDS4-PKKQ

### Notes

- Deep Learning is a subset of Machine Learning.
- Machine Learning can rougly be broken apart to three categories
  - Supervised: inferring a function from a labeled set of data
  - Unsupervised: inferring a function from an unlabled set of data
  - Reinforcement: inferring a function through exploration and rewarding
    positive behavior.
- Programming is about defining the steps needed to accomplish an outcome.  Deep
  learning is about defining the outcome and letting an algorithm find the steps
  to accomplish the outcome.

- Hyperparameters: Are the tuning notches for machine learning algorithms

- Linear Regression can be used to model linear sets of data.
  - It won't work for non linear sets.
  - It can easily be thrown off by outliers.
  - You can have multiple linear regression (3d linear plane).
- Linear regression typically has the following hyperparameters: learning rate,
  iteration count, y intercept, slope.

- Gradient Descent
  - Used to find the smallest error rate in a 3d plane of x, y, and error rate.
  - Think of it as dropping a ball in a landscape and watching it roll to the
    lowest point.
  - It works by having a starting point and finding the two relative slopes at
    that starting point. Once the slopes around found, the learning rate is used
    to move the starting point in the direction of the slopes.  Then the process
    is repeated.  The slopes are found using partial derivatives.

- Partial Derivative
  - The same as normal derivatives, but with multi variable functions.
  - f(x, y) you'd either derivate in respect to x or y in which case you'd treat
    one of them as a constant. f(x, y) = x^2 + yx. in respect to y being 2,
    you'd get fx(x, y) = 2x + 2.
  - Think of the 3d graph for a multi variable function and slicing it with a
    plane, then using the intersect against that plane to find slope.

### Draft Notes (to clean)

Maybe come back to this video and implmement the nueral network:
https://classroom.udacity.com/nanodegrees/nd101/parts/2a9dba0b-28eb-4b0e-acfa-bdcf35680d90/modules/329a736b-1700-43d4-9bf0-753cc461bebc/lessons/c3dd053a-7660-4fc5-8bdc-53301ac7ce51/concepts/9f6f9683-3f44-43a3-9d5b-f30fad2b6127

http://natureofcode.com/book/chapter-10-neural-networks/
https://www.oreilly.com/learning/hello-tensorflow
https://www.tensorflow.org/tutorials/mnist/beginners/

https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/

Add notes from notebook
  - Perceptron
  - single layer feed forward network

law of intelligence endcoded in the universe and we are discovering it

tensor calculus

vector is a 1d tensor
matrix is a 2d tensor
list of list of list is a 3d tensor, and so on.

libraries used
- pandas
- tensorflow
- numpy
- matplot

softmax = sigmoid (out activation fuction?)

cost function - mean squared error

---

logistic regression is the building block for nueral networks.
- think line drawn through two sets of colored points on a graph.
- an ideal logic regression line will split the two sets of points in half.
- to get an ideal line, gradient descent is used. step in the direction that
  minimizes the error count.

a logic regression can have multiple lines that split the data in different
ways. this is what a nueral network is made up of.

the perceptron is the base unit of a nueral network

weights and bias determine the overall importance of a perceptron

generally, this is how a perceptron works
- sum all inputs times their respective weights.
- take the summed value above and add a bias
- take the previous value and run it through an activation function
- the activation function is used to generate an output value for the entire
  perceptron.
- an example activation function is heaviside which returns 1 if the value is
  greater than or equal to 0 and returns 0 if the value is less than 0.

an and perceptron would work like the following
point (0, 0) = 0
point (0, 1) = 0
point (1, 0) = 0
point (1, 1) = 1
it could be implemented with:
weight1 = 1.0
weight2 = 1.0
bias = -2.0

not, or, and can all be implemented in perceptrons, but they work with ranges of
values instead of binary values
you can chain together perceptrons to build an xor nueral network.
nueral networks can solve any problem a computer can solve
- think about logic gates making up a cpu. inefficent though.
the real power of a nueral network is not building them by hand, but the ability
for them to set their own weights and biases.

list of common activation functions
- logistic (sigmoid)
- tanh
- softmax

these functions are continous and differentiable which allow training via
gradient descent (partial derivatives).

sigmoid has an output (y) range of 0 to 1. it can be considered a probability of
success.

nueral networks are getting better than humans with things because of their pure
focus. a nueral network doesn't have any distractions and only knows to do one
thing. it'd be like a human that knew nothing but chess and didn't get tired or
distracted or have any feelings.

---
