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
