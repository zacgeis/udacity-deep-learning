# Udacity Deep Learning Nanodegree

## General

Forum: https://discussions.udacity.com/c/nd101-Project-1/
Slack: https://nd101.slack.com/messages/announcements/

## Term 1

### Additional Resources

#### Books

- Buy this book when the paper copy comes out: https://www.manning.com/books/grokking-deep-learning
- http://neuralnetworksanddeeplearning.com/
- http://www.deeplearningbook.org/

#### Study material

- https://www.khanacademy.org/math/linear-algebra
- https://www.khanacademy.org/math/multivariable-calculus
  - https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/partial-derivatives/v/partial-derivatives-introduction
- https://youtu.be/BR9h47Jtqyw - Friendly introduction to deep learning
- https://youtu.be/Q9Z20HCPnww - Deep learning dymistiied
- https://youtu.be/FmpDIaiMIeA - How CNNs work
- https://youtu.be/Ih5Mr93E-2c - Lecture 10 Neural Networks
- https://youtu.be/vq2nnJ4g6N0 - Tensorflow and deeplearning without a PHD
- https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/

#### TODO

- Gradients, partial derivates, dot product, matrix, vectors, chain rule,
gradient descent, scalable vector machine (SVM)
- There's a case where both dot and multiplication will return the same value
3x1 (x/.) 1x2
- TODO Create simple 2-3-1 neural network with only variable names and show where
the full derivations come from. Use multiple dimensional array, but index
everything by hand and don't use sum functions.

#### Resources

- Gradient descent: https://youtu.be/eikJboPQDT0
- Partial derivatives: https://youtu.be/1CMDS4-PKKQ

- Some of the best resources I've found:
  - https://iamtrask.github.io/2015/07/27/python-network-part2/
  - http://iamtrask.github.io/2015/07/12/basic-python-network/
  - http://karpathy.github.io/neuralnets/
  - https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
  - https://en.m.wikipedia.org/wiki/Delta_rule
  - https://en.wikipedia.org/wiki/Chain_rule
  - http://colah.github.io/posts/2015-08-Backprop/
  - http://cs231n.github.io/
  - http://staff.itee.uq.edu.au/janetw/cmc/chapters/BackProp/index2.html
  - http://natureofcode.com/book/chapter-10-neural-networks/
  - https://www.oreilly.com/learning/hello-tensorflow
  - https://www.tensorflow.org/tutorials/mnist/beginners/
  - https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/
history of deep learning: https://vimeo.com/170189199
understanding back prop: https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.eazs6hdr1
stanford lecture on back prop: https://www.youtube.com/watch?v=59Hbtz7XgjM
latex primer from udacity: http://data-blog.udacity.com/posts/2016/10/latex-primer/
vector dot product khan: https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces/dot-cross-products/v/vector-dot-product-and-vector-length
matrix dot product khan: https://www.khanacademy.org/math/linear-algebra/matrix-transformations/composition-of-transformations/v/linear-algebra-matrix-product-examples
no bs linear algebra has a great
great intro to nueral networks and back propagation: https://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
also great details on back prop: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
and: http://cs231n.github.io/optimization-2/#patters

- Great resource for determining the number of hidden nodes and layers:
  - http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

### Basic Neural Network Notes

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

- Is there a general equation or algorithm of intelligence that we are
discovering?

- Tensor calculus
  - Vector is a 1d tensor
  - Matrix is a 2d tensor
  - List of list of list is a 3d tensor, and so on.

- Libraries used:
  - pandas
  - tensorflow
  - numpy
  - matplot

- Activation function is used to calculate the output of a node given its inputs
  - Sigmoid is an activation function commonly used (also called softmax)
  - You can also have just linear activation functions
- The cost function determines how well the model performed
  - Mean squared error is a common cost function

- Logistic regression is the building block of nueral networks.
  - Think line drawn through two sets of colored points on a graph.
  - An ideal logic regression line will split the two sets of points in half.
  - To get an ideal line, gradient descent is used. Step in the direction that
    minimizes the error count.

- Logic regression can have multiple lines that split the data in different
ways. This is what a nueral network is made up of.

- The perceptron is the base unit of a nueral network

- Weights and bias determine the overall importance of an individual perceptron
or node

- Generally, this is how a perceptron works:
  - Sum all inputs times their respective weights. `sum(input[i] * weight[i])`
  - Take the summed value above and add a bias
  - Take the previous value and run it through an activation function
  - The activation function is used to generate an output value for the entire
    perceptron.
  - An example activation function is heaviside which returns 1 if the value is
    greater than or equal to 0 and returns 0 if the value is less than 0.

Simple example perceptron line with data points:

```
weight1 = 1.0
weight2 = 1.0
bias = -2.0

point (0, 0) = 0
point (0, 1) = 0
point (1, 0) = 0
point (1, 1) = 1
```

- Logic gates (and, or, not) can all be implemented with perceptrons, but they work with ranges of
values (confidence) instead of binary values.
- You can chain together perceptrons to build an xor nueral network.
nueral networks can solve any problem a computer can solve
- Think about logic gates making up a CPU. It'd be pretty inefficent though.
- The real power of a nueral network is not building them by hand, but the ability
for them to set their own weights and biases through backprop.  Basically, they
can learn how to achieve a certain output.

- It's important that activation functions are continous and differentiable which allow training via
gradient descent (partial derivatives).

- Sigmoid has an output (y) range of 0 to 1. It can be considered a probability of
success.

- Nueral networks are getting better than humans with things because of their pure
focus. A nueral network doesn't have any distractions and only knows to do one
thing. It'd be like a human that knew nothing but chess and didn't get tired or
distracted or have any feelings and was in many was much less complicated.

- Logic gates are powerful, but perceptrons allow them to learn and change
behavior - adapt. Basically creating a logic flow / program by learning. Learning happens with
derivatives at the math level.

- The current largest network is 1B nuerons - the human brain is 100B. It might
take that level of complexity for true human levels of understanding.

- It's like a flexible set of logic gates that can be trained to have a certain
output.  The training only takes the inputs and what you'd like the outputs to
be.

- It'd be interesting to train a large neural network to be a standard VM with op
codes and memory.

- Originally modeled after human neurons. They can be trained without gradient
descent, and gradient descent is where much of the underlying power and speed
comes from. It's basically a search function looking for the best possible
configuration of numbers to make the best network with the lowest error output.

- Khan video on gradient descent:
  - https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/gradient-and-directional-derivatives/v/gradient

- Momentum can be used to avoid falling into local minima with gradient descent
  - http://sebastianruder.com/optimizing-gradient-descent/index.html#momentum

- Common metric for error function is sum of squared errors (SSE)
- The square function is nice because:
  - It keeps all values positive
  - It penalizes large values with large increases in the overall error

- Gradient is just a vector of slopes
- Backprop is nothing more than the chain rule applied iteratively.
- Epocs is the common term used for iterations of training

- Basic list of numpy types:
  - https://docs.scipy.org/doc/numpy/user/basics.types

---

- Partial derivative of the error with respect to each of the weights.
- The sum of squared errors is what we are trying to minimize.
- The sum of squared errors is what we need partial derivatives for.
- Since larger inputs drive more error, we scale the weight proportional to this
term.
- The resulting gradient tells us the proportional effect of the term on the
output:
  - If there's a large gradeint (slope), then a change to the term will have a
    large effect on the output.
  - If there's a small gradient, then a change to the term will have a small
    effect on the output.
  - Adding this term into the delta adjustment will cause larger gradients to
    produce larger steps, effecting the change in the error more.
- Checkout the backprop example in this repo (backprop.py)

- To get the hidden unit error, we scale the overall error by the weight
connecting the ouput to the hidden unit. (how much of the error is the hidden
weight responsible for).

- Once you have the error, use this to calculate the gradient descent step
learning rate * ouput unit error * hidden unit activation values (activation
value is after it went through the sigmoid function).

- Backprop is nothing more than the chain rule applied with an iterative
approach - basically remembering previous derivaties.
- When backflowing the gradients, the starting point is the direction of the error
function derivative, then it goes back thourgh the weighted sums and sigmoid
applications, it's important to take into account both.
- The gradient backflow starts with an error as the initial derivate input. It
either looks to increase the network or decrease it by a step
- Always try to start with the simplest possible example first and ensure that you
complete understand it.
- Take time to rephase things in your own words after reading to ensure you really
understand it.

- Watch out for returning an flipped matrix into the error function. This
happened on the first project and caused everything to run slow and not work.
The fix was to include a transpose on the final result.
- Gradient is a vector of slopes.
- A general rule of thumb is for the hidden nodes to be the input and output
size combined, divided by two.
- For some great thoughts on how a model should look, reference submission 1
feedback located in this repository.

### Model Evaluation and Validation Notes

- It's important to split your dataset into three parts.
  - training set: used to actually train your model
  - validation set: used to verify during the training process
  - test set: only used at the very end after training to verify accuracy

- Two different types of models
  - Regression models: used to predict a numeric value.
  - Classification models: used to predict a label (ex yes/no)

- The golden rule is to never use the test set to actually train data.

- A confusion matrix is split into four quadrants:
  - model: positive, actual: positive (correct)
  - model: positive, actual: negative (incorrect)
  - model: negative, actual: positive (incorrect)
  - model: negative, actual: negative (correct)
- Accuracy is both correct categories added together over the total number of
data points.

- Mean absolute error is the square of differences from the regression line.

- R2 score is two regressions compared. Generally, your newly created regression
to a simpler linear regression.

- Overfitting is too specific and over complicated
  - Error due to variance
  - Good on training set, bad on testing set
- Underfitting is too general and not complicated enough
  - Error due to bias
  - Bad on both training and testing set
- Good fit is good on both the training and testing sets

- Regression models can have different levels of complexity
  - 1 degree = linear
  - 2 degree = quadratic
  - 6 degree = polynomial

- The goal of training a model is to find where the training and testing errors
are the closest.

- K-Fold Cross Validation is where you break apart your data set into K number
of bins. You then select one bin to be a testing bin and use the rest as
training.  You can then repeat the previous process with different testing bins
to fully utilize your dataset.

### Sentiment Analysis Notes

- Sentiment analysis is basically identifying subjective information in text.
  - Does the text have a positive tone or negative?

- One way to approach something like sentiment analysis is to look at the
problem and think how would you approach it if you had to categorize the data
points manually yourself?  Then try to understand what you are doing manually to
help drive an automated approach.
  - Look for what's creating a correlation between the training input and target
    data sets.

- An example of this is when predicting whether a sentence has a positive or
negative tone.  If you use the entire sentence for training, it might be great
for predicting that exact sentce, but fail with sentences that have the same
meaning just with different words.  If you use the specific letters, it might
not give you much because both negative and positive words can easily have the
same letters.  So it's best to land on using words.  Determining how to frame
the problem is an important step.

- Given a set of movie reviews with a label of positive or negative, how might you
start approaching the problem?

- Break each review apart by words, then keep track of word counts of words
found in positive and negative reviews. Once you have the word counts, use log
on both the pos and neg counts for a word and subtract log(pos) from log(neg).
This will give you a decent comparison of words that are primarily found in
negative or positive reviews.

- After I did the steps above, I found that a number of actors names were in the
top results, so I introduced a minimum total count metric which required a word
to appear atleast 300 times before considering it a meaningful word.

```
pos['edie'], neg['edie']
np.log(1000) - np.log(10)
np.log(10000) - np.log(9000)
```

- This is a good way to initialize inputs with numpy: `np.zeros((1, length))`

- "no non-linearity in hidden layer"
  - This means to have the hidden layer not use an activation function. Instead
    just use f(x) = x.

- This is an interesting two sided relationship to keep in mind:
  - Weights control how much an input effects the hidden layer.
  - Inputs control how much weights effect a hidden layer.

- Neural Networks are basically all about weighting inputs
  - (Complicated regression models)

- One way to use words as an input to a neural network is to create a fixed size
vector where each indicy always maps to a single word, then set the indicies to
0 or 1 depending on if the word appears in a sentence or not.
  - Always stay trying to think about how to view things numerically

- Take time to frame the problem best:
	- Always strive to increase the signal and decrease the noise of training data
	- Make the problem as clear as possible to the neural network

- We made two interesting performance improvements to our neural network:
- Since our input nodes are always either 1 or 0, we can update the interaction
between the input and hidden layer to take advantage of the mathemtical
properties of 1 and 0.
  - If the node is 1, anything times 1 is itself, so just use the weight values.
  - If the node is 0, anything times 0 is 0, so just use 0.

- Had a nasty issue that took a while to debug:
  - I was using a list instead of a set for index values that should have been
    unique. Pay attention to this moving forward.

- Here are the following changes for the 1 and 0 optimization mentioned above:

```
# layer_1 = self.layer_0.dot(self.weights_0_1)
self.layer_1 *= 0
for index in self.layer_0_indexes:
	self.layer_1 += self.weights_0_1[index]

# self.weights_0_1 -= self.layer_0.T.dot(layer_1_delta) * self.learning_rate # update input-to-hidden weights with gradient descent step
for index in self.layer_0_indexes:
  self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate
```

- Another issue to watch out for is when things slow down during matrix operations. Ensure that you are using the correct size and dimension matrixes.

- Computers can't understand characters, so we use asicc to encode our
characters as numbers.  Similarly, they can't understand setences.  For Neural
Networks, you can encode sentences with numbers by having a dictionary of words
with a number assigned to each, then creating an array for the sentence
referncing the numer for each word.

- Word embeddings attempt to define words as more than just a single number, and
add multiple numbers to provide additional context.
- SkipGram is a neural network that takes a word and tries to predict the words
around it. This allows the neural network to start learning the abstract meaning
of words.

- Intelligence is just information processing.

### TFLearn Notes

```
net = tflearn.input_data([None, 784]) # image falttened into array
# ReLU - rectified linear unit - basically max(0, x)
net = tflearn.fully_connected(net, 200, activation='ReLU')
net = tflearn.fully_connected(net, 10, activation='softmax')
# optimizer: stochastic gradient descent, error function categorical cross-entropy
net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')
```

### LSTM, RNN, CNN Notes

- Great article: http://colah.github.io/posts/2015-08-Understanding-LSTMs/

- RNN are basically networks that can pass an input into the next iteration of the
network.
- The hidden result in the network is basically an input to the next iteration
hidden input.
- RNNs struggle to relate datapoints that occur far apart.

- LSTMs are a special kind of RNN that don't have trouble relating far apart data
points.
- CNNs can recognize patterns across space and can learn to combine the learned
patterns.
- RNNs can recognize patterns across time and can learn to combine the learned
patterns.

- CNNs look for the same pattern across the various subfields

- RNNs feed hidden layers from the previous step as an additional input to the
next step.  This is how RNNs build up memory.

- It's possible to run something like an RNN over an image and treat the time
aspect as space on the image.

- The deepmind atari playing CNN didn't have memory, it just looked at the current
pixels and determined what the best possible next step was.

- CNNs match smaller parts of an image - features of an image

- Convultion is a proces of repatedily applying a feature using a filtering
algorithm across an entire image.

- CNN features are generally 2x2 pixels

- Great video for CNNS https://www.youtube.com/watch?v=FmpDIaiMIeA&t=111s

- CNNs can also be used with sound where the x and y are time and intensity in
each frequency band.

- CNNs can also be used with text where the x and y are position in sentence and
words in the dictionary.

- CNNs are really only useful for spatial data that has local closely related
patterns. Does the position of the data matter in the x y grid? If not, then CNN
is probably not useful.

### TensorFlow Notes

- tf.constant and tf.placeholder are both immutable tensors
- tensorflow is built around the idea of a computational graph.
- tensorflow sessions are used to run computational graphs.

- You can can session.run multiple times - an example of this is when initalizing
globabl variables and then running the actual computation graph itself.

- tf.placeholders let the graph be more reusable because you can pass any input to
take place of the placeholder at runtime.

- Floating point math can cause issues when adding lots of really large numebrs
with really small numbers.
- To avoid this, it's best to normalize the incoming dataset by having all the
inputs have 0 mean and equal variance.

- What does a large sigma mean and why would it cause the function to have high
peaks?
- Small sigma would mean that your distribution is very uncretain about things?

- To get 0 mean and equal variance with image data that has values between 0 and
255, we take each value and subtract 128, then divide by 128 ((x - 128) / 128).

- Most machine learning is about designing the right loss function to optimize.

- Stoachistic gradient descent works by selecting a small number of resulsts to
correct the average loss on instead of calculating the average loss on the
entire dataset. instead of opimtizing the weights and bias based on the entire
set of data, chose a small random set to use instead.

- Because SGD takes you in slightly random directions and isn't as guarnteed as
just gradient descent, there are a few ways to help optimize it:
  - momentum: take a running average of the gradients and use that to guide the
    direction.
  - learing rate decay: as the process continues, take smaller and smaller steps.

- Golden rule: when things don't work, always try lowering your learning rate
first.

- Know the following:
  - Byte, Bit, Float, Int, max range for all types, etc.

- Usually, you run the entire set of data through the neural network as one
computation.
- If the entire set is too large for your computer's memory, you can use mini
batching and sgd, which is where you basically break apart your entire set into
small batches. you try to have all batches of equal size and possibly one that's
not.

- Preprocessing the data is normally an important step.
- Preprocessing can include cleaning, transforming, and reducing.
- This is where you'd update the values to have a mean of 0 and equal small
variance.
- It's also where you can fill in missing values with the set average or remove
large outliers.
- The PCA algorithm can be used to help visualize high dimensional data sets by
transforming them to lower dimensions.

```
# 0 dimensional int32 tensor
tf.constant(1)
# 1 dimensional int32 tensor
tf.constant([1, 2, 3])
# 2 dimensional int32 tensor
tf.constant([[1, 2, 3], [4, 5, 6]])
```

- During the tensorflow non minst lab, I found that lower epocs and higher
learning rate worked better for the particular dataset. Ended up going with 1
epoch and 0.1 learning rate.

- Feature scaling approaches: https://en.wikipedia.org/wiki/Feature_scaling

### Deep Neural Networks

- Logits are short for logistic regression.
- ReLU (Rectified linear units) max(0, x) - basically zero out negative weights.
- Logistic classifier is the same as a logistic regression.

- When building NNs, deeper is generally better than wider.
- Adding depth over width can be more performant
- If the problem you are trying to model has an underlying deep hierarchical structure, then
deeper is better.
- An example of why deeper is better is during CNN image recogition, the first
layer generally finds edges and lines, the second layer parital shapes, and by
the third layer you start seeing objects.

- Tensorflow provides a saver class to help save checkpoints when training a
model. It can also save the computation graph.

- Tensorflow uses a string identifier for tensors and operations.

- Tensorflow variables allows you to pass in a name which can help avoid errors while
restoring pervious weights after making adjustments to the model.

- To prevent your network from over optimization, you can use early termination.
- Early termination is where you watch the growth rate of the loss function and
terminate as it slows.
- L2 regularization adds another term to the loss which penalizes large weights.
- L2 norm is the sum of the squares of the individual elements in a vector.
- L2 regularization is another way to prevent overfitting.

- Values that travel between layers are generally called activations.
- Dropout involves dropping random activation values flowing through your network
to 0.
- Dropout temporarily drops random units and their inputs and outputs from the network
- Dropout then forces your network to develop a sort of internal redudancy because
it can't rely on any one given activation to help achieve its goal.
- Only use dropout during training to help prevent overfitting.
- In addition to setting half of the activation values to 0, you should x2 the other non 0
activation values to keep the overall average similar.

- In Tensorflow, make sure not to apply dropout for the validation or test sets.

- See `examples/dropout_tensorflow.py` for example.

### Convolutional Networks

- Statistical invariance are facts that don't chang over time or space.
- The core idea of a convolutional network is to take an image with a width,
height, and depth, and tranform it through layers that shrink the width and
height, but increase the layers of depth.  Think of a telescoping animation.
- CNNs have several layers and each layer will capture a different level in the
hierarchy of objects (edegs, shapes, objects).
- In CNNs, the stride is the amount of pixels you move the filter around by.  A
stride of 2 will provide a new image half the size of the original image.
- CNNs treat relative location as an important factor.  If pixels are adjacent
to each other, they are grouped.  In a standard NN, this is not the case.  All
inputs are mapped to neurons and the relative location of the inputs means
nothing in the overall calulation.
- Filter Depth is the overall number of filters in an CNN.
- Multiple neurons can be helpful in the layers because a single layer might
have more than one interesting characteristic that you'd like to capture.
- CNNs aren't programmed to look for specific characteristics, they just learn
this overtime with forward and backward propogation.
- In CNNs, weights are shared across all of the various patches.  So if there is
a dog's face in the top right of the image or the bottom left, the same weights
will be applied.
- Padding can be used to adjust the size of the original image to better fit a
filter size.
- To calculate the neurons in a layer of an CNN:
  - W = input layer volume
  - F = filter volume (width * height * depth)
  - P = padding used on the input layer
  - Volume of next layer = (W-F+2P)/S+1
    - Height of next layer = (input_height - filter_height + 2P) / S + 1
- Depth of layer is equal to the total number of filters.
- With weight sharing, you use the same filter for an entire depth slice.
- Each single nueron represents a pattern that causes it to activate.

- Pooling: Let's you keep a stride of 1 but pool together local fields to
down sample the input image.  The max function is popular for this, it basically
grabs a single point and calculates the max values of all neighboring points.
  - Average pooling also works as an alternative to max pooling.
  - Pooling is used to decrease ouput size and prevent overfitting
  - Recently pooling has dropped out of popularity in favor of dropout.
  - It should also be noted that pooling represents a loss of infomation.
- 1x1 Convolutions: Add a 1x1 neural network after your convolution.
- Each layer of the convolution, you can decide what kind of filter you want -
pooling, 1x1, 3x3, etc.
- Inception Architecure: Use many layers. Specifically:
  - Average pooling following by 1x1
  - 1x1
  - 1x1 followed by a 3x3
  - 1x1 followed by a 5x5

- A typical CNN could have multiple fully connected layers, max pooling layers,
and convolution layers.

- CNNs learn the filters at the convolutional layers

- What does one hot encode mean?
- Lookup non linear relationship?

### Project 2

changing sizes did much more than adding layers or removing them.
input, conv_num_outputs, conv_ksize, conv_strides, pool_ksize,
pool_strides
input 32x32x3
don't always need to change shapes between, two layers can be the same
size.
depths were way too low at first
print(x) to get the shape of each layer
change back?

relationship between all of these numbers are important

couldn't get the convnet accuracy above 50% and the issue ended up being
entirely with how i was creating the weights. when i updated the weights to a
standard deviation of 0.1, everything started working correctly.

### Cheat Sheets

http://www.souravsengupta.com/cds2016/lectures/Savov_Notes.pdf
http://tutorial.math.lamar.edu/pdf/Calculus_Cheat_Sheet_All.pdf
http://web.mit.edu/~csvoss/Public/usabo/stats_handout.pdf

### Docker Notes

docker ps
docker exec -it ${container_id} /bin/bash
docker stop ${container_id}
docker run -p 8888:8888 -p 6006:6006 gcr.io/tensorflow/tensorflow:latest-py3

docker run -it gcr.io/tensorflow/tensorflow /bin/bash
newgrp docker

---

docker pull google/cloud-sdk
docker run -t -i --name gcloud-config google/cloud-sdk gcloud init
