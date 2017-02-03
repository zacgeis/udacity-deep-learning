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


var errorOutputLayer = subtract(examples.output, results.outputResult);
var deltaOutputLayer = dot(results.outputSum.transform(activatePrime), errorOutputLayer);
var hiddenOutputChanges = scalar(multiply(deltaOutputLayer, results.hiddenResult.transpose()), learningRate);
var deltaHiddenLayer = dot(multiply(weights.hiddenOutput.transpose(), deltaOutputLayer), results.hiddenSum.transform(activatePrime));
var inputHiddenChanges = scalar(multiply(deltaHiddenLayer, examples.input.transpose()), learningRate);

errorOutput = targetResult - actualResult
deltaOutput = sigmoid_prime(outputSum) * errorOutput
hiddenOutputChanges = (deltaOutput * hiddenResult.transpose) * learningRate
deltaHidden = sigmoid_prime(hiddenSum) * (hiddenOutput * deltaOutputLayer)
inputHiddenChanges = (deltaHidden * input.transpose) * learningRate
