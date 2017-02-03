from __future__ import division

def f(x):
    return x ** 2

def f_prime(x):
    return 2 * x

iterations = 10
learning_rate = 0.01

weight = 3
input = 2
target = 10

###

for i in range(iterations):
    actual = f(weight * input)
    error = target - actual
    print "Actual: {0}, Error {1}, Weight: {2}".format(actual, error, weight)
    delta = f_prime(weight * input) * error
    weight_adjustment = (delta / input) * learning_rate # do we divide delat / input or multiply?
    weight = weight + weight_adjustment

### graph x^2, 36, 10, 12x - 36
### derive (w*x)^2
