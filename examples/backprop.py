step_size = 0.01

### simple single variable example

def z(x):
    return x ** 2

def z_prime(x):
    return 2 * x

def test(f, f_prime, input):
    print("f({0}) = {1}".format(input, f(input)))
    tmp = f_prime(input)
    print("df/dx({0}) = {1}".format(input, tmp))
    increased_input = input + tmp * step_size * 1
    print("f({0}) = {1}, with small increase".format(increased_input, f(increased_input)))
    decreased_input = input + tmp * step_size * -1
    print("f({0}) = {1}, with small decrease".format(decreased_input, f(decreased_input)))

print("- with positive input -")
test(z, z_prime, -4)
print("")

print("- with negative input -")
test(z, z_prime, 4)
print("")

### now with partial derivs and nested functions

def f(a, b):
    return a * b

def q(c, d):
    return c + d

# to take parital derivate, act like all other variables other than the "with respect to __" are a constant.
# z = f(x, y) = x^2 + xy + y^2
# dz/dx = 2x + y
# dz/dy = x + 2y

def f_prime_a(a, b):
    return b

def f_prime_b(a, b):
    return a

def q_prime_c(c, d):
    return 1

def q_prime_d(c, d):
    return 1

def sample(input1, input2, input3, initial_backflow_deriv):
    nested1 = q(input1, input2)
    result = f(nested1, input3)
    print("f({0}, {1}, {2}) = {3}".format(input1, input2, input3, result))

    # multiplication comes from the chain rule
    # backprop is nothing more than the chain rule applied iteratively
    # f(g(x))' = f'(g(x))*g'(x)
    # backflow basically ends up showing how much each input variable effects the overall output
    # (how much of an effect each part has on the overall system)
    # think of entire neural network as one big nested function
    backflow_input3 = f_prime_b(nested1, input3) * initial_backflow_deriv
    backflow_nested1 = f_prime_a(nested1, input3) * initial_backflow_deriv
    backflow_input2 = q_prime_c(input1, input2) * backflow_nested1
    backflow_input1 = q_prime_d(input1, input2) * backflow_nested1

    new_input1 = input1 + (backflow_input1 * step_size)
    new_input2 = input2 + (backflow_input2 * step_size)
    new_input3 = input3 + (backflow_input3 * step_size)

    print("{0} -> {1}".format(input1, new_input1))
    print("{0} -> {1}".format(input2, new_input2))
    print("{0} -> {1}".format(input3, new_input3))

    nested1 = q(new_input1, new_input2)
    result = f(nested1, new_input3)
    print("f({0}, {1}, {2}) = {3}, with initial deriv of {4}".format(new_input1, new_input2, new_input3, result, initial_backflow_deriv))
    print("")

sample(6.0, 5.0, 7.0, -1)
sample(6.0, 5.0, 7.0, 1)
