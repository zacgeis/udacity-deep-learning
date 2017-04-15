import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    pass

def relu_deriv(x):
    pass

# https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications
def matrix_sum_squared_error(actual, expected):
    actual_dimensions = get_matrix_dimensions(actual)
    expected_dimensions = get_matrix_dimensions(expected)
    same_rows = actual_dimensions[0] == expected_dimensions[0]
    same_columns = actual_dimensions[1] == expected_dimensions[1]
    assert same_rows and same_columns, 'incorrect dimensions for matrix sum squared error'
    result = 0
    for row in range(actual_dimensions[0]):
        for column in range(actual_dimensions[1]):
            result += math.pow(actual[row][column] - expected[row][column], 2)
    return result / 2

def get_matrix_dimensions(matrix):
    rows = len(matrix)
    columns = len(matrix[0])
    return (rows, columns)

def generate_zero_matrix(rows, columns):
    weights = [None] * rows
    for row in range(rows):
        weights[row] = [0] * columns
    return weights

def matrix_multiply(mat1, mat2):
    mat1_dimensions = get_matrix_dimensions(mat1)
    mat2_dimensions = get_matrix_dimensions(mat2)
    assert mat1_dimensions[1] == mat2_dimensions[0], 'incorrect dimensions for matrix multiplication'
    matrix_result = generate_zero_matrix(mat1_dimensions[0], mat2_dimensions[1])
    for mat1_row in range(mat1_dimensions[0]):
        for mat2_column in range(mat2_dimensions[1]):
            cell_result = 0
            for mat1_column in range(mat1_dimensions[1]):
                cell_result += mat1[mat1_row][mat1_column] * mat2[mat1_column][mat2_column]
            matrix_result[mat1_row][mat2_column] = cell_result
    return matrix_result

def matrix_transpose(matrix):
    dimensions = get_matrix_dimensions(matrix)
    matrix_result = generate_zero_matrix(dimensions[1], dimensions[0])
    for row in range(dimensions[0]):
        for column in range(dimensions[1]):
            matrix_result[column][row] = matrix[row][column]
    return matrix_result

def matrix_element_wise_multiply(mat1, mat2):
    return matrix_element_wise_apply_two_arg(mat1, mat2, lambda a, b: a * b)

def matrix_element_wise_subtract(mat1, mat2):
    return matrix_element_wise_apply_two_arg(mat1, mat2, lambda a, b: a - b)

def matrix_element_wise_add(mat1, mat2):
    return matrix_element_wise_apply_two_arg(mat1, mat2, lambda a, b: a + b)

def matrix_scalar(matrix, scalar):
    return matrix_element_wise_apply_one_arg(matrix, lambda x: scalar * x)

def matrix_element_wise_apply_one_arg(matrix, operation):
    dimensions = get_matrix_dimensions(matrix)
    matrix_result = generate_zero_matrix(dimensions[0], dimensions[1])
    for row in range(dimensions[0]):
        for column in range(dimensions[1]):
            matrix_result[row][column] = operation(matrix[row][column])
    return matrix_result

def matrix_element_wise_apply_two_arg(mat1, mat2, operation):
    mat1_dimensions = get_matrix_dimensions(mat1)
    mat2_dimensions = get_matrix_dimensions(mat2)
    same_rows = mat1_dimensions[0] == mat2_dimensions[0]
    same_columns = mat1_dimensions[1] == mat2_dimensions[1]
    assert same_rows and same_columns, 'incorrect dimensions for matrix apply two arg'
    matrix_result = generate_zero_matrix(mat1_dimensions[0], mat1_dimensions[1])
    for row in range(mat1_dimensions[0]):
        for column in range(mat1_dimensions[1]):
            matrix_result[row][column] = operation(mat1[row][column], mat2[row][column])
    return matrix_result

def print_matrix(matrix):
    for row in matrix:
        for num in row:
            print('{:10.8f}'.format(num), end=' ')
        print()

# add stddev?
def generate_random_weights(rows, columns):
    weights = generate_zero_matrix(rows, columns)
    for row in range(rows):
        for column in range(columns):
            weights[row][column] = (random.random() * 2) - 1
    return weights
