import unittest
import learn

class TestLearn(unittest.TestCase):
    def test_matrix_multiply(self):
        a = [[1, 2, 3], [4, 5, 6]]
        b = [[7, 8], [9, 10], [11, 12]]
        expected = [[58, 64], [139, 154]]
        actual = learn.matrix_multiply(a, b)
        self.assertEqual(expected, actual)

    def test_matrix_element_wise_apply_one_arg(self):
        a = [[1, 2, 3], [4, 5, 6]]
        expected = [[2, 3, 4], [5, 6, 7]]
        add1 = lambda x: x + 1
        actual = learn.matrix_element_wise_apply_one_arg(a, add1)
        self.assertEqual(expected, actual)

    def test_matrix_element_wise_subtract(self):
        a = [[1, 2], [4, 5]]
        b = [[3, 2], [1, 7]]
        expected = [[-2, 0], [3, -2]]
        actual = learn.matrix_element_wise_subtract(a, b)
        self.assertEqual(expected, actual)

    def test_matrix_element_wise_add(self):
        a = [[1, 2], [4, 5]]
        b = [[3, 2], [1, 7]]
        expected = [[4, 4], [5, 12]]
        actual = learn.matrix_element_wise_add(a, b)
        self.assertEqual(expected, actual)

    def test_matrix_element_wise_multiply(self):
        a = [[1, 2], [4, 5]]
        b = [[3, 2], [1, 7]]
        expected = [[3, 4], [4, 35]]
        actual = learn.matrix_element_wise_multiply(a, b)
        self.assertEqual(expected, actual)

    def test_matrix_scalar(self):
        a = [[1, 2], [4, 5]]
        expected = [[2, 4], [8, 10]]
        actual = learn.matrix_scalar(a, 2)
        self.assertEqual(expected, actual)

    def test_matrix_transpose(self):
        a = [[1], [2]]
        expected = [[1, 2]]
        actual = learn.matrix_transpose(a)
        self.assertEqual(expected, actual)

    def test_matrix_sum_squared_error(self):
        a = [[1, 2], [4, 5]]
        b = [[3, 2], [1, 7]]
        expected = 8.5
        actual = learn.matrix_sum_squared_error(a, b)
        self.assertEqual(expected, actual)

    def test_single_layer_network(self):
        input_data = [
            [0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1],
        ]
        output_target = [
            [0],
            [0],
            [1],
            [1],
        ]

        learning_rate = 1.0
        # weights = learn.generate_random_weights(3, 1)
        weights = [[-0.49509813149263415], [0.9885298437461054], [-0.7954209155228866]]

        for i in range(100):
            hidden_sum = learn.matrix_multiply(input_data, weights)
            hidden_result = learn.matrix_element_wise_apply_one_arg(hidden_sum, learn.sigmoid)
            error = learn.matrix_element_wise_subtract(output_target, hidden_result)
            gradients = learn.matrix_element_wise_apply_one_arg(hidden_sum, learn.sigmoid_deriv)
            delta = learn.matrix_element_wise_multiply(gradients, error)
            delta = learn.matrix_scalar(delta, learning_rate)
            weight_updates = learn.matrix_multiply(learn.matrix_transpose(input_data), delta)
            weights = learn.matrix_element_wise_add(weight_updates, weights)

            # print('error', matrix_sum_squared_error(hidden_result, output_target))
            # print('error', error)
            # print('result', hidden_result)
            # print('grad', gradients)
            # print('delta', delta)
            # print('weights', weights)
            # print()

            if i == 1:
                self.assertGreater(learn.matrix_sum_squared_error(hidden_result, output_target), 0.5)
            if i == 99:
                self.assertLess(learn.matrix_sum_squared_error(hidden_result, output_target), 0.02)

    def test_multi_layer_network(self):
        print('test')
        # http://iamtrask.github.io/2015/07/12/basic-python-network/
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()
