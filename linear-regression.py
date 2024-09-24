from numpy import *

def compute_error_for_line_given_points(b, m, points):
    #init error at 0
    total_err = 0

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        #difference and square

        total_err += (y - (m * x + b ))**2

    return total_err / float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, iterations):
    b = starting_b
    m = starting_m
    #gradient descent

    for i in range(iterations):
        #get more accurate b and m from using gradient descent
        b, m = step_gradient(b, m, array(points), learning_rate)

    return [b, m]


def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = len(points)

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]



        #compute partial derivative of error function
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

    #update b and m 
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return[new_b, new_m]

def run():

    #collect data
    points = genfromtxt('data.csv', delimiter=',')

    #define hyper parameters
    learning_rate = 0.0001

    #slope formula: y = mx + b
    initial_b = 0
    initial_m = 0
    iterations = 1000

    print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, iterations)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(iterations, b, m, compute_error_for_line_given_points(b, m, points)))

if __name__ == '__main__':
    run()