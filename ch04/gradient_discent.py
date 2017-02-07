import numpy as np
from ch04.gradient import numerical_gradient

def gradient_discent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


def function_2(x):
    return x[0]**2 + x[1]**2


init_x = np.array([-3.0, 4.0])
print(gradient_discent(function_2, init_x=init_x, lr=1e-10, step_num=100))
print(gradient_discent(function_2, init_x=init_x, lr=0.01, step_num=100))
print(gradient_discent(function_2, init_x=init_x, lr=0.1, step_num=100))
print(gradient_discent(function_2, init_x=init_x, lr=1.0, step_num=100))
print(gradient_discent(function_2, init_x=init_x, lr=10.0, step_num=100))
