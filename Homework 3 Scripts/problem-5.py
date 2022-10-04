import numpy as np
import pandas as pd

epsilon = 1/100000
max_iter = 1000

start1 = np.array([[-50], [7]])
start2 = np.array([[20], [7]])
start3 = np.array([[20], [-18]])
start4 = np.array([[5], [-10]])

starting_points = [start1, start2, start3, start4]

def f1(array):
    x1 = float(array[0])
    x2 = float(array[1])
    result = -13 + x1 + ((5 - x2) * x2 - 2) * x2
    return result

def f2(array):
    x1 = float(array[0])
    x2 = float(array[1])
    result = -29 + x1 + ((x2 + 1) * x2 - 14) * x2
    return result

def f(array):
    result = f1(array) ** 2 + f2(array) ** 2
    return result

def gradf(array):
    x1 = float(array[0])
    x2 = float(array[1])
    comp1 = f1(array) + f2(array)
    comp2 = f1(array) * (10 * x2 - 3 * (x2**2) - 2) + f2(array) * (3 * (x2**2) + 2 * x2 - 14)
    result = 2 * np.array([[comp1], [comp2]])
    return result

def hessianf(array):
    x1 = float(array[0])
    x2 = float(array[1])    
    comp1 = 2
    comp2 = 12 * x2 - 16
    comp3 = 12 * x2 - 16
    comp4 = 12 * x1 + 30 * (x2 ** 4) - 80 * (x2 ** 3) + 12 * (x2 ** 2) - 240 * x2 + 12
    result = np.array([[comp1, comp2], [comp3, comp4]])
    return result

def backtracking_gradient(x0, epsilon, max_iter, s, alpha, beta):
    x = x0
    gradient = gradf(x)
    value = f(x)
    iter_count = 0
    while np.linalg.norm(gradf(x)) > epsilon and iter_count < max_iter:
        iter_count += 1
        t = s
        while value - f(x - t * gradf(x)) < -alpha * t * np.linalg.norm(gradf(x)) ** 2:
            t = beta * t
        x = x - t * gradient
        value = f(x)
        gradient = gradf(x)
        print("Iteration: " + str(iter_count) + "\nValue: " + str(value) + "\n ||grad(f)|| = " + str(np.linalg.norm(gradf(x))))
    if iter_count >= max_iter:
        print("The algorithm did not converge before the maximum number of iterations. Last function value: " + str(value))
    else:
        print("Algorithm converged. \n Total Iterations: " + str(iter_count) + "\nFinal function value: " + str(value) + "\n Optimal Solution:")
        print(x)
    return [float(x[0]), float(x[1]), value, iter_count]

def hybrid_newton(x0, alpha, beta, epsilon, max_iter):
    x = x0
    value = f(x)
    iter_count = 0
    while np.linalg.norm(gradf(x)) > epsilon and iter_count < max_iter:
        iter_count += 1
        hessian_eigs = np.linalg.eigvals(hessianf(x))
        non_pos_def_flag = 0
        for i in hessian_eigs:
            if i <= 0:
                non_pos_def_flag += 1
        if non_pos_def_flag == 0:
            d = -np.matmul(np.linalg.inv(hessianf(x)), gradf(x))
        else:
            d = -gradf(x)
        t = 1
        while value - f(x + t * d) < -alpha * t * np.matmul(np.transpose(gradf(x)), d):
            t = beta * t
        x = x + t * d
        value = f(x)
        print("Iteration: " + str(iter_count) + "\nValue: " + str(value) + "\n ||grad(f)|| = " + str(np.linalg.norm(gradf(x))))
    if iter_count >= max_iter:
        print("The algorithm did not converge before the maximum number of iterations. Last function value: " + str(value))
    else:
        print("Algorithm converged. \n Total Iterations: " + str(iter_count) + "Final function value: " + str(value) + "\nOptimal Solution: ")
        print(x)
    return [float(x[0]), float(x[1]), value, iter_count]

def damped_newton_backtrack(x0, alpha, beta, s, epsilon, max_iter):
    x = x0
    value = f(x)
    iter_count = 0
    while np.linalg.norm(gradf(x)) > epsilon and iter_count < max_iter:
        iter_count += 1
        d = -np.matmul(np.linalg.inv(hessianf(x)), gradf(x))
        t = s
        while value - f(x + t * d) < -alpha * t * np.matmul(np.transpose(gradf(x)), d):
            t = beta * t
        x = x + t * d 
        print("Iteration: " + str(iter_count) + "\nValue: " + str(value) + "\n ||grad(f)|| = " + str(np.linalg.norm(gradf(x))))
    if iter_count >= max_iter:
        print("The algorithm did not converge before the maximum number of iterations. \nLast function value: " + str(value) + "\n Last Hessian:")
        print(hessianf(x))
        print("Last Iterate:")
        print(x)
    else:
        print("Algorithm converged. \n Total Iterations: " + str(iter_count) + "\nFinal function value: " + str(value) + "\nOptimal Solution: ")
        print(x)
    return [float(x[0]), float(x[1]), value, iter_count]

backtracking_gradient_results = {}
hybrid_newton_results = {}
damped_newton_backtrack_results = {}

for point in starting_points:
    backtracking_gradient_results[str(point[0]) + ", " + str(point[1])] = backtracking_gradient(point, epsilon, 10000, 1, 0.5, 0.5)

bg_results = pd.DataFrame.from_dict(backtracking_gradient_results, orient = 'index', columns = ['Solution x1', 'Solution x2', 'Func Value', 'Iterations'])

for point in starting_points:
    hybrid_newton_results[str(point[0]) + ", " + str(point[1])] = hybrid_newton(point, 0.5, 0.5, epsilon, 1000)

hn_results = pd.DataFrame.from_dict(hybrid_newton_results, orient = 'index', columns = ['Solution x1', 'Solution x2', 'Func Value', 'Iterations'])

for point in starting_points:
    damped_newton_backtrack_results[str(point[0]) + ", " + str(point[1])] = damped_newton_backtrack(point, 0.5, 0.5, 1, epsilon, 1000)

dnb_results = pd.DataFrame.from_dict(damped_newton_backtrack_results, orient = 'index', columns = ['Solution x1', 'Solution x2', 'Func Value', 'Iterations'])

print("Backtracking Gradient Results: \n")
print(bg_results)

print("Hybrid Newton Results: \n")
print(hn_results)

print("Damped Gauss-Newton Method Results: \n")
print(dnb_results)


print("The backtracking gradient method takes thousands of iterations but converged using all four starting points. It did not always converge to the global minimum. The hybrid Newton method converged in far fewer iterations. It also did not always converge to the global minimum. For both of these methods, in the cases where the algorithm did not converge to the global minimum, it instead converged to the strict local minimum that we identified in part (i) of the problem.")
print("The Damped Gauss Newton Method does not converge from any of the starting points because the starting step size of 1 is too large. If we run the function again with s = 0.05, the algorithm converges:")
input("Hit enter to run the Damped Gauss Newton Method with s = 0.05...")

better_damped_newton_backtrack_results = {}
for point in starting_points:
    better_damped_newton_backtrack_results[str(point[0]) + ", " + str(point[1])] = damped_newton_backtrack(point, 0.5, 0.5, 0.05, epsilon, 1000)

print("Damped Gauss Newton Method Results w/ s = 0.05")
bdnb_results = pd.DataFrame.from_dict(better_damped_newton_backtrack_results, orient = 'index', columns = ['Solution x1', 'Solution x2', 'Func Value', 'Iterations'])
print(bdnb_results)
print("In this case, we have similar results in the sense that the algorithm converged from all four starting points. The number of iterations lies somewhere in between the hybrid Newton method and the backtracking gradient.")