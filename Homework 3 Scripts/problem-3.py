import numpy as np
from matplotlib import pyplot as plt

def circle_fit(A):
    num_of_cols = np.shape(A)[1]
    ones = -np.ones(num_of_cols)
    A_tilde = np.column_stack(((2*np.transpose(A)), ones))
    b = []
    for col in range(np.shape(A)[1]):
        norm_squared = np.linalg.norm(A[:, col])**2
        b.append(norm_squared)
    y = np.matmul(np.linalg.inv(np.matmul(np.transpose(A_tilde), A_tilde)), np.matmul(np.transpose(A_tilde), b))
    x = y[0:np.shape(A)[0]]
    r = np.sqrt(np.linalg.norm(x)**2 - y[np.shape(A)[0]])
    print("The center of the fitted circle is:")
    print(x)
    print("The radius of the fitted circle is:")
    print(r)

A = np.array([[0, 0.5, 1, 1, 0], [0, 0, 0, 1, 1]])

circle_fit(A)

a = np.array([0, 0.5, 1, 1, 0])
b = np.array([0, 0, 0, 1, 1])

figure, axes = plt.subplots()
circle = plt.Circle((0.5, 0.541666667), 0.6782841915041545, color = 'r', fill = False)
axes.set_aspect(1)
axes.add_artist(circle)
plt.plot(a, b, 'k.')
plt.xlim([-4, 4])
plt.ylim([-4, 4])
plt.show()