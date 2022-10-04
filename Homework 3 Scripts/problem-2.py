import numpy as np
from matplotlib import pyplot as plt

np.random.seed(12395)

a = 0
b = 1
n = 30
point = (b - a) / (n - 1)

x = np.array([a + point * i for i in range(n)])

rand = np.random.normal(0, 1, 30)

y = np.array(2*np.square(x) - 3 * x + 1 + 0.05 * rand)
y = np.reshape(y, (30, 1))

# Need to construct A. The first column will correpond to c, the second column to b, and the third column to a.

col1 = np.ones(30)
col2 = x
col3 = np.square(x)
A = np.column_stack((col1, col2, col3))

# Provided A^T A is invertible, the solution to the least squares problem is given by the following:

ATA = np.matmul(A.transpose(), A)
ATA_inv = np.linalg.inv(ATA)
ATy = np.matmul(A.transpose(), y)
sol = np.matmul(ATA_inv, ATy)

print(sol)

# The constants returned are as follows:
# c = 1.03281328
# b = -3.12406987
# a = 2.10080233

f = int(sol[2])*(x**2) + int(sol[1])*x + int(sol[0])

plt.plot(x, y, 'k.', x, f, 'g')
plt.show()


