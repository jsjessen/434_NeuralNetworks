# http://mlwiki.org/index.php/Normal_Equation

import matplotlib.pylab as plt
import numpy as np

class Line:
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def calculate(self, x1):
        x2 = x1 * self.slope + self.intercept
        return x2

A = np.array([[1, 1], [1, 2], [1, 3]])
b = np.array([1, 2, 2])

x0, x1 = np.linalg.solve(A.T.dot(A), A.T.dot(b))
lsq = Line(x1, x0)

# figure
plt.scatter(A[:, 1], b, marker='x', color='black')

points = np.array([0.5, 3.5])
plt.plot(points, lsq.calculate(points))

plt.scatter(A[:, 1], lsq.calculate(A[:, 1]), marker='o', color='red')
plt.vlines(A[:, 1], b, lsq.calculate(A[:, 1]))

plt.show()

x = np.array([[x0], [x1]])
p = A.dot(x).reshape(-1)
e = p - b
print(p.dot(e))