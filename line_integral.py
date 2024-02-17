"""
This script animates the following elements:
- Some nice curve in R3
- A vector field, with field lines (nice and curvy)
- Slide bar to fragment the curve into k vectors
- Projection of vector field onto these k vector sections
- Display area to show flux of vector field along this curve
"""

import timeit
from main import *


# # Define curve parametrization
def y(t): return np.sin(t)
def x(t): return np.cos(t)
def z(t): return 0.5 * sin(1.5 * t) ** 2
# def y(t): return 10 * sin(t)
# def x(t): return 10 * cos(t)
# def z(t): return t * 0


# parameter space
t = np.linspace(0,  2 * np.pi, 10000)

# create curve object
curve = Curve(x, y, z, t)


# Define vector field which displays flow around parametrized curve
def F(x, y, z):
    return vector(-y, x, 0 * z)


D = 1000
# create field object
field = Field(
    F, domain=curve.get_coordinates(density=D)
)

# create some surface
r, theta = polar_points()
domain = polar_to_cartesian(r, theta)


def f(x, y):
    t = np.arccos(x)
    z = 0.5 * sin(1.5 * t) ** 2 + 1

    return z - (x**2 + y**2)


surface = Surface(
    function=f,
    domain=domain
)


curve.line_integral_scene(field, surface=surface, alpha=0.5)
