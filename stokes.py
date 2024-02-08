import numpy as np

from main import *


def x(t): return np.cos(t)  # define functions to parametrize a circle at z = 0
def y(t): return np.sin(t)
def z(t): return 0


curve = Curve(
    x, y, z,
    t=np.linspace(0, 2 * np.pi, 1000)
)

################


def f(x, y):  # define surface
    return 1 - (x**2 + y**2)


x, y = polar_to_cartesian(
    *polar_points()
)

surface = Surface(
    f,
    (x, y)
)
################


def F(x, y, z):  # define a vector field
    return vector(x, y, z)


field = Field(F, domain=sparse_domain((x, y, f(x, y)), 20))

################
sin = np.sin
cos = np.cos


def vertical(x, y, z):
    """
    Vertical vector field
    """
    return vector(0, 0, 1)


def x(u, t): return (1 + u * cos(t)) * cos(2 * t)
def y(u, t): return (1 + u * cos(t)) * sin(2 * t)
def z(u, t): return u * sin(t)


mobius = ParamSurface(
    x, y, z,
    np.linspace(-0.5, 0.5, 100),
    np.linspace(0, np.pi, 100)
)

boundary = Curve(
    x=lambda t: x(0.5, t),
    y=lambda t: y(0.5, t),
    z=lambda t: z(0.5, t),
    t=np.linspace(0, 2*np.pi, 100)
)


verticalF = Field(vertical,
                  mobius.get_domain(10))

mobius_plot = Plot(
    [mobius, boundary, verticalF]
)

mobius_plot.visualize()
mobius_plot.show()

# plot = Plot([surface, curve, field])
# plot.visualize()
# plot.show()
