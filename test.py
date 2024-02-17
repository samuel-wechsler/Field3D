from main import *

# # Define curve parametrization


def y(t): return 1 * np.sin(t)
def x(t): return 1.8 * np.cos(t)
def z(t): return 0.5 * sin(1.5 * t) ** 2
# def y(t): return 10 * sin(t)
# def x(t): return 10 * cos(t)
# def z(t): return t * 0


# parameter space
t = np.linspace(0,  2 * np.pi, 10000)

# create curve object
curve = Curve(x, y, z, t)
shadow = Shadow()
shadow = shadow.calculate_shadow(curve)

shadow.visualize()
