import matplotlib.pyplot as plt
import numpy as np
from main import cartesian_points


def vector(i, j, k):
    return np.array([i, j, k], dtype=float)


def F(x, y, z):
    i = y**2
    j = x**2 + z
    k = y**3

    return vector(i, j, k)


def get_streamline(field, seed, n=1000):
    x, y, z = [], [], []

    v0 = seed
    delta = 0.01

    for _ in range(n):
        x.append(v0[0])
        y.append(v0[1])
        z.append(v0[2])

        # Fix here, pass v0 as arguments to field function
        v0 += delta * field(*v0)

    return x, y, z


def plot_streamline(ax, x, y, z):
    ax.plot(x, y, z)


def streamline(field):
    """
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.grid(True)

    x_vals, y_vals, z_vals = cartesian_points()

    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            for k in range(len(z_vals)):

                x, y, z = x_vals[i, j, k], y_vals[i, j, k], z_vals[i, j, k]
                seed = np.array([x, y, z])

                sl = get_streamline(field, seed)
                plot_streamline(ax, *sl)


# get_streamline(F, np.array([0.5, 0.5, 0.5]))  # Ensure seed is float array
streamline(F)
plt.show()
