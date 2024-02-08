# import libraries
import warnings
import matplotlib.pyplot as plt
import numpy as np


def vector(i, j, k):
    return np.array([i, j, k], dtype=float)


def cartesian_points(domain_range=1.0, dim=3, n=5, singularities=[]):
    """
    Parameters:
    - domain_range (float): Range of the domain in each dimension.
    - dim (int): Number of dimensions
    - n (int): Number of points in each dimension

    Returns:
    - Tuple of meshgrid arrays representing sampled points.
    """
    vals = [np.linspace(-domain_range, domain_range, n) for i in range(dim)]

    for point in singularities:
        for sing, i in zip(point.flatten(), range(vals)):
            vals[i] = np.setdiff1d(vals[i], [sing])

    return np.meshgrid(*tuple(vals))


def polar_points(r_range=1, theta_range=2*np.pi, singularities=[]):
    """
    Parameters:
    - r_range (float): Range of the radial coordinate.
    - theta_range (float): Range of the angular coordinate in radians.
    - singularities (list of tuples): List of tuples representing singular points in (r, theta) coordinates.

    Returns:
    - Tuple of meshgrid arrays representing sampled points in polar coordinates.
    """
    # Generate radial and angular values
    rs = np.linspace(0, r_range, 100)
    thetas = np.linspace(0, theta_range, 100)

    # Create meshgrid for polar coordinates
    r_mesh, theta_mesh = np.meshgrid(rs, thetas)

    # Remove singularities
    for singularity in singularities:
        r_singular, theta_singular = singularity
        r_mesh = np.setdiff1d(r_mesh, [r_singular])
        theta_mesh = np.setdiff1d(theta_mesh, [theta_singular])

    return r_mesh, theta_mesh


def cartesian_to_polar(x, y):
    """
    Convert Cartesian coordinates (x, y) to polar coordinates (r, theta).

    Parameters:
    - x (float): x-coordinate.
    - y (float): y-coordinate.

    Returns:
    - Tuple (r, theta) representing polar coordinates.
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def polar_to_cartesian(r, theta):
    """
    Convert polar coordinates (r, theta) to Cartesian coordinates (x, y).

    Parameters:
    - r (float): Radial coordinate.
    - theta (float): Angular coordinate in radians.

    Returns:
    - Tuple (x, y) representing Cartesian coordinates.
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def sparse_domain(domain, density):
    """
    Thins out a given domain.

    - domain: numpy meshgrid
    - density: factor by which to reduce density of numpy meshgrid

    FIY: thanks to the help of GPT 3.5!
    """
    # Ensure that density is a positive integer
    density = max(int(density), 1)

    # Get the shape of the original domain
    shape = domain[0].shape

    # Create a 2D mask based on density
    mask = np.zeros(shape, dtype=bool)
    mask[::density, ::density] = True

    # Apply the mask to each axis of the domain
    sparse_domain = [axis[mask] for axis in domain]

    return sparse_domain


def find_singularities(field, domain):
    """
    Detect singularities in a 3D field function.

    Parameters:
    - field (callable): The 3D field function to analyze.
    - domain (tuple): Tuple of 1D arrays (x, y, z) representing the spatial domain.

    Returns:
    - singularities (list): List of vectors where singularities were detected.

    Notes:
    - Uses warnings to catch division by zero errors during field function evaluation.
    """

    singularities = []
    x_vals, y_vals, z_vals = domain

    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            for k in range(len(z_vals)):
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")

                    x, y, z = x_vals[i, j, k], y_vals[i, j, k], z_vals[i, j, k]

                    try:
                        # compute the result and see what happens...
                        result = field(x, y, z)

                        if any(np.isinf(val) for val in result.flatten()):
                            singularities.append(vector(x, y, z))

                    except RuntimeWarning:
                        singularities.append(vector(x, y, z))
    return singularities


class PlotUtils:
    def __init__(self) -> None:
        pass

    def plot(self):
        fig = plt.figure()

        # Create a 3D plot
        self.ax = fig.add_subplot(111, projection="3d")

        # Make x, y, z-axis pane transparent
        self.ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # Remove grid
        self.ax.grid(True)

        # set labels
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        return self.ax

    def show(self, title=None):
        self.ax.legend()
        if title is not None:
            self.ax.set_title(title)
        plt.show()


class Plot(PlotUtils):
    def __init__(self, elements, domain_range=1, n=5) -> None:
        super().__init__()

        self.elements = elements
        self.domain_range = domain_range
        self.n = n

        self.ax = self.plot()

    def visualize(self):
        for element in self.elements:
            element.visualize(ax=self.ax,
                              domain_range=self.domain_range,
                              n=self.n)


class Curve(PlotUtils):
    """
    Concept so far: a curve is described by three functions
    x = x(t), y = y(t), z = z(t) allowing for its parametrization
    """

    def __init__(self, x, y, z, t) -> None:
        """
        x, y, z: functions accepting one numerical variable
        t: some numpy array containing parameter range
        ax: matplotlib axes object
        """
        super().__init__()

        self.x = x(t)
        self.y = y(t)
        self.z = z(t)

        self.t = t
        self.ax = None

    def visualize(self, ax=None, domain_range=None, n=None, color="red"):
        """
        method that visualizes curve

        TODO: implement restriction of plotting to domain_range...
        """
        ax = ax or self.ax or self.plot()

        ax.plot(self.x, self.y, self.z, color=color, label="Curve")


class Surface(PlotUtils):
    def __init__(self, function, domain) -> None:
        """
        function: python function accepting two variables as input (x,y) and returning z
        """
        super().__init__()

        self.f = function
        self.x, self.y = domain
        self.ax = None

    def visualize(self, ax=None, domain_range=None, n=None):
        """
        Visualizes surface z = f(x, y) in R3.

        TODO: implement restriction of plotting to domain_range.
        """
        ax = ax or self.ax or self.plot()

        ax.plot_surface(
            self.x, self.y, self.f(self.x, self.y),
            color="blue", alpha=0.5
        )

        # ax.plot_wireframe(self.x, self.y, self.f(self.x, self.y),
        #                   rcount=15, ccount=5, color="black"
        #                   )


class ParamSurface(PlotUtils):
    def __init__(self, x, y, z, u, t) -> None:
        super().__init__()

        u, t = np.meshgrid(u, t)

        self.x = x(u, t)
        self.y = y(u, t)
        self.z = z(u, t)

        self.ax = None

    def get_domain(self, density):
        return sparse_domain((self.x, self.y, self.z), density)

    def visualize(self, ax=None, domain_range=None, n=None):
        ax = ax or self.ax or self.plot()
        ax.plot_surface(self.x, self.y, self.z)


class Boundary:
    """
    Class that calculates the boundary curve of a surface.
    """

    def __init__(self, surface: Surface) -> None:
        pass


class Shadow:
    """
    Class to obtain the shadow of a surface / curve
    """

    def __init__(self) -> None:
        pass


class Field(PlotUtils):
    def __init__(self, field, domain=None, domain_range=1, n=5, singularities=None) -> None:
        """
        field: python function(s) either defining a scalar field or a 3d-vector field.
        """
        super().__init__()

        self.field = field
        self.domain = domain
        self.domain_range = domain_range
        self.n = n
        self.singularities = singularities

        self.scalar = False
        self.ax = None

    def M(self, x, y, z):
        return self.field(x, y, z)[0]

    def N(self, x, y, z):
        return self.field(x, y, z)[1]

    def P(self, x, y, z):
        return self.field(x, y, z)[2]

    def curl(self, x, y, z):
        delta = 1e-8

        Py = (self.P(x, y + delta, z) - self.P(x, y, z)) / delta
        Nz = (self.N(x, y, z + delta) - self.N(x, y, z)) / delta

        Mz = (self.M(x, y, z + delta) - self.M(x, y, z)) / delta
        Px = (self.P(x + delta, y, z) - self.P(x, y, z)) / delta

        Nx = (self.N(x + delta, y, z) - self.N(x, y, z)) / delta
        My = (self.M(x, y + delta, z) - self.M(x, y, z)) / delta

        return np.array([Py - Nz, Mz - Px, Nx - My])

    # def divergence(self, x, y, z):

    def sample_points(self, domain_range, n):
        """
        Parameters:
        - domain_range (float): Range of the domain in each dimension.
        - n (int): Number of points in each dimension

        Returns:
        - Tuple of meshgrid arrays representing sampled points.
        """
        # Create an array of evenly spaced values along each dimension
        x_vals = np.linspace(-domain_range, domain_range, n)
        y_vals = np.linspace(-domain_range, domain_range, n)
        z_vals = np.linspace(-domain_range, domain_range, n)

        # preliminary domain
        domain = np.meshgrid(x_vals, y_vals, z_vals)

        if self.singularities is None:
            # if no singularities specified, perform control
            singularities = find_singularities(
                self.field, domain
            )

        for s in singularities:
            x, y, z = s.flatten()

            # remove singularities
            x_vals = np.setdiff1d(x_vals, [x])
            y_vals = np.setdiff1d(y_vals, [y])
            z_vals = np.setdiff1d(z_vals, [z])

        return np.meshgrid(x_vals, y_vals, z_vals)

    def get_domain_lims(self, domain, tolerance=0):
        """
        Returns limits (i.e., max and mins) of domain for each coordinate
        """
        x, y, z = (domain[0].flatten(),
                   domain[1].flatten(),
                   domain[2].flatten())

        def min_max(v): return (min(v)-tolerance, max(v)+tolerance)

        return (min_max(x), min_max(y), min_max(z))

    def visualize(self, ax=None, domain_range=2, n=5, fieldlines=True):
        """
        A method that visualizes the vector field in R3.

        Parameters
        - domain_range (float): Range of the domain in each dimension.
        - num_points_per_dim (int): Number of points to sample along each dimension.
        """
        ax = ax or self.ax or self.plot()
        x, y, z = self.domain or self.sample_points(domain_range, n)

        values = self.field(x, y, z)

        # Vector field visualization
        ax.quiver(x, y, z, values[0], values[1], values[2],
                  colors="black", label="Vectors", length=0.4, normalize=True)

        if fieldlines:
            self.visualize_streamlines(ax=ax, domain_range=domain_range, n=n)

    def visualize_curl(self, domain_range=20.0, n=5):
        if self.ax is None:
            self.plot()

        x, y, z = self.sample_points(domain_range=20.0, n=5)
        curl = self.curl(x, y, z)

        self.ax.quiver(x, y, z, curl[0], curl[1], curl[2],
                       colors="blue", label="Curl", length=2, normalize=True)
        self.ax.legend()

    def streamlines(self, domain_range=2, n=5, delta=0.1, iter=10000, domain_lims=True):
        """
        This method computes streamlines.
        - iter (int): number of iterations, O(n^2)
        - delta (float): scalar multiple of vector steps - basically specifies granularity
                         of field lines
        - domain_lims (bool): specifies whether streamlines can extend beyond vector field
                              domains
        """
        seeds = self.domain or self.sample_points(domain_range, n)
        x_lim, y_lim, z_lim = self.get_domain_lims(seeds, tolerance=0.5)

        # list of tuples (x, y, z) where x, y, z specify coordinates of a streamline
        streamlines = []

        # compute streamlines
        for seed in zip(seeds[0].flatten(), seeds[1].flatten(), seeds[2].flatten()):
            x, y, z = [], [], []

            # forwards
            v0 = seed

            for _ in range(iter):
                x_coord, y_coord, z_coord = v0
                x.append(x_coord)
                y.append(y_coord)
                z.append(z_coord)

                v0 = v0 + self.field(*v0) * delta

                # check if streamlines are still within domain limits
                if not all(lim[0] <= coord <= lim[1] for lim, coord in zip((x_lim, y_lim, z_lim), (x_coord, y_coord, z_coord))):
                    break

            # backwards
            v0 = seed

            for _ in range(n):
                x_coord, y_coord, z_coord = v0
                x.append(x_coord)
                y.append(y_coord)
                z.append(z_coord)

                v0 = v0 - self.field(*v0) * delta

                # check if streamlines are still within domain limits
                if not all(lim[0] <= coord <= lim[1] for lim, coord in zip((x_lim, y_lim, z_lim), (x_coord, y_coord, z_coord))):
                    break

            streamlines.append((x, y, z))

        return streamlines

    def visualize_streamlines(self, ax=None, domain_range=2, n=5, color="black", linestyle="--", linewidth=0.9):
        ax = ax or self.ax or self.plot()

        streamlines = self.streamlines(domain_range, n)

        for streamline in streamlines:
            ax.plot(*streamline,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth)
