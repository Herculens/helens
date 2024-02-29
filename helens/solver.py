# Copyright (c) 2023, herculens developers and contributors
# Copyright (c) 2024, helens developers and contributors

__author__ = 'austinpeel', 'aymgal'


import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)


class LensEquationSolver(object):
    """Solver for the multiple lensed image positions of a source point.
    The coordinates grid is assumed to be Cartesian, i.e. aligned with the 
    image axes and equally spaced in along both directions.

    Parameters
    ----------
    grid_x : array-like
        2D array containing the image plane x coordinates 
        used to search for multiple images.
    grid_y : array-like
        2D array containing the image plane y coordinates 
        used to search for multiple images.
    ray_shooting_func :
        Function that takes as input three arguments:
        - x coordinate (array or scalar),
        - y coordinate (array or scalar),
        - parameters of the lens mass model in the right format.
        The function should return the ray-shooted coordinates: 
        beta = theta - alpha(theta), where beta is the deflected source 
        coordinate of theta, and alpha is the deflection field.

    """

    def __init__(self, grid_x, grid_y, ray_shooting_func):
        self._xcoords, self._ycoords = grid_x.flatten(), grid_y.flatten()
        self._pix_scl = abs(grid_x[0, 0] - grid_x[0, 1])
        self._num_pix = grid_x.size
        self._ray_shooting_func = ray_shooting_func

    def estimate_accuracy(self, niter, scale_factor, nsubdivisions):
        """Gives an estimate of the accuracy of predicted image positions"""
        return self._pix_scl * (scale_factor / 4**nsubdivisions)**(niter / 2.)

    def solve(self, beta, lens_params, 
              nsolutions=5, niter=5, scale_factor=2, nsubdivisions=1):
        """Solve the lens equation.

        Parameters
        ----------
        beta : jax array of shape (2,)
            Position of a point source in the source plane.
        lens_params : dict or list or array
            Parameters defining the lens mass model, that are passed
            to the function `ray_shooting_func`.
        nsolutions: int, optional
            Number of expected solutions (e.g. 5 for a quad including the
            central image)
        niter : int
            Number of iterations of the solver.
        scale_factor : float, optional
            Factor by which to scale the selected triangle areas at each iteration.
        nsubdivisions : int, optional
            Number of times to subdivide (into 4) the selected triangles at
            each iteration.

        Returns
        -------
        theta, beta : tuple of 2D jax arrays
            Image plane positions and their source plane counterparts are
            returned as arrays of shape (N, 2).

        """
        # Triangulate the image plane
        img_triangles = self._triangulate()

        # Compute source plane images of the image plane triangles
        src_triangles = self._source_plane_triangles(img_triangles, lens_params)

        # Retain only those image plane triangles whose source image contains the point source
        inds = self._indices_containing_point(src_triangles, beta)
        img_selection = img_triangles[jnp.where(inds, size=nsolutions)]

        for _ in range(niter):
            # Scale up triangles
            img_selection = self._scale_triangles(img_selection, scale_factor)

            # Subdivide each triangle into 4
            img_selection = self._subdivide_triangles(
                img_selection, nsubdivisions)

            # Ray-shoot subtriangles to the source plane
            src_selection = self._source_plane_triangles(
                img_selection, lens_params)

            # Select corresponding image plane triangles containing the source point
            inds = self._indices_containing_point(src_selection, beta)
            img_selection = img_selection[jnp.where(inds, size=nsolutions)]

        # Ray-shoot the final image plane triangles to the source plane
        src_selection = self._source_plane_triangles(img_selection, lens_params)

        return self._centroids(img_selection), self._centroids(src_selection)

    def shoot_rays(self, x, y, lens_params):
        return self._ray_shooting_func(x, y, lens_params)

    def _triangulate(self):
        """Triangulate the coordinates grid.

        Returns
        -------
        out : jax array of shape (2 * N, 3, 2)
            Vertices of 2 * N triangles, where each pixel is divided in half
            along the diagonal.

        """
        delta = 0.5 * self._pix_scl

        # Coordinates of the four corners of each pixel
        x_LL, y_LL = self._xcoords - delta, self._ycoords - delta
        x_LR, y_LR = self._xcoords + delta, self._ycoords - delta
        x_UL, y_UL = self._xcoords - delta, self._ycoords + delta
        x_UR, y_UR = self._xcoords + delta, self._ycoords + delta
        t1 = jnp.array([[x_LL, y_LL], [x_LR, y_LR], [
                       x_UL, y_UL]]).transpose(2, 0, 1)
        t2 = jnp.array([[x_LR, y_LR], [x_UR, y_UR], [
                       x_UL, y_UL]]).transpose(2, 0, 1)

        # Interleave arrays so that the two triangles corresponding to a pixel are adjacent
        triangles = jnp.column_stack((t1, t2))

        return triangles.reshape(2 * self._num_pix, 3, 2)

    def _source_plane_triangles(self, image_triangles, ray_shoot_params):
        """Source plane triangles corresponding to image plane counterparts.

        Parameters
        ----------
        image_triangles : jax array of shape (N, 3, 2)
            Vertices defining N triangles in the image plane.
        ray_shoot_params : dict
            Parameters defining the mass model.

        """
        # Unpack into (x, y) triangle vertex arrays
        n = len(image_triangles)
        theta1, theta2 = image_triangles.transpose(
            (2, 0, 1)).reshape((2, 3 * n))

        # Shoot vertices to the source plane
        beta1, beta2 = self.shoot_rays(theta1, theta2, ray_shoot_params)

        # Repack into an array of triangle vertices
        return jnp.vstack([beta1, beta2]).reshape((2, n, 3)).transpose((1, 2, 0))

    def _indices_containing_point(self, triangles, point):
        """Determine whether a point lies within a triangle.

        Points lying along a triangle's edges are not considered
        to be within in it.

        Parameters
        ----------
        triangles : jax array of shape (N, 3, 2)
            Vertices defining N triangles.
        point : jax array of shape (2,)
            Point to test.

        Returns
        -------
        bool : jax array of shape (N,)
            Whether each triangle contains the input point.

        """
        # Distances between each vertex and the input point
        delta = triangles - jnp.atleast_1d(point)

        sign1 = jnp.sign(jnp.cross(delta[:, 0], delta[:, 1]))
        sign2 = jnp.sign(jnp.cross(delta[:, 1], delta[:, 2]))
        sign3 = jnp.sign(jnp.cross(delta[:, 2], delta[:, 0]))
        return jnp.abs(sign1 + sign2 + sign3) == 3

    def _scale_triangles(self, triangles, scale_factor):
        """Scale triangles about their centroids.

        Parameters
        ----------
        triangles : jax array of shape (N, 3, 2)
            Vertices defining N triangles in the image plane.
        scale_factor : float
            Factor by which each triangle's area is scaled.

        """
        c = self._centroids(triangles)
        c = jnp.repeat(jnp.expand_dims(c, 1), repeats=3, axis=1)
        return c + scale_factor**0.5 * (triangles - c)

    def _subdivide_triangles(self, triangles, niter=1):
        """Divide a set of triangles into 4 congruent triangles.

        Parameters
        ----------
        triangles : jax array of shape (N, 3, 2)
            Vertices defining N triangles in the image plane.
        niter : int
            Number of times to subdivide each triangle.

        """
        v1, v2, v3 = triangles.transpose(1, 0, 2)
        v4 = 0.5 * (v1 + v2)
        v5 = 0.5 * (v2 + v3)
        v6 = 0.5 * (v3 + v1)
        t1 = [v1, v4, v6]
        t2 = [v4, v2, v5]
        t3 = [v6, v4, v5]
        t4 = [v6, v5, v3]
        subtriangles = jnp.column_stack((t1, t2, t3, t4)).transpose(1, 0, 2)

        for k in range(1, niter):
            v1, v2, v3 = subtriangles.transpose(1, 0, 2)
            v4 = 0.5 * (v1 + v2)
            v5 = 0.5 * (v2 + v3)
            v6 = 0.5 * (v3 + v1)
            t1 = [v1, v4, v6]
            t2 = [v4, v2, v5]
            t3 = [v6, v4, v5]
            t4 = [v6, v5, v3]
            subtriangles = jnp.column_stack(
                (t1, t2, t3, t4)).transpose(1, 0, 2)

        return subtriangles.reshape(4**niter * len(triangles), 3, 2)

    def _centroids(self, triangles):
        """The centroid positions of a set of triangles.

        Parameters
        ----------
        triangles : jax array of shape (N, 3, 2)
            Vertices defining N triangles in the image plane.

        """
        return triangles.sum(axis=1) / 3.

    def _signed_areas(self, triangles):
        """The signed area of a set of triangles.

        Parameters
        ----------
        triangles : jax array of shape (N, 3, 2)
            Vertices defining N triangles in the image plane.

        """
        side1 = triangles[:, 1] - triangles[:, 0]
        side2 = triangles[:, 2] - triangles[:, 1]
        return 0.5 * jnp.cross(side1, side2)

    