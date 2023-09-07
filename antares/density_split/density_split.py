import numpy as np
from pyrecon import RealMesh
from pandas import qcut
import sys


class DensitySplit:
    def __init__(self, data_positions, boxsize=None, boxcenter=None,
        data_weights=None, randoms_positions=None, randoms_weights=None,
        cellsize=None, wrap=False, boxpad=1.5, nthreads=None):
        self.data_positions = data_positions
        self.randoms_positions = randoms_positions
        self.boxsize = boxsize
        self.boxcenter = boxcenter
        self.cellsize = cellsize
        self.boxpad = boxpad
        self.wrap = wrap
        self.nthreads = nthreads

        if data_weights is not None:
            self.data_weights = data_weights
        else:
            self.data_weights = np.ones(len(data_positions))

        if boxsize is None:
            if randoms_positions is None:
                raise ValueError(
                    'boxsize is set to None, but randoms were not provided.')
            if randoms_weights is None:
                self.randoms_weights = np.ones(len(randoms_positions))
            else:
                self.randoms_weights = randoms_weights


    def get_density_mesh(self, sampling_positions, smoothing_radius,
        check=False, ran_min=0.01):
        """
        Get the overdensity field.

        Parameters
        ----------
        smooth_radius : float
            Radius of the smoothing filter.
        sampling_positions : array_like
            Positions where the density field should be sampled.
        Returns
        -------
        density : array_like
            Density field at the sampling positions.
        """
        self.data_mesh = RealMesh(boxsize=self.boxsize, cellsize=self.cellsize,
                                  boxcenter=self.boxcenter, nthreads=self.nthreads,
                                  positions=self.randoms_positions, boxpad=self.boxpad)
        self.data_mesh.assign_cic(self.data_positions, wrap=self.wrap)
        self.data_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=True,)
        if self.boxsize is None:
            self.randoms_mesh = RealMesh(boxsize=self.boxsize, cellsize=self.cellsize,
                                         boxcenter=self.boxcenter, nthreads=self.nthreads,
                                         positions=self.randoms_positions, boxpad=self.boxpad)
            self.randoms_mesh.assign_cic(self.randoms_positions, wrap=self.wrap)
            if check:
                mask_nonzero = self.randoms_mesh.value > 0.
                nnonzero = mask_nonzero.sum()
                if nnonzero < 2: raise ValueError('Very few randoms.')
            self.randoms_mesh.smooth_gaussian(smoothing_radius, engine='fftw', save_wisdom=True)
            sum_data, sum_randoms = np.sum(self.data_mesh.value), np.sum(self.randoms_mesh.value)
            alpha = sum_data * 1. / sum_randoms
            self.delta_mesh = self.data_mesh - alpha * self.randoms_mesh
            threshold = ran_min * sum_randoms / len(self.randoms_positions)
            mask = self.randoms_mesh > threshold
            self.delta_mesh[mask] /= alpha * self.randoms_mesh[mask]
            self.delta_mesh[~mask] = 0.0
            del self.data_mesh
            del self.randoms_mesh
        else:
            self.delta_mesh = self.data_mesh / np.mean(self.data_mesh) - 1.
            del self.data_mesh
        self.delta = self.delta_mesh.read_cic(sampling_positions) 
        self.sampling_positions = sampling_positions
        return self.delta


    def get_quantiles(self, nquantiles, return_idx=False):
        """
        Get the quantiles of the density field.

        Parameters
        ----------
        nquantiles : int
            Number of quantiles.
        return_idx : bool, optional
            Whether to return index of the quantile of each query point.

        Returns
        -------
        quantiles : array_like
            Quantiles of the density field.
        quantiles_idx : array_like, optional
            Index of the quantile of each query point.
        """
        quantiles_idx = qcut(self.delta, nquantiles, labels=False)
        quantiles = []
        for i in range(nquantiles):
            quantiles.append(self.sampling_positions[quantiles_idx == i])
        self.quantiles = quantiles
        if return_idx:
            return quantiles, quantiles_idx
        return quantiles
