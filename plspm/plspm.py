#!/usr/bin/python3
#
# Copyright (C) 2019 Google Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pandas as pd

import plspm.config as c
import plspm.inner_model as im
import plspm.inner_summary as pis
import plspm.outer_model as om
import plspm.weights as w
from plspm.bootstrap import Bootstrap
from plspm.estimator import Estimator
from plspm.scheme import Scheme
from plspm.unidimensionality import Unidimensionality


class Plspm:
    """Estimates path models with latent variables using partial least squares algorithm

    Create an instance of this class in order to estimate a path model using the partial least squares algorithm.
    When the algorithm has performed the calculations to create the estimate, you can then retrieve the inner and outer
    models, scores, the path coefficients, effects, and reliability indicators such as goodness-of-fit
    and uni-dimensionality. Bootstrapping results can also be retrieved if they were requested.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: c.Config,
        scheme: Scheme = Scheme.CENTROID,
        iterations: int = 100,
        tolerance: float = 0.000001,
        bootstrap: bool = False,
        bootstrap_iterations: int = 100,
        processes: int = 2,
    ):
        """Creates an instance of the path model calculator.

        :param data: A Pandas DataFrame containing the dataset to be analyzed
        :param config: An instance of :obj:`.config.Config`
        :param scheme:
            The inner weighting scheme to use: :attr:`.Scheme.CENTROID` (default),
            :attr:`.Scheme.FACTORIAL` or :attr:`.Scheme.PATH`
            (see documentation for :mod:`.scheme`)
        :param iterations:
            The maximum number of iterations to try to get the algorithm to converge
            (default and minimum 100).
        :param tolerance:
            The tolerance criterion for iterations (default 0.000001, must be >0)
        :param bootstrap:
            Whether to perform bootstrap validation (default is not to perform validation)
        :param bootstrap_iterations:
            The number of bootstrap samples to use if bootstrap validation is enabled
            (default and minimum 100)
        :param processes:
            The number of processes to use while bootstrapping
            (bootstrap_iterations must be a multiple of processes)
        :raises Exception:
            if the algorithm cannot converge, or if the requested configuration
            could not be calculated
        """

        if iterations < 100:
            iterations = 100
        assert tolerance > 0
        assert scheme in Scheme
        if bootstrap_iterations < 10:
            bootstrap_iterations = 100
        assert processes > 0
        assert bootstrap_iterations % processes == 0

        estimator = Estimator(config)
        filtered_data = config.filter(data)
        correction = np.sqrt(filtered_data.shape[0] / (filtered_data.shape[0] - 1))

        calculator = w.WeightsCalculatorFactory(
            config, iterations, tolerance, correction, scheme
        )
        final_data, scores, weights = estimator.estimate(calculator, filtered_data)
        config = estimator.config()

        self.__inner_model = im.InnerModel(config.path(), scores)
        self.__outer_model = om.OuterModel(
            final_data,
            scores,
            weights,
            config.odm(config.path()),
            self.__inner_model.r_squared(),
        )
        self.__inner_summary = pis.InnerSummary(
            config,
            self.__inner_model.r_squared(),
            self.__inner_model.r_squared_adj(),
            self.__outer_model.model(),
        )
        self.__unidimensionality = Unidimensionality(config, filtered_data, correction)
        self.__scores = scores
        self.__bootstrap = None
        if bootstrap:
            if filtered_data.shape[0] < 10:
                raise AttributeError(
                    "Bootstrapping could not be performed, at least 10 observations are required."
                )
            self.__bootstrap = Bootstrap(
                config,
                filtered_data,
                self.__inner_model,
                self.__outer_model,
                calculator,
                bootstrap_iterations,
                processes,
            )

    def scores(self) -> pd.DataFrame:
        """Gets the latent variable scores

        :returns:
            a DataFrame with the latent variable scores, with a column for each latent variable.
            The index is the same as the index of the data passed in.
        """
        return self.__scores

    def outer_model(self) -> pd.DataFrame:
        """Gets the outer model

        :returns:
            a DataFrame with columns for weight, loading, communality, and redundancy,
            and a row for each manifest variable
        """
        return self.__outer_model.model()

    def inner_model(self) -> pd.DataFrame:
        """
        Gets the inner model for the endogenous latent variables

        :returns:
            a DataFrame with a row for each latent variable with a direct path to it,
            and columns for estimate, std error, t, and p>|t|.
        """
        return self.__inner_model.inner_model()

    def path_coefficients(self) -> pd.DataFrame:
        """
        Gets the path coefficient matrix

        :returns:
            a DataFrame of similar form to the Path matrix passed into
            :class:`plspm.config.Config`, with the relevant path coefficients in each cell
        """
        return self.__inner_model.path_coefficients()

    def crossloadings(self) -> pd.DataFrame:
        """Gets the crossloadings

        :returns:
            a DataFrame with the latent variables as the columns and the
            manifest variables as the index
        """
        return self.__outer_model.crossloadings()

    def inner_summary(self) -> pd.DataFrame:
        """Gets a summary of the inner model

        :returns:
            a DataFrame with the latent variables as the index, and columns for
            latent variable type (Exogenous or Endogenous), R squared, block communality,
            mean redundancy, and AVE (average variance extracted)
        """
        return self.__inner_summary.summary()

    def goodness_of_fit(self) -> float:
        """Gets goodness-of-fit

        :returns:
            goodness-of-fit
        """
        return self.__inner_summary.goodness_of_fit()

    def effects(self) -> pd.DataFrame:
        """Gets direct, indirect, and total effects for each path

        :returns:
            a DataFrame with an entry in the index for every path in the model,
            and a column for direct, indirect, and total effects for the corresponding path.
        """
        return self.__inner_model.effects()

    def unidimensionality(self) -> pd.DataFrame:
        """
        Gets the results of checking the unidimensionality of blocks
        (only meaningful for reflective / mode A blocks)

        :returns:
            a DataFrame with the latent variables as the index, and columns for
            Cronbach's Alpha, Dillon-Goldstein Rho, and the eigenvalues of the
            first and second principal components.
        """
        return self.__unidimensionality.summary()

    def bootstrap(self) -> Bootstrap:
        """Gets the results of bootstrap validation, if requested

        :returns:
            an instance of :class:`.bootstrap.Bootstrap` which can be queried
            for bootstrapping results

        :raises Exception:
            if bootstrap validation was not requested or if there were
            insufficient (<10) observations
        """
        if self.__bootstrap is None:
            raise AttributeError(
                "To perform bootstrap validation, "
                + "set the parameter bootstrap to True when calling Plspm"
            )
        return self.__bootstrap
