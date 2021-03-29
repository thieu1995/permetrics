#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:07, 18/07/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy import max, round, sqrt, abs, mean, dot, divide, arctan, sum, any, median, log, var, std
from numpy import ndarray, array, isfinite, isnan, argsort, zeros, concatenate, diff, sign
from numpy import min, histogram, unique, where, logical_and


class Metrics:
    """
        https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
    """

    EPSILON = 1e-10

    def __init__(self, y_true=None, y_pred=None):
        """
        :param y_true:
        :param y_pred:
        """
        if (y_true is not None) and (y_pred is not None):
            self.y_true, self.y_pred, self.y_true_clean, self.y_pred_clean, self.onedim = self.__clean_data__(y_true, y_pred)

    def __clean_data__(self, y_true=None, y_pred=None):
        """
        Parameters
        ----------
        y_true :
        y_pred :

        Returns
            y_true: after remove all Nan and Inf values
            y_pred: after remove all Nan and Inf values
            y_true_clean: after remove all Nan, Inf and 0 values
            y_pred_clean: after remove all Nan, Inf and 0 values
            dim: number of dimension in y_true
        -------

        """
        if type(y_true) is ndarray and type(y_pred) is ndarray:
            y_true = y_true.astype('float64')
            y_pred = y_pred.astype('float64')  # x = x[~np.isnan(x)] can't remove if array is dtype object, only work with dtype float
            if y_true.ndim == 1 and y_pred.ndim == 1:
                ## Remove all Nan in y_pred
                y_true = y_true[~isnan(y_pred)]
                y_pred = y_pred[~isnan(y_pred)]
                ## Remove all Inf in y_pred
                y_true = y_true[isfinite(y_pred)]
                y_pred = y_pred[isfinite(y_pred)]

                y_true_clean = y_true[y_pred != 0]
                y_pred_clean = y_pred[y_pred != 0]
                return y_true, y_pred, y_true_clean, y_pred_clean, True
            else:
                if y_true.ndim == y_pred.ndim:
                    ## Remove all row with Nan in y_pred
                    y_true = y_true[~isnan(y_pred).any(axis=1)]
                    y_pred = y_pred[~isnan(y_pred).any(axis=1)]
                    ## Remove all row with Inf in y_pred
                    y_true = y_true[isfinite(y_pred).any(axis=1)]
                    y_pred = y_pred[isfinite(y_pred).any(axis=1)]

                    y_true_clean = y_true[~any(y_pred == 0, axis=1)]
                    y_pred_clean = y_pred[~any(y_pred == 0, axis=1)]
                    return y_true, y_pred, y_true_clean, y_pred_clean, False
                else:
                    print("=====Failed! y_true and y_pred need to have same number of dimensions=======")
                    exit(0)
        else:
            print("=====Failed! y_true and y_pred need to be a ndarray=======")
            exit(0)

    def __get_data__(self, clean:bool, kwargs:dict):
        if ('y_true' in kwargs) and ('y_pred' in kwargs):
            y_true, y_pred, y_true_clean, y_pred_clean, onedim = self.__clean_data__(kwargs['y_true'], kwargs['y_pred'])
        else:
            if hasattr(self, 'y_true') and hasattr(self, 'y_pred'):
                y_true, y_pred, y_true_clean, y_pred_clean, onedim = self.y_true, self.y_pred, self.y_true_clean, self.y_pred_clean, self.onedim
            else:
                print("Dataset Error: Need y_true and y_pred!!!")
                exit(0)
        if clean:
            y_true, y_pred = y_true_clean, y_pred_clean
        return y_true, y_pred, onedim

    def __multi_output_result__(self, result=None, multi_output=None, decimal=None):
        if multi_output is None:
            return round(mean(result), decimal)
        elif isinstance(multi_output, (tuple, list, set)):
            weights = array(multi_output)
            if self.y_true.ndim != len(weights):
                print("==========Failed! Multi-output weights has different length with y_true==============")
                exit(0)
            return round(dot(result, multi_output), decimal)
        elif isinstance(multi_output, ndarray) and len(multi_output) == self.y_true.ndim:
            return round(dot(result, multi_output), decimal)
        elif multi_output == "raw_values":
            return round(result, decimal)
        else:
            print("==========Failed! Multi-output not supported==============")
            exit(0)

    def explained_variance_score(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Explained Variance Score. Best possible score is 1.0, lower values are worse.
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            return round(1 - var(y_true - y_pred) / var(y_true), decimal)
        else:
            temp = 1 - var(y_true - y_pred, axis=0) / var(y_true, axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def max_error(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Max Error: Smaller is better
        :param clean: Pre-processing y_true and y_pred, no 0 values
        :param multi_output: string in [‘raw_values’, ‘uniform_average’, ‘variance_weighted’] or array-like of shape (n_outputs)
        :param decimal: the number after ","
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            return round(max(abs(y_true - y_pred)), decimal)
        else:
            temp = max(abs(y_true - y_pred), axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def mean_absolute_error(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Mean Absolute Error: Smaller is better
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            return round(sum(abs(y_pred - y_true)) / len(y_true), decimal)
        else:
            temp = sum(abs(y_pred - y_true), axis=0) / len(y_true)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def mean_squared_error(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Mean Squared Error: Smaller is better
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            return round(sum((y_pred - y_true)**2) / len(y_true), decimal)
        else:
            temp = sum((y_pred - y_true)**2, axis=0) / len(y_true)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def root_mean_squared_error(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Root Mean Squared Error: Smaller is better
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            return round(sqrt(sum((y_pred - y_true) ** 2) / len(y_true)), decimal)
        else:
            temp = sqrt(sum((y_pred - y_true) ** 2, axis=0) / len(y_true))
            return self.__multi_output_result__(temp, multi_output, decimal)


    def mean_squared_log_error(self, clean=True, multi_output="raw_values", decimal=3, **kwargs):
        """
            Mean Squared Log Error: Smaller is better
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            y_true = y_true[y_pred > 0]
            y_pred = y_pred[y_pred > 0]
            return round(sum(log(y_true / y_pred) ** 2) / len(y_true), decimal)
        else:
            y_true = y_true[any(y_pred > 0, axis=1)]
            y_pred = y_pred[any(y_pred > 0, axis=1)]
            temp = sum(log(y_true / y_pred) ** 2, axis=0) / len(y_true)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def median_absolute_error(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Median Absolute Error: Smaller is better
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            return round(median(abs(y_true - y_pred)), decimal)
        else:
            temp = median(abs(y_true - y_pred), axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def mean_relative_error(self, clean=True, multi_output="raw_values", decimal=3, **kwargs):
        """
            Mean Relative Error: Smaller is better
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            return round(mean(divide(abs(y_true - y_pred), y_true)), decimal)
        else:
            temp = mean(divide(abs(y_true - y_pred), y_true), axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def mean_absolute_percentage_error(self, clean=True, multi_output="raw_values", decimal=3, **kwargs):
        """
            Mean Absolute Percentage Error: Good if mape < 30%. Smaller is better
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            temp = abs(y_true - y_pred) / abs(y_true)
            return round(mean(temp), decimal)
        else:
            temp = mean( abs(y_true - y_pred) / abs(y_true), axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def symmetric_mean_absolute_percentage_error(self, clean=True, multi_output="raw_values", decimal=3, **kwargs):
        """
            Symmetric Mean Absolute Percentage Error: Good if mape < 20%. Smaller is better
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            return round(mean(2 * abs(y_pred - y_true) / (abs(y_true) + abs(y_pred))), decimal)
        else:
            temp = mean(2 * abs(y_pred - y_true) / (abs(y_true) + abs(y_pred)), axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def mean_arctangent_absolute_percentage_error(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Mean Arctangent Absolute Percentage Error (output: radian values)
            https://support.numxl.com/hc/en-us/articles/115001223463-MAAPE-Mean-Arctangent-Absolute-Percentage-Error
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            return round(mean(arctan(divide(abs(y_true - y_pred), y_true))), decimal)
        else:
            temp = mean(arctan(divide(abs(y_true - y_pred), y_true)), axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def mean_absolute_scaled_error(self, m=1, clean=True, multi_output="raw_values", decimal=3, **kwargs):
        """
            Mean Absolute Scaled Error (m = 1 for non-seasonal data, m > 1 for seasonal data)
            https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            return round(mean(abs(y_true - y_pred)) / mean(abs(y_true[m:] - y_true[:-m])), decimal)
        else:
            temp = mean(abs(y_true - y_pred), axis=0) / mean(abs(y_true[m:] - y_true[:-m]), axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def nash_sutcliffe_efficiency_coefficient(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Nash-Sutcliffe Efficiency Coefficient (-unlimited < NSE < 1.   Larger is better)
            https://agrimetsoft.com/calculators/Nash%20Sutcliffe%20model%20Efficiency%20coefficient
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            return round(1 - sum((y_true - y_pred) ** 2) / sum((y_true - mean(y_true)) ** 2), decimal)
        else:
            temp = 1 - sum((y_true - y_pred) ** 2, axis=0) / sum((y_true - mean(y_true, axis=0)) ** 2, axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)

    def willmott_index(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Willmott Index (Willmott, 1984, 0 < WI < 1. Larger is better)
            Reference evapotranspiration for Londrina, Paraná, Brazil: performance of different estimation methods
        https://www.researchgate.net/publication/319699360_Reference_evapotranspiration_for_Londrina_Parana_Brazil_performance_of_different_estimation_methods
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            m1 = mean(y_true)
            return round(1 - sum((y_pred - y_true) ** 2) / sum((abs(y_pred - m1) + abs(y_true - m1)) ** 2), decimal)
        else:
            m1 = mean(y_true, axis=0)
            temp = 1 - sum((y_pred - y_true) ** 2, axis=0) / sum((abs(y_pred - m1) + abs(y_true - m1)) ** 2, axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def pearson_correlation_index(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Pearson’s Correlation Index (Willmott, 1984): -1 < R < 1. Larger is better
            Reference evapotranspiration for Londrina, Paraná, Brazil: performance of different estimation methods
            Remember no absolute in the equations
            https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            m1, m2 = mean(y_true), mean(y_pred)
            temp = sum((y_true - m1) * (y_pred - m2)) / (sqrt(sum((y_true - m1) ** 2)) * sqrt(sum((y_pred - m2) ** 2)))
            return round(temp, decimal)
        else:
            m1, m2 = mean(y_true, axis=0), mean(y_pred, axis=0)
            t1 = sqrt(sum((y_true - m1) ** 2, axis=0))
            t2 = sqrt(sum((y_pred - m2) ** 2, axis=0))
            t3 = sum((y_true - m1) * (y_pred - m2), axis=0)
            temp = t3 / (t1 * t2)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def confidence_index(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Confidence Index (or Performance Index)
        https://www.researchgate.net/publication/319699360_Reference_evapotranspiration_for_Londrina_Parana_Brazil_performance_of_different_estimation_methods
        Reference evapotranspiration for Londrina, Paraná, Brazil: performance of different estimation methods
            > 0.85          Excellent
            0.76-0.85       Very good
            0.66-0.75       Good
            0.61-0.65       Satisfactory
            0.51-0.60       Poor
            0.41-0.50       Bad
            ≤ 0.40          Very bad
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        r = self.pearson_correlation_index(clean=clean, multi_output="raw_values", decimal=decimal, y_true=y_true, y_pred=y_pred)
        d = self.willmott_index(clean=clean, multi_output="raw_values", decimal=decimal, y_true=y_true, y_pred=y_pred)
        return self.__multi_output_result__(r * d, multi_output, decimal)


    def coefficient_of_determination(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            R2: Coefficient of Determination - regression score function. Best possible score is 1.0 and it can be negative
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            t1 = sum((y_true - y_pred)**2)
            t2 = sum((y_true - mean(y_true))**2)
            return round(1 - t1/t2, decimal)
        else:
            t1 = sum((y_true - y_pred) ** 2, axis=0)
            t2 = sum((y_true - mean(y_true, axis=0)) ** 2, axis=0)
            temp = 1 - t1 / t2
            return self.__multi_output_result__(temp, multi_output, decimal)

    def pearson_correlation_index_square(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            (Pearson’s Correlation Index)^2 = R2
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        temp = self.pearson_correlation_index(clean, "raw_values", decimal, y_true=y_true, y_pred=y_pred)
        return self.__multi_output_result__(temp**2, multi_output, decimal)

    def deviation_of_runoff_volume(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Deviation of Runoff Volume (DRV)
            https://rstudio-pubs-static.s3.amazonaws.com/433152_56d00c1e29724829bad5fc4fd8c8ebff.html
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            return round(sum(y_pred)/sum(y_true), decimal)
        else:
            temp = sum(y_pred, axis=0) / sum(y_true, axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)

    def kling_gupta_efficiency(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Kling-Gupta Efficiency (KGE)
            https://rstudio-pubs-static.s3.amazonaws.com/433152_56d00c1e29724829bad5fc4fd8c8ebff.html
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        r = self.pearson_correlation_index(clean, multi_output, decimal, y_true=y_true, y_pred=y_pred)
        if onedim:
            beta = mean(y_pred)/mean(y_true)
            gamma = (std(y_pred)/mean(y_pred))/(std(y_true)/mean(y_true))
            out = 1 - sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)
            return round(out, decimal)
        else:
            beta = mean(y_pred, axis=0) / mean(y_true, axis=0)
            gamma = (std(y_pred, axis=0) / mean(y_pred, axis=0)) / (std(y_true, axis=0) / mean(y_true, axis=0))
            out = 1 - sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
            return self.__multi_output_result__(out, multi_output, decimal)

    def gini_coefficient(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Gini coefficient (Gini)
            https://github.com/benhamner/Metrics/blob/master/MATLAB/metrics/gini.m
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            idx_sort = argsort(-y_pred)
            population_delta = 1.0 / len(y_true)
            accumulated_population_percentage_sum, accumulated_loss_percentage_sum, score = 0, 0, 0
            total_losses = sum(y_true)
            for i in range(0, len(y_true)):
                accumulated_loss_percentage_sum += y_true[idx_sort[i]] / total_losses
                accumulated_population_percentage_sum += population_delta
                score += accumulated_loss_percentage_sum - accumulated_population_percentage_sum
            score = score / len(y_true)
            return round(score, decimal)
        else:
            col = y_true.shape[1]
            idx_sort = argsort(-y_pred, axis=0)
            population_delta = 1.0 / len(y_true)
            accumulated_population_percentage_sum, accumulated_loss_percentage_sum, score = zeros(col), zeros(col), zeros(col)
            total_losses = sum(y_true, axis=0)
            for i in range(0, col):
                for j in range(0, len(y_true)):
                    accumulated_loss_percentage_sum[i] += y_true[idx_sort[j, i], i] / total_losses[i]
                    accumulated_population_percentage_sum[i] += population_delta
                    score[i] += accumulated_loss_percentage_sum[i] - accumulated_population_percentage_sum[i]
            score = score / len(y_true)
            return self.__multi_output_result__(score, multi_output, decimal)

    def gini_coefficient_wiki(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Gini coefficient (Gini)
            https://en.wikipedia.org/wiki/Gini_coefficient
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            y = concatenate((y_true, y_pred), axis=0)
            score = 0
            for i in range(0, len(y)):
                for j in range(0, len(y)):
                    score += abs(y[i] - y[j])
            y_mean = mean(y)
            score = score / (2*len(y)**2 * y_mean)
            return round(score, decimal)
        else:
            y = concatenate((y_true, y_pred), axis=0)
            col = y.shape[1]
            d = len(y)
            score = zeros(col)
            for k in range(0, col):
                for i in range(0, d):
                    for j in range(0, d):
                        score[k] += abs(y[i, k] - y[j, k])
            y_mean = mean(y, axis=0)
            score = score / (2 * len(y) ** 2 * y_mean)
            return self.__multi_output_result__(score, multi_output, decimal)

    def prediction_of_change_in_direction(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Prediction of change in direction
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            d = diff(y_true)
            dp = diff(y_pred)
            return round(mean(sign(d) == sign(dp)), decimal)
        else:
            d = diff(y_true, axis=0)
            dp = diff(y_pred, axis=0)
            score = mean(sign(d) == sign(dp), axis=0)
            return self.__multi_output_result__(score, multi_output, decimal)

    def entropy(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
            Entropy Loss function
            https://datascience.stackexchange.com/questions/20296/cross-entropy-loss-explanation
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            score = -sum(y_true * log(y_pred.clip(self.EPSILON, None)))
            return round(score, decimal)
        else:
            score = -sum(y_true * log(y_pred.clip(self.EPSILON, None)), axis=0)
            return self.__multi_output_result__(score, multi_output, decimal)

    def cross_entropy(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            f_true, intervals = histogram(y_true, bins=len(unique(y_true)) - 1)
            intervals[0] = min([min(y_true), min(y_pred)])
            intervals[-1] = max([max(y_true), max(y_pred)])
            f_true = f_true / len(f_true)
            f_pred = histogram(y_pred, bins=intervals)[0] / len(y_pred)
            value = self.entropy(clean, None, decimal, y_true=f_true, y_pred=f_pred)
            return round(value, decimal)
        else:
            score = []
            for i in range(y_true.shape[1]):
                f_true, intervals = histogram(y_true[:,i], bins=len(unique(y_true[:,i])) - 1)
                intervals[0] = min([min(y_true[:,i]), min(y_pred[:,i])])
                intervals[-1] = max([max(y_true[:,i]), max(y_pred[:,i])])
                f_true = f_true / len(f_true)
                f_pred = histogram(y_pred[:,i], bins=intervals)[0] / len(y_pred[:,i])
                score.append(self.entropy(clean, "raw_values", decimal, y_true=f_true, y_pred=f_pred))
            return self.__multi_output_result__(score, multi_output, decimal)

    def kullback_leibler_divergence(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
        Kullback-Leibler Divergence
        https://machinelearningmastery.com/divergence-between-probability-distributions/
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            f_true, intervals = histogram(y_true, bins=len(unique(y_true)) - 1)
            intervals[0] = min([min(y_true), min(y_pred)])
            intervals[-1] = max([max(y_true), max(y_pred)])
            f_true = f_true / len(f_true)
            f_pred = histogram(y_pred, bins=intervals)[0] / len(y_pred)
            score = self.entropy(clean, None, decimal, y_true=f_true, y_pred=f_pred) - \
                    self.entropy(clean, None, decimal, y_true=f_true, y_pred=f_true)
            return round(score, decimal)
        else:
            score = []
            for i in range(y_true.shape[1]):
                f_true, intervals = histogram(y_true[:, i], bins=len(unique(y_true[:, i])) - 1)
                intervals[0] = min([min(y_true[:, i]), min(y_pred[:, i])])
                intervals[-1] = max([max(y_true[:, i]), max(y_pred[:, i])])
                f_true = f_true / len(f_true)
                f_pred = histogram(y_pred[:, i], bins=intervals)[0] / len(y_pred[:, i])
                temp1 = self.entropy(clean, None, decimal, y_true=f_true, y_pred=f_pred)
                temp2 = self.entropy(clean, None, decimal, y_true=f_true, y_pred=f_true)
                score.append(temp1 - temp2)
            return self.__multi_output_result__(score, multi_output, decimal)

    def jensen_shannon_divergence(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
        Jensen-Shannon Divergence
        https://machinelearningmastery.com/divergence-between-probability-distributions/
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            f_true, intervals = histogram(y_true, bins=len(unique(y_true)) - 1)
            intervals[0] = min([min(y_true), min(y_pred)])
            intervals[-1] = max([max(y_true), max(y_pred)])
            f_true = f_true / len(f_true)
            f_pred = histogram(y_pred, bins=intervals)[0] / len(y_pred)
            m = 0.5 * (f_true + f_pred)
            temp1 = self.entropy(clean, None, decimal, y_true=f_true, y_pred=m) - \
                    self.entropy(clean, None, decimal, y_true=f_true, y_pred=f_true)
            temp2 = self.entropy(clean, None, decimal, y_true=f_pred, y_pred=m) - \
                    self.entropy(clean, None, decimal, y_true=f_pred, y_pred=f_pred)
            return round(0.5 * temp1 + 0.5 * temp2, decimal)
        else:
            score = []
            for i in range(y_true.shape[1]):
                f_true, intervals = histogram(y_true[:, i], bins=len(unique(y_true[:, i])) - 1)
                intervals[0] = min([min(y_true[:, i]), min(y_pred[:, i])])
                intervals[-1] = max([max(y_true[:, i]), max(y_pred[:, i])])
                f_true = f_true / len(f_true)
                f_pred = histogram(y_pred[:, i], bins=intervals)[0] / len(y_pred[:, i])
                m = 0.5 * (f_true + f_pred)
                temp1 = self.entropy(clean, None, decimal, y_true=f_true, y_pred=m) - \
                        self.entropy(clean, None, decimal, y_true=f_true, y_pred=f_true)
                temp2 = self.entropy(clean, None, decimal, y_true=f_pred, y_pred=m) - \
                        self.entropy(clean, None, decimal, y_true=f_pred, y_pred=f_pred)
                score.append(0.5 * temp1 + 0.5 * temp2)
            return self.__multi_output_result__(score, multi_output, decimal)

    def variance_accounted_for(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
        Variance Accounted For between 2 signals
        https://www.dcsc.tudelft.nl/~jwvanwingerden/lti/doc/html/vaf.html
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            vaf = (1 - (y_true - y_pred).var()/y_true.var()) * 100
            return round(vaf, decimal)
        else:
            vaf = (1 - (y_true - y_pred).var(axis=0) / y_true.var(axis=0)) * 100
            return self.__multi_output_result__(vaf, multi_output, decimal)

    def relative_absolute_error(self, clean=False, multi_output="raw_values", decimal=3, **kwargs):
        """
        Relative Absolute Error
        https://stackoverflow.com/questions/59499222/how-to-make-a-function-of-mae-and-rae-without-using-librarymetrics
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            mean_true = mean(y_true)
            rae = sum(abs(y_true - y_pred)) / sum(abs(y_true - mean_true))
            return round(rae, decimal)
        else:
            mean_true = mean(y_true, axis=0)
            rae = sum(abs(y_true - y_pred), axis=0) / sum(abs(y_true - mean_true), axis=0)
            return self.__multi_output_result__(rae, multi_output, decimal)

    def a10_index(self, clean=True, multi_output="raw_values", decimal=3, **kwargs):
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            div = y_true / y_pred
            div = where(logical_and(div >= 0.9, div <=1.1), 1, 0)
            return round(mean(div), decimal)
        else:
            div = y_true / y_pred
            div = where(logical_and(div >= 0.9, div <= 1.1), 1, 0)
            return self.__multi_output_result__(mean(div, axis=0), multi_output, decimal)

    def a20_index(self, clean=True, multi_output="raw_values", decimal=3, **kwargs):
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            div = y_true / y_pred
            div = where(logical_and(div >= 0.8, div <= 1.2), 1, 0)
            return round(mean(div), decimal)
        else:
            div = y_true / y_pred
            div = where(logical_and(div >= 0.8, div <= 1.2), 1, 0)
            return self.__multi_output_result__(mean(div, axis=0), multi_output, decimal)

    def normalized_root_mean_square_error(self, clean=True, multi_output="raw_values", decimal=3, **kwargs):
        """
        Normalized Root Mean Square Error
        https://medium.com/microsoftazure/how-to-better-evaluate-the-goodness-of-fit-of-regressions-990dbf1c0091
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        rmse = self.root_mean_squared_error(clean, multi_output, decimal, y_true=y_true, y_pred=y_pred)
        if "model" in kwargs:
            model = kwargs["model"]
        else:
            model = 0
        if onedim:
            if model == 0:
                dif = max(y_true) - min(y_true)
                return round(rmse/dif, decimal)
            elif model == 1:
                mean_pred = mean(y_pred)
                return round(rmse/mean_pred, decimal)
            elif model == 2:
                std_pred = y_pred.std()
                return round(rmse/std_pred, decimal)
            else:
                value = sqrt(sum(log((y_pred+1) / (y_true+1))**2)/len(y_true))
                return round(value, decimal)
        else:
            if model == 0:
                dif = max(y_true, axis=0) - min(y_true, axis=0)
                value = rmse / dif
            elif model == 1:
                mean_pred = mean(y_pred, axis=0)
                value = rmse / mean_pred
            elif model == 2:
                std_pred = y_pred.std(axis=0)
                value = rmse / std_pred
            else:
                value = sqrt(sum(log((y_pred + 1) / (y_true + 1)) ** 2, axis=0) / len(y_true))
            return self.__multi_output_result__(value, multi_output, decimal)

    def residual_standard_error(self, clean=True, multi_output="raw_values", decimal=3, **kwargs):
        """
        Residual Standard Error
        https://www.statology.org/residual-standard-error-r/
        """
        y_true, y_pred, onedim = self.__get_data__(clean, kwargs)
        if onedim:
            x = y_pred
            y = y_true / y_pred
            up = (sum((x - mean(x)) * (y - mean(y)))) ** 2
            down = sum((x - mean(x)) ** 2) * sum((y - mean(y)) ** 2)
            return round(up/down, decimal)
        else:
            x = y_pred
            y = y_true / y_pred
            up = (sum((x - mean(x, axis=0)) * (y - mean(y, axis=0)), axis=0)) ** 2
            down = sum((x - mean(x, axis=0)) ** 2, axis=0) * sum((y - mean(y, axis=0)) ** 2, axis=0)
            return round(up / down, decimal)

    def get_metric_by_name(self, func_name:str, paras=None) -> dict:
        """
        Parameters
        ----------
        func_name : str. For example: "RMSE"
        paras : dict. For example:
            Default: Don't specify it. leave it there
            Else: It has to be a dictionary such as {"decimal": 3, "multi_output": "raw_values", }

        Returns
        -------
        dict: { "RMSE": 0.2 }
        """
        temp = {}
        obj = getattr(self, func_name)
        if paras is None:
            temp[func_name] = obj()
        else:
            temp[func_name] = obj(**paras)
        return temp

    def get_metrics_by_list_names(self, list_func_names:list, list_paras=None) -> dict:
        """
        Parameters
        ----------
        func_names : list. For example: ["RMSE", "MAE", "MAPE"]
        paras : list. For example: [ {"decimal": 5, }, None, { "decimal": 4, "multi_output": "raw_values" } }       # List of dict

        Returns
        -------
        dict. For example: { "RMSE": 0.25, "MAE": 0.7, "MAPE": 0.15 }
        """
        temp = {}
        for idx, func_name in enumerate(list_func_names):
            obj = getattr(self, func_name)
            if list_paras is None:
                temp[func_name] = obj()
            else:
                if len(list_func_names) != len(list_paras):
                    print("Failed! Different length between list of functions and list of parameters")
                    exit(0)
                if list_paras[idx] is None:
                    temp[func_name] = obj()
                else:
                    temp[func_name] = obj(**list_paras[idx])
        return temp

    def get_metrics_by_dict(self, metrics_dict:dict) -> dict:
        """
        Parameters
        ----------
        metrics_dict : dict inside dict, for examples:
            {
                "RMSE": { "multi_output": multi_output, "decimal": 4 }
                "MAE": { "clean": True, "multi_output": multi_output, "decimal": 6 }
            }
        Returns
        -------
            A dict: For example
                { "RMSE": 0.3524, "MAE": 0.445263 }
        """
        temp = {}
        for func_name, paras_dict in metrics_dict.items():
            obj = getattr(self, func_name)
            if paras_dict is None:
                temp[func_name] = obj()
            else:
                temp[func_name] = obj(**paras_dict)     # Unpacking a dictionary and passing it to function
        return temp

    EVS = explained_variance_score
    ME = max_error
    MAE = mean_absolute_error
    MSE = mean_squared_error
    RMSE = root_mean_squared_error
    MSLE = mean_squared_log_error
    MedAE = median_absolute_error
    MRE = mean_relative_error
    MAPE = mean_absolute_percentage_error
    SMAPE = symmetric_mean_absolute_percentage_error
    MAAPE = mean_arctangent_absolute_percentage_error
    MASE = mean_absolute_scaled_error
    NSE = nash_sutcliffe_efficiency_coefficient
    WI = willmott_index
    R = pearson_correlation_index
    R2s = pearson_correlation_index_square
    CI = confidence_index
    R2 = coefficient_of_determination
    DRV = deviation_of_runoff_volume
    KGE = kling_gupta_efficiency
    GINI = gini_coefficient
    GINI_WIKI = gini_coefficient_wiki
    PCD = prediction_of_change_in_direction
    E = entropy
    CE = cross_entropy
    KLD = kullback_leibler_divergence
    JSD = jensen_shannon_divergence
    VAF = variance_accounted_for
    RAE = relative_absolute_error
    A10 = a10_index
    A20 = a20_index
    NRMSE = normalized_root_mean_square_error
    RSE = residual_standard_error