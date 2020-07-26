#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:07, 18/07/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import max, round, sqrt, abs, mean, dot, divide, arctan, sum, any, median, log, var, std
from numpy import ndarray, array, isfinite, isnan


class Metrics:
    """
        https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
    """

    def __init__(self, y_true, y_pred):
        """
        :param y_true:
        :param y_pred:
        """
        self.onedim = False
        if type(y_true) is ndarray and type(y_pred) is ndarray:
            y_true = y_true.astype('float64')
            y_pred = y_pred.astype('float64')  # x = x[~np.isnan(x)] can't remove if array is dtype object, only work with dtype float
            if y_true.ndim == 1 and y_pred.ndim == 1:
                self.onedim = True
                ## Remove all Nan in y_pred
                y_true = y_true[~isnan(y_pred)]
                y_pred = y_pred[~isnan(y_pred)]
                ## Remove all Inf in y_pred
                self.y_true = y_true[isfinite(y_pred)]
                self.y_pred = y_pred[isfinite(y_pred)]

                self.y_true_clean = self.y_true[self.y_pred != 0]
                self.y_pred_clean = self.y_pred[self.y_pred != 0]
            else:
                if y_true.ndim == y_pred.ndim:
                    ## Remove all row with Nan in y_pred
                    y_true = y_true[~isnan(y_pred).any(axis=1)]
                    y_pred = y_pred[~isnan(y_pred).any(axis=1)]
                    ## Remove all row with Inf in y_pred
                    self.y_true = y_true[isfinite(y_pred).any(axis=1)]
                    self.y_pred = y_pred[isfinite(y_pred).any(axis=1)]

                    self.y_true_clean = self.y_true[~any(self.y_pred == 0, axis=1)]
                    self.y_pred_clean = self.y_pred[~any(self.y_pred == 0, axis=1)]
                else:
                    print("=====Failed! y_true and y_pred need to have same number of dimensions=======")
                    exit(0)
        else:
            print("=====Failed! y_true and y_pred need to be a ndarray=======")
            exit(0)


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

    def evs_func(self, clean=False, multi_output="raw_values", decimal=3):
        """
            Explained Variance Score. Best possible score is 1.0, lower values are worse.
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        if self.onedim:
            return round(1 - var(y_true - y_pred) / var(y_true), decimal)
        else:
            temp = 1 - var(y_true - y_pred, axis=0) / var(y_true, axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def me_func(self, clean=False, multi_output="raw_values", decimal=3):
        """
            Max Error: Smaller is better
        :param clean: Pre-processing y_true and y_pred, no 0 values
        :param multi_output: string in [‘raw_values’, ‘uniform_average’, ‘variance_weighted’] or array-like of shape (n_outputs)
        :param decimal: the number after ","
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        if self.onedim:
            return round(max(abs(y_true - y_pred)), decimal)
        else:
            temp = max(abs(y_true - y_pred), axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def mae_func(self, clean=False, multi_output="raw_values", decimal=3):
        """
            Mean Absolute Error: Smaller is better
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        if self.onedim:
            return round(sum(abs(y_pred - y_true)) / len(y_true), decimal)
        else:
            temp = sum(abs(y_pred - y_true), axis=0) / len(y_true)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def mse_func(self, clean=False, multi_output="raw_values", decimal=3):
        """
            Mean Squared Error: Smaller is better
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        if self.onedim:
            return round(sum((y_pred - y_true)**2) / len(y_true), decimal)
        else:
            temp = sum((y_pred - y_true)**2, axis=0) / len(y_true)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def rmse_func(self, clean=False, multi_output="raw_values", decimal=3):
        """
            Root Mean Squared Error: Smaller is better
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        if self.onedim:
            return round(sqrt(sum((y_pred - y_true) ** 2) / len(y_true)), decimal)
        else:
            temp = sqrt(sum((y_pred - y_true) ** 2, axis=0) / len(y_true))
            return self.__multi_output_result__(temp, multi_output, decimal)


    def msle_func(self, clean=True, multi_output="raw_values", decimal=3):
        """
            Mean Squared Log Error: Smaller is better
        """
        if self.onedim:
            y_true = self.y_true[self.y_pred > 0]
            y_pred = self.y_pred[self.y_pred > 0]
        else:
            y_true = self.y_true[any(self.y_pred > 0, axis=1)]
            y_pred = self.y_pred[any(self.y_pred > 0, axis=1)]
        if clean:
            if self.onedim:
                y_true = self.y_true_clean[self.y_pred_clean > 0]
                y_pred = self.y_pred_clean[self.y_pred_clean > 0]
            else:
                y_true = self.y_true_clean[any(self.y_pred_clean > 0, axis=1)]
                y_pred = self.y_pred_clean[any(self.y_pred_clean > 0, axis=1)]
        if self.onedim:
            return round(sum(log(y_true / y_pred) ** 2) / len(y_true), decimal)
        else:
            temp = sum(log(y_true / y_pred) ** 2, axis=0) / len(y_true)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def medae_func(self, clean=False, multi_output="raw_values", decimal=3):
        """
            Median Absolute Error: Smaller is better
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        if self.onedim:
            return round(median(abs(y_true - y_pred)), decimal)
        else:
            temp = median(abs(y_true - y_pred), axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def mre_func(self, clean=True, multi_output="raw_values", decimal=3):
        """
            Mean Relative Error: Smaller is better
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        if self.onedim:
            return round(mean(divide(abs(y_true - y_pred), y_true)), decimal)
        else:
            temp = mean(divide(abs(y_true - y_pred), y_true), axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def mape_func(self, clean=True, multi_output="raw_values", decimal=3):
        """
            Mean Absolute Percentage Error: Good if mape < 30%. Smaller is better
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        if self.onedim:
            return round(mean(divide(abs(y_true - y_pred), abs(y_true))) * 100, decimal)
        else:
            temp = mean(divide(abs(y_true - y_pred), abs(y_true)), axis=0) * 100
            return self.__multi_output_result__(temp, multi_output, decimal)


    def smape_func(self, clean=True, multi_output="raw_values", decimal=3):
        """
            Symmetric Mean Absolute Percentage Error: Good if mape < 20%. Smaller is better
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        if self.onedim:
            return round(mean(2 * abs(y_pred - y_true) / (abs(y_true) + abs(y_pred))) * 100, decimal)
        else:
            temp = mean(2 * abs(y_pred - y_true) / (abs(y_true) + abs(y_pred)), axis=0) * 100
            return self.__multi_output_result__(temp, multi_output, decimal)


    def maape_func(self, clean=False, multi_output="raw_values", decimal=3):
        """
            Mean Arctangent Absolute Percentage Error (output: radian values)
            https://support.numxl.com/hc/en-us/articles/115001223463-MAAPE-Mean-Arctangent-Absolute-Percentage-Error
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        if self.onedim:
            return round(mean(arctan(divide(abs(y_true - y_pred), y_true))), decimal)
        else:
            temp = mean(arctan(divide(abs(y_true - y_pred), y_true)), axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def mase_func(self, m=1, clean=True, multi_output="raw_values", decimal=3):
        """
            Mean Absolute Scaled Error (m = 1 for non-seasonal data, m > 1 for seasonal data)
            https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        if self.onedim:
            return round(mean(abs(y_true - y_pred)) / mean(abs(y_true[m:] - y_true[:-m])), decimal)
        else:
            temp = mean(abs(y_true - y_pred), axis=0) / mean(abs(y_true[m:] - y_true[:-m]), axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def nse_func(self, clean=False, multi_output="raw_values", decimal=3):
        """
            Nash-Sutcliffe Efficiency Coefficient (-unlimited < NSE < 1.   Larger is better)
            https://agrimetsoft.com/calculators/Nash%20Sutcliffe%20model%20Efficiency%20coefficient
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        if self.onedim:
            return round(1 - sum((y_true - y_pred) ** 2) / sum((y_true - mean(y_true)) ** 2), decimal)
        else:
            temp = 1 - sum((y_true - y_pred) ** 2, axis=0) / sum((y_true - mean(y_true, axis=0)) ** 2, axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)

    def wi_func(self, clean=False, multi_output="raw_values", decimal=3):
        """
            Willmott Index (Willmott, 1984, 0 < WI < 1. Larger is better)
            Reference evapotranspiration for Londrina, Paraná, Brazil: performance of different estimation methods
        https://www.researchgate.net/publication/319699360_Reference_evapotranspiration_for_Londrina_Parana_Brazil_performance_of_different_estimation_methods
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        if self.onedim:
            m1 = mean(self.y_true)
            return round(1 - sum((y_pred - y_true) ** 2) / sum((abs(y_pred - m1) + abs(y_true - m1)) ** 2), decimal)
        else:
            m1 = mean(self.y_true, axis=0)
            temp = 1 - sum((y_pred - y_true) ** 2, axis=0) / sum((abs(y_pred - m1) + abs(y_true - m1)) ** 2, axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def r_func(self, clean=False, multi_output="raw_values", decimal=3):
        """
            Pearson’s Correlation Index (Willmott, 1984): -1 < R < 1. Larger is better
            Reference evapotranspiration for Londrina, Paraná, Brazil: performance of different estimation methods
            https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        if self.onedim:
            m1, m2 = mean(y_true), mean(y_pred)
            temp = sum((abs(y_true - m1) * abs(y_pred - m2))) / (sqrt(sum((y_true - m1) ** 2)) * sqrt(sum((y_pred - m2) ** 2)))
            return round(temp, decimal)
        else:
            m1, m2 = mean(y_true, axis=0), mean(y_pred, axis=0)
            t1 = sqrt(sum((y_true - m1) ** 2, axis=0))
            t2 = sqrt(sum((y_pred - m2) ** 2, axis=0))
            t3 = sum((abs(y_true - m1) * abs(y_pred - m2)), axis=0)
            temp = t3 / (t1 * t2)
            return self.__multi_output_result__(temp, multi_output, decimal)


    def ci_func(self, clean=False, multi_output="raw_values", decimal=3):
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
        r = self.r_func(clean=clean)
        d = self.wi_func(clean=clean)
        return self.__multi_output_result__(r * d, multi_output, decimal)


    def r2_func(self, clean=False, multi_output="raw_values", decimal=3):
        """
            Coefficient of Determination - regression score function. Best possible score is 1.0 and it can be negative
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        if self.onedim:
            t1 = sum((y_true - y_pred)**2)
            t2 = sum((y_true - mean(y_true))**2)
            return round(1 - t1/t2, decimal)
        else:
            t1 = sum((y_true - y_pred) ** 2, axis=0)
            t2 = sum((y_true - mean(y_true, axis=0)) ** 2, axis=0)
            temp = 1 - t1 / t2
            return self.__multi_output_result__(temp, multi_output, decimal)

    def r2s_func(self, clean=False, multi_output="raw_values", decimal=3):
        """
            (Pearson’s Correlation Index)^2 = R2
        """
        temp = self.r2_func(clean, multi_output, decimal)
        return round(temp ** 2, decimal)

    def drv_func(self, clean=False, multi_output="raw_values", decimal=3):
        """
            Deviation of Runoff Volume (DRV)
            https://rstudio-pubs-static.s3.amazonaws.com/433152_56d00c1e29724829bad5fc4fd8c8ebff.html
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        if self.onedim:
            return round(sum(y_pred)/sum(y_true), decimal)
        else:
            temp = sum(y_pred, axis=0) / sum(y_true, axis=0)
            return self.__multi_output_result__(temp, multi_output, decimal)

    def kge_func(self, clean=False, multi_output="raw_values", decimal=3):
        """
            Kling-Gupta Efficiency (KGE)
            https://rstudio-pubs-static.s3.amazonaws.com/433152_56d00c1e29724829bad5fc4fd8c8ebff.html
        """
        y_true, y_pred = self.y_true, self.y_pred
        if clean:
            y_true, y_pred = self.y_true_clean, self.y_pred_clean
        r = self.r_func(clean, multi_output, decimal)
        if self.onedim:
            beta = mean(y_pred)/mean(y_true)
            gamma = (std(y_pred)/mean(y_pred))/(std(y_true)/mean(y_true))
            out = 1 - sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)
            return round(out, decimal)
        else:
            beta = mean(y_pred, axis=0) / mean(y_true, axis=0)
            gamma = (std(y_pred, axis=0) / mean(y_pred, axis=0)) / (std(y_true, axis=0) / mean(y_true, axis=0))
            out = 1 - sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
            return self.__multi_output_result__(out, multi_output, decimal)


    def get_metrics_by_name(self, *func_names):
        temp = []
        for idx, func_name in enumerate(func_names):
            obj = getattr(self, func_name)
            temp.append(obj())
        return temp

    def get_metrics_by_list(self, func_name_list=None, func_para_list=None):
        temp = []
        for idx, func_name in enumerate(func_name_list):
            obj = getattr(self, func_name)
            if func_para_list is None:
                temp.append(obj())
            else:
                if len(func_name_list) != len(func_para_list):
                    print("Failed! Different length between functions and parameters")
                    exit(0)
                temp.append(obj(**func_para_list[idx]))
        return temp

    EVS = evs_func
    ME = me_func
    MAE = mae_func
    MSE = mse_func
    RMSE = rmse_func
    MSLE = msle_func
    MedAE = medae_func
    MRE = mre_func
    MAPE = mape_func
    SMAPE = smape_func
    MAAPE = maape_func
    MASE = mase_func
    NSE = nse_func
    WI = wi_func
    R = r_func
    CI = ci_func
    R2 = r2_func
    R2s = r2s_func
    DRV = drv_func
    KGE = kge_func

