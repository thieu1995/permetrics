#!/usr/bin/env python
# Created by "Thieu" at 09:40, 04/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from permetrics import RegressionMetric

rmse = RegressionMetric.get_support(name="RMSE", verbose=True)
print(rmse)

all_metrics = RegressionMetric.get_support(name="all", verbose=True)
print(all_metrics)
