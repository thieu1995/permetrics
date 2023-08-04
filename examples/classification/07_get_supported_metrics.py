#!/usr/bin/env python
# Created by "Thieu" at 09:34, 04/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from permetrics import ClassificationMetric

ascore = ClassificationMetric.get_support(name="AS", verbose=True)
print(ascore)

all_metrics = ClassificationMetric.get_support(name="all", verbose=True)
print(all_metrics)
