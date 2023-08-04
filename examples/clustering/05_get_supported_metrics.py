#!/usr/bin/env python
# Created by "Thieu" at 09:23, 04/08/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from permetrics import ClusteringMetric

bhi = ClusteringMetric.get_support(name="BHI", verbose=True)
print(bhi)

all_metrics = ClusteringMetric.get_support(name="all", verbose=True)
print(all_metrics)
