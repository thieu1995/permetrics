Clustering Metrics
==================

+-----+--------+-------------------------------------------+--------------------------------------------------------+
| STT | Metric | Metric Fullname                           | Characteristics                                        |
+=====+========+===========================================+========================================================+
| 1   | BHI    | Ball Hall Index                           | Smaller is better (Best = 0), Range=[0, +inf)          |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 2   | XBI    | Xie Beni Index                            | Smaller is better (Best = 0), Range=[0, +inf)          |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 3   | DBI    | Davies Bouldin Index                      | Smaller is better (Best = 0), Range=[0, +inf)          |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 4   | BRI    | Banfeld Raftery Index                     | Smaller is better (No best value), Range=(-inf, inf)   |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 5   | KDI    | Ksq Detw Index                            | Smaller is better (No best value), Range=(-inf, +inf)  |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 6   | DRI    | Det Ratio Index                           | Bigger is better (No best value), Range=[0, +inf)      |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 7   | DI     | Dunn Index                                | Bigger is better (No best value), Range=[0, +inf)      |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 8   | CHI    | Calinski Harabasz Index                   | Bigger is better (No best value), Range=[0, inf)       |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 9   | LDRI   | Log Det Ratio Index                       | Bigger is better (No best value), Range=(-inf, +inf)   |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 10  | LSRI   | Log SS Ratio Index                        | Bigger is better (No best value), Range=(-inf, +inf)   |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 11  | SI     | Silhouette Index                          | Bigger is better (Best = 1), Range = [-1, +1]          |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 12  | SSEI   | Sum of Squared Error Index                | Smaller is better (Best = 0), Range = [0, +inf)        |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 13  | MSEI   | Mean Squared Error Index                  | Smaller is better (Best = 0), Range = [0, +inf)        |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 14  | DHI    | Duda-Hart Index                           | Smaller is better (Best = 0), Range = [0, +inf)        |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 15  | BI     | Beale Index                               | Smaller is better (Best = 0), Range = [0, +inf)        |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 16  | RSI    | R-squared Index                           | Bigger is better (Best=1), Range = (-inf, 1]           |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 17  | DBCVI  | Density-based Clustering Validation Index | Bigger is better (Best=0), Range = [0, 1]              |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 18  | HI     | Hartigan Index                            | Bigger is better (best=0), Range = [0, +inf)           |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 19  | MIS    | Mutual Info Score                         | Bigger is better (No best value), Range = [0, +inf)    |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 20  | NMIS   | Normalized Mutual Info Score              | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 21  | RaS    | Rand Score                                | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 22  | ARS    | Adjusted Rand Score                       | Bigger is better (Best = 1), Range = [-1, 1]           |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 23  | FMS    | Fowlkes Mallows Score                     | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 24  | HS     | Homogeneity Score                         | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 25  | CS     | Completeness Score                        | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 26  | VMS    | V-Measure Score                           | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 27  | PrS    | Precision Score                           | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 28  | ReS    | Recall Score                              | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 29  | FmS    | F-Measure Score                           | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 30  | CDS    | Czekanowski Dice Score                    | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 31  | HGS    | Hubert Gamma Score                        | Bigger is better (Best = 1), Range=[-1, +1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 32  | JS     | Jaccard Score                             | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 33  | KS     | Kulczynski Score                          | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 34  | MNS    | Mc Nemar Score                            | Bigger is better (No best value), Range=(-inf, +inf)   |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 35  | PhS    | Phi Score                                 | Bigger is better (No best value), Range = (-inf, +inf) |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 36  | RTS    | Rogers Tanimoto Score                     | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 37  | RRS    | Russel Rao Score                          | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 38  | SS1S   | Sokal Sneath1 Score                       | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 39  | SS2S   | Sokal Sneath2 Score                       | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 40  | PuS    | Purity Score                              | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 41  | ES     | Entropy Score                             | Smaller is better (Best = 0), Range = [0, +inf)        |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 42  | TS     | Tau Score                                 | Bigger is better (No best value), Range = (-inf, +inf) |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 43  | GAS    | Gamma Score                               | Bigger is better (Best = 1), Range = [-1, 1]           |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 44  | GPS    | Gplus Score                               | Smaller is better (Best = 0), Range = [0, 1]           |
+-----+--------+-------------------------------------------+--------------------------------------------------------+



Most of the clustering metrics is implemented based on the paper :cite:`desgraupes2013clustering`

There are several types of clustering metrics that are commonly used to evaluate the quality of clustering results.

+ Internal evaluation metrics: These are metrics that evaluate the clustering results based solely on the data and the clustering algorithm used, without any external information. Examples of internal evaluation metrics include Silhouette Coefficient, Calinski-Harabasz Index, and Davies-Bouldin Index.

+ External evaluation metrics: These are metrics that evaluate the clustering results by comparing them to some external reference, such as expert labels or a gold standard. Examples of external evaluation metrics include Adjusted Rand score, Normalized Mutual Information score, and Fowlkes-Mallows score.


It's important to choose the appropriate clustering metrics based on the specific problem and data at hand.

**In this library, metrics that belong to the internal evaluation category will have a metric name suffix of "index" On the other hand, metrics that belong to the external evaluation category will have a metric name suffix of "score"**


.. toctree::
   :maxdepth: 3

   clustering/DHI.rst
   clustering/SSEI.rst
   clustering/BI.rst
   clustering/RSI.rst
   clustering/DBCVI.rst
   clustering/CHI.rst
   clustering/BHI.rst
   clustering/DI.rst
   clustering/HI.rst
   clustering/ES.rst
   clustering/PuS.rst
   clustering/TS.rst

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3
