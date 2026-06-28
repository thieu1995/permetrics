Clustering Metrics
==================

.. toctree::
   :maxdepth: 1

   clustering/BHI.rst
   clustering/CHI.rst
   clustering/XBI.rst
   clustering/DBI.rst
   clustering/BRI.rst
   clustering/DRI.rst
   clustering/KDI.rst
   clustering/LDRI.rst
   clustering/DI.rst
   clustering/LSRI.rst
   clustering/SI.rst
   clustering/SSEI.rst
   clustering/MSEI.rst
   clustering/DHI.rst
   clustering/BI.rst
   clustering/RSI.rst
   clustering/DBCVI.rst
   clustering/HI.rst

   clustering/MIS.rst
   clustering/NMIS.rst
   clustering/RaS.rst
   clustering/ARS.rst
   clustering/FMS.rst
   clustering/HS.rst
   clustering/CS.rst
   clustering/VMS.rst
   clustering/PrS.rst
   clustering/ReS.rst
   clustering/FS.rst
   clustering/CDS.rst
   clustering/HGS.rst
   clustering/JS.rst
   clustering/KS.rst
   clustering/MNS.rst
   clustering/PhS.rst
   clustering/RTS.rst
   clustering/RRS.rst
   clustering/SS12S.rst
   clustering/PuS.rst
   clustering/EnS.rst
   clustering/TauS.rst
   clustering/GAS.rst
   clustering/GPS.rst


======================
All Clustering Metrics
======================

The majority of the clustering metrics in this library are implemented based on the comprehensive study by :cite:`desgraupes2013clustering`.

There are two primary categories of metrics commonly used to evaluate the quality of clustering results:

* **Internal Evaluation Metrics**: These metrics evaluate the clustering structure based solely on the intrinsic properties of the data itself, without relying on any external information or ground truth labels. Examples include the Silhouette Coefficient, Calinski-Harabasz Index, and Davies-Bouldin Index.

* **External Evaluation Metrics**: These metrics assess the clustering results by comparing them to an external reference, such as expert-annotated labels or a gold standard. Examples include the Adjusted Rand Score, Normalized Mutual Information Score, and Fowlkes-Mallows Score.

It is highly recommended to evaluate your models using appropriate metrics tailored to your specific problem domain and data distribution.

.. note::
   **Library Naming Convention**

   To help users easily distinguish between the two evaluation categories, this library strictly enforces the following naming rule:

   * Metrics belonging to the **internal evaluation** category will have the suffix **"Index"** (e.g., Dunn Index).
   * Metrics belonging to the **external evaluation** category will have the suffix **"Score"** (e.g., Adjusted Rand Score).


+-----+--------+-------------------------------------------+--------------------------------------------------------+
| STT | Metric | Metric Fullname                           | Characteristics                                        |
+=====+========+===========================================+========================================================+
| 1   | BHI    | Ball Hall Index                           | Smaller is better (Best = 0), Range=[0, +inf)          |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 2   | XBI    | Xie Beni Index                            | Smaller is better (Best = 0), Range=[0, +inf)          |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 3   | DBI    | Davies Bouldin Index                      | Smaller is better (Best = 0), Range=[0, +inf)          |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 4   | BRI    | Banfeld Raftery Index                     | Smaller is better (No best value), Range=(-inf, +inf)  |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 5   | KDI    | Ksq Detw Index                            | Smaller is better (No best value), Range=(-inf, +inf)  |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 6   | DRI    | Det Ratio Index                           | Bigger is better (No best value), Range=[1, +inf)      |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 7   | DI     | Dunn Index                                | Bigger is better (No best value), Range=[0, +inf)      |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 8   | CHI    | Calinski Harabasz Index                   | Bigger is better (No best value), Range=[0, +inf)      |
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
| 16  | RSI    | R-squared Index                           | Bigger is better (Best=1), Range = [0, 1]              |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 17  | DBCVI  | Density-based Clustering Validation Index | Bigger is better (Best=1), Range = [-1, 1]             |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 18  | HI     | Hartigan Index                            | Smaller is better (best=0), Range = [0, +inf)          |
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
| 29  | FS     | F-Measure Score                           | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 30  | CDS    | Czekanowski Dice Score                    | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 31  | HGS    | Hubert Gamma Score                        | Bigger is better (Best = 1), Range=[-1, 1]             |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 32  | JS     | Jaccard Score                             | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 33  | KS     | Kulczynski Score                          | Bigger is better (Best = 1), Range = [0, 1]            |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 34  | MNS    | Mc Nemar Score                            | Bigger is better (No best value), Range=(-inf, +inf)   |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 35  | PhS    | Phi Score                                 | Bigger is better (Best = 1), Range = [-1, 1]           |
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
| 41  | EnS    | Entropy Score                             | Smaller is better (Best = 0), Range = [0, +inf)        |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 42  | TauS   | Tau Score                                 | Bigger is better (Best = 1), Range = [-1, 1]           |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 43  | GAS    | Gamma Score                               | Bigger is better (Best = 1), Range = [-1, 1]           |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
| 44  | GPS    | Gplus Score                               | Smaller is better (Best = 0), Range = [0, 1]           |
+-----+--------+-------------------------------------------+--------------------------------------------------------+
