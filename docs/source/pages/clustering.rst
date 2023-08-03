Clustering Metrics
==================


+-----+--------+------------------------------+--------------------------------------------------------+
| STT | Metric | Metric Fullname              | Characteristics                                        |
+=====+========+==============================+========================================================+
| 1   | BHI    | Ball Hall Index              | Smaller is better (Best = 0), Range=[0, +inf)          |
+-----+--------+------------------------------+--------------------------------------------------------+
| 2   | XBI    | Xie Beni Index               | Smaller is better (Best = 0), Range=[0, +inf)          |
+-----+--------+------------------------------+--------------------------------------------------------+
| 3   | DBI    | Davies Bouldin Index         | Smaller is better (Best = 0), Range=[0, +inf)          |
+-----+--------+------------------------------+--------------------------------------------------------+
| 4   | BRI    | Banfeld Raftery Index        | Smaller is better (No best value), Range=(-inf, inf)   |
+-----+--------+------------------------------+--------------------------------------------------------+
| 5   | KDI    | Ksq Detw Index               | Smaller is better (No best value), Range=(-inf, +inf)  |
+-----+--------+------------------------------+--------------------------------------------------------+
| 6   | DRI    | Det Ratio Index              | Bigger is better (No best value), Range=[0, +inf)      |
+-----+--------+------------------------------+--------------------------------------------------------+
| 7   | DI     | Dunn Index                   | Bigger is better (No best value), Range=[0, +inf)      |
+-----+--------+------------------------------+--------------------------------------------------------+
| 8   | CHI    | Calinski Harabasz Index      | Bigger is better (No best value), Range=[0, inf)       |
+-----+--------+------------------------------+--------------------------------------------------------+
| 9   | LDRI   | Log Det Ratio Index          | Bigger is better (No best value), Range=(-inf, +inf)   |
+-----+--------+------------------------------+--------------------------------------------------------+
| 10  | LSRI   | Log SS Ratio Index           | Bigger is better (No best value), Range=(-inf, +inf)   |
+-----+--------+------------------------------+--------------------------------------------------------+
| 11  | SI     | Silhouette Index             | Bigger is better (Best = 1), Range = [-1, +1]          |
+-----+--------+------------------------------+--------------------------------------------------------+
| 12  | MIS    | Mutual Info Score            | Higher is better (No best value), Range = [0, +inf)    |
+-----+--------+------------------------------+--------------------------------------------------------+
| 13  | NMIS   | Normalized Mutual Info Score | Higher is better (Best = 1), Range = [0, 1]            |
+-----+--------+------------------------------+--------------------------------------------------------+
| 14  | RaS    | Rand Score                   | Higher is better (Best = 1), Range = [0, 1]            |
+-----+--------+------------------------------+--------------------------------------------------------+
| 15  | FMS    | Fowlkes Mallows Score        | Higher is better (Best = 1), Range = [0, 1]            |
+-----+--------+------------------------------+--------------------------------------------------------+
| 16  | HS     | Homogeneity Score            | Higher is better (Best = 1), Range = [0, 1]            |
+-----+--------+------------------------------+--------------------------------------------------------+
| 17  | CS     | Completeness Score           | Higher is better (Best = 1), Range = [0, 1]            |
+-----+--------+------------------------------+--------------------------------------------------------+
| 18  | VMS    | V-Measure Score              | Higher is better (Best = 1), Range = [0, 1]            |
+-----+--------+------------------------------+--------------------------------------------------------+
| 19  | PrS    | Precision Score              | Higher is better (Best = 1), Range = [0, 1]            |
+-----+--------+------------------------------+--------------------------------------------------------+
| 20  | ReS    | Recall Score                 | Higher is better (Best = 1), Range = [0, 1]            |
+-----+--------+------------------------------+--------------------------------------------------------+
| 21  | FmS    | F-Measure Score              | Higher is better (Best = 1), Range = [0, 1]            |
+-----+--------+------------------------------+--------------------------------------------------------+
| 22  | CDS    | Czekanowski Dice Score       | Higher is better (Best = 1), Range = [0, 1]            |
+-----+--------+------------------------------+--------------------------------------------------------+
| 23  | HGS    | Hubert Gamma Score           | Higher is better (Best = 1), Range=[-1, +1]            |
+-----+--------+------------------------------+--------------------------------------------------------+
| 24  | JS     | Jaccard Score                | Higher is better (Best = 1), Range = [0, 1]            |
+-----+--------+------------------------------+--------------------------------------------------------+
| 25  | KS     | Kulczynski Score             | Higher is better (Best = 1), Range = [0, 1]            |
+-----+--------+------------------------------+--------------------------------------------------------+
| 26  | MNS    | Mc Nemar Score               | Higher is better (No best value), Range=(-inf, +inf)   |
+-----+--------+------------------------------+--------------------------------------------------------+
| 27  | PhS    | Phi Score                    | Higher is better (No best value), Range = (-inf, +inf) |
+-----+--------+------------------------------+--------------------------------------------------------+
| 28  | RTS    | Rogers Tanimoto Score        | Higher is better (Best = 1), Range = [0, 1]            |
+-----+--------+------------------------------+--------------------------------------------------------+
| 29  | RRS    | Russel Rao Score             | Higher is better (Best = 1), Range = [0, 1]            |
+-----+--------+------------------------------+--------------------------------------------------------+
| 30  | SS1S   | Sokal Sneath1 Score          | Higher is better (Best = 1), Range = [0, 1]            |
+-----+--------+------------------------------+--------------------------------------------------------+
| 31  | SS2S   | Sokal Sneath2 Score          | Higher is better (Best = 1), Range = [0, 1]            |
+-----+--------+------------------------------+--------------------------------------------------------+


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

   clustering/ES.rst
   clustering/PuS.rst

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3
