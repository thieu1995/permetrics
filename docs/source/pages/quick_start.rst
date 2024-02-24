
============
Installation
============

* Install the `current PyPI release <https://pypi.python.org/pypi/permetrics />`_::

   $ pip install permetrics==2.0.0


* Install directly from source code::

   $ git clone https://github.com/thieu1995/permetrics.git
   $ cd permetrics
   $ python setup.py install

* In case, you want to install the development version from Github::

   $ pip install git+https://github.com/thieu1995/permetrics


After installation, you can import Permetrics as any other Python module::

   $ python
   >>> import permetrics
   >>> permetrics.__version__


Let's go through some examples.


========
Examples
========

There are several ways you can use a performance metrics in this library. However, the most used are these two ways: functional-based and object-oriented
based programming. We will go through detail of how to use 3 main type of metrics (regression, classification, and clustering) with these two methods.


.. include:: examples/regression.rst
.. include:: examples/classification.rst
.. include:: examples/clustering.rst


.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3

.. toctree::
   :maxdepth: 3
