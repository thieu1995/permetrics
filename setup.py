#!/usr/bin/env python
# Created by "Thieu" at 13:24, 27/02/2022 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import setuptools
import os
import re


def get_version():
    init_path = os.path.join(os.path.dirname(__file__), 'permetrics', '__init__.py')
    with open(init_path, 'r', encoding='utf-8') as f:
        init_content = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]+)['\"]", init_content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def readme():
    with open('README.md', encoding='utf-8') as f:
        res = f.read()
    return res


setuptools.setup(
    name="permetrics",
    version=get_version(),
    author="Thieu",
    author_email="nguyenthieu2102@gmail.com",
    description="PerMetrics: A Framework of Performance Metrics for Machine Learning Models",
    long_description=readme(),
    long_description_content_type="text/markdown",
    keywords=["regression", "classification", "clustering", "metrics", "performance metrics",
              "rmse", "mae", "mape", "nse", "nash-sutcliffe-efficiency", "willmott-index",
              "precision", "accuracy", "recall", "f1 score", "pearson correlation coefficient",
              "r2", "Kling-Gupta Efficiency", "Gini Coefficient", "Matthews Correlation Coefficient",
              "Cohen's Kappa score", "Jaccard score", "ROC-AUC", "mutual information", "rand score",
              "Davies Bouldin score", "completeness score", "Silhouette Coefficient score",
              "V-measure score", "Folkes Mallows score", "Czekanowski-Dice score", "Huber Gamma score",
              "Kulczynski score", "McNemar score", "Phi score", "Rogers-Tanimoto score", "Russel-Rao score",
              "Sokal-Sneath score", "confusion matrix", "pearson correlation coefficient (PCC)",
              "spearman correlation coefficient (SCC)", "Performance analysis"],
    url="https://github.com/thieu1995/permetrics",
    project_urls={
        'Documentation': 'https://mafese.readthedocs.io/',
        'Source Code': 'https://github.com/thieu1995/permetrics',
        'Bug Tracker': 'https://github.com/thieu1995/permetrics/issues',
        'Change Log': 'https://github.com/thieu1995/permetrics/blob/master/ChangeLog.md',
        'Forum': 'https://t.me/+fRVCJGuGJg1mNDg1',
    },
    packages=setuptools.find_packages(exclude=['tests*', 'examples*']),
    include_package_data=True,
    license="GPLv3",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: System :: Benchmark",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    install_requires=["numpy>=2.2.2", "scipy>=1.13.0"],
    extras_require={
        "dev": ["pytest>=9.0.3", "pytest-cov>=4.1.0", "flake8>=6.0.0", "scikit-learn>=1.5.0"],
    },
    python_requires='>=3.11',
)
