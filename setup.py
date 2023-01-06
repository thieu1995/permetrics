from setuptools import setup, find_packages


def readme():
    with open('README.md', encoding='utf-8') as f:
        README = f.read()
    return README


setup(
    name="permetrics",
    version="1.3.2",
    author="Nguyen Van Thieu",
    author_email="nguyenthieu2102@gmail.com",
    description="PerMetrics: A framework of PERformance METRICS for machine learning models",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/thieu1995/permetrics",
    download_url="https://github.com/thieu1995/permetrics/archive/v1.3.2.zip",
    packages=find_packages(exclude=['tests*', 'examples*']),
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: System :: Benchmark",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    install_requires=["numpy>=1.15.1"],
    python_requires='>=3.6',
)