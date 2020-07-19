from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        README = f.read()
    return README

setup(
    name="permetrics",
    version="1.0.0",
    author="Thieu Nguyen",
    author_email="nguyenthieu2102@gmail.com",
    description="A framework of PERformance METRICS (PerMetrics) for artificial intelligence models",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/thieunguyen5991/permetrics",
    download_url="https://github.com/thieunguyen5991/permetrics/archive/v1.0.0.zip",
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: System :: Benchmark",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Science/Research"
    ],
    install_requires=["numpy"],
    python_requires='>=3.7',
)