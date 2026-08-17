from setuptools import setup, find_packages

setup(
    name="vsf",
    version="1.0.0",
    description="Visual Sufficiency Framework (VSF): Information-Theoretic Adaptive Dimensionality Selection",
    author="VSF Core Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
