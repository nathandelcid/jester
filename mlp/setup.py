"""
Setup script for MLP from Scratch package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlp-from-scratch",
    version="0.1.0",
    author="Nathan Delcid",
    description="A minimal neural network framework built from scratch using NumPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mlp-from-scratch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "nltk>=3.6.0",
        "pillow>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.9",
            "black>=21.0",
        ],
    },
)
