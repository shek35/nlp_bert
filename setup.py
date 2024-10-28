# File: setup.py

from setuptools import setup, find_packages

setup(
    name="ds_project",
    version="0.1",
    packages=find_packages(include=['src', 'src.*']),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'nltk',
        'pytest',
        'joblib'
    ]
)