from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name = "esce",
    version = "0.6.2",
    author = "Alexander Koch",
    author_email = "kochalexander@gmx.net",
    description = "Empirical Sample Complexity Estimator",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/maschulz/esce",
    packages = find_packages(),
    classifiers = [],
    python_requires = '>=3.6',
    entry_points = {
        'console_scripts': ['esce=esce.command_line:main'],
    },
    install_requires = [
        "numpy>=1.19",
        "pandas>=0.23",
        "scikit-learn>=0.23",
        "torchvision>=0.4",
        "h5py>=2.8",
        "joblib>=0.12",
        "matplotlib>=3.0",
        "seaborn>=0.9",
        "tqdm>=4.26",
        "requests>=2.22",
        "PyYAML>=3.13"
    ]
)