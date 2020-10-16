from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name = "esce",
    version = "0.0.1",
    author = "Alexander Koch",
    author_email = "kochalexander@gmx.net",
    description = "Empirical Sample Complexity Estimator",
    long_description= long_description,
    long_description_content_type= "text/markdown",
    url = "https://github.com/maschulz/esce",
    packages = find_packages(),
    classifiers = [],
    python_requires = '>=3.6',
    entry_points = {
        'console_scripts': ['esce=esce.command_line:main'],
    }
)