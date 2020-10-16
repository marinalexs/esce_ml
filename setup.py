from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name = "crit4nonlin",
    version = "0.0.1",
    author = "Alexander Koch",
    author_email = "kochalexander@gmx.net",
    description = "TODO",
    long_description= long_description,
    long_description_content_type= "text/markdown",
    url = "https://github.com/maschulz/crit4nonlin",
    packages = find_packages(),
    classifiers = [],
    python_requires = '>=3.6',
    entry_points = {
        'console_scripts': ['crit4nonlin=crit4nonlin.command_line:main'],
    }
)