from setuptools import setup

setup(
    name='esce',
    version='0.1.1',    
    py_modules=['esce'],
    install_requires=['scikit-learn',
                      'pandas',
                      'scipy',
                      'imbalanced-learn',
                      'plotly',
                      'kaleido'
                      ],
)
