import setuptools
import codecs
import os
import re


with open(os.path.join(os.path.dirname(__file__), 'ffn', '__init__.py'), 'r') as fp:
    version = re.search(
        "^__version__ = \((\d+), (\d+), (\d+)\)$",
        fp.read(),
        re.MULTILINE
    ).groups()


with open(os.path.join(os.path.dirname(__file__), 'README.rst'), 'r') as fp:
    description = fp.read().replace("\r\n", "\n")

setuptools.setup(
    name="ffn",
    version='.'.join(version),
    author='Philippe Morissette',
    author_email='morissette.philippe@gmail.com',
    description='Financial functions for Python',
    keywords='python finance quant functions',
    url='https://github.com/pmorissette/ffn',
    license='MIT',
    install_requires=[
        'decorator>=4',
        'numpy>=1.5',
        'pandas>=0.19',
        'pandas-datareader>=0.2',
        'tabulate>=0.7.5',
        'matplotlib>=1',
        'scikit-learn>=0.15',
        'scipy>=0.15',
        'future>=0.15'
    ],
    extras_require={
        'dev': [
            'codecov',
            'coverage',
            'future',
            'mock',
            'nose',
        ],
    },
    packages=['ffn'],
    long_description=description,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python'
    ]
)
