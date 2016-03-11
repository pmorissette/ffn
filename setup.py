import setuptools
import codecs
import os
import re


def local_file(filename):
    return codecs.open(
        os.path.join(os.path.dirname(__file__), filename), 'r', 'utf-8'
    )


version = re.search(
    "^__version__ = \((\d+), (\d+), (\d+)\)$",
    local_file('ffn/__init__.py').read(),
    re.MULTILINE
).groups()


setuptools.setup(
    name="ffn",
    version='.'.join(version),
    author='Philippe Morissette',
    author_email='morissette.philippe@gmail.com',
    description='Financial functions for Python',
    keywords='python finance quant functions',
    url='https://github.com/pmorissette/ffn',
    install_requires=[
        'decorator',
        'numpy',
        'pandas',
        'pandas_datareader',
        'tabulate',
        'matplotlib',
        'scikit-learn',
        'scipy',
        'future'
    ],
    packages=['ffn'],
    long_description=local_file('README.rst').read(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python'
    ]
)
