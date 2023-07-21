Installation
************

A virtual environment using :py:mod:`virtualenv` should by used.

Installation from GitHub
========================

Install directly from the git repository::

    pip install https://github.com/aiandit/pyadi/archive/refs/heads/main.zip


Installation from Source
========================

Clone the PyADi repository::

  git clone https://github.com/aiandit/pyadi.git

And change into the directory::

  cd pyadi

Install the dependencies::

  pip install -r requirements.txt

And install the package::

  pip install .

Or use the Makefile to do both::

  make install
