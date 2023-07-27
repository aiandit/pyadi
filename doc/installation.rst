Installation
************

Make sure that Python is installed. A virtual environment using
:py:mod:`virtualenv` should by used, for example, on a Debian or
Ubuntu system use this command to install the basic python3 backages
and virtualenv::

  sudo apt install python3-virtualenv

Create a virtualenv and activate it::

  mkdir ~/.venvs/
  virtualenv ~/.venvs/testenv
  . ~/.venvs/testenv/bin/activate

This changes the shell environment so that an isolated python
environment is available by just calling ``python`` (not ``python3``)
and ``pip`` to install further packages into that virtualenv.

Installation from GitHub
========================

Install directly from the git repository, dependencies (:py:mod:`numpy`) first::

    pip install -r https://raw.githubusercontent.com/aiandit/pyadi/master/requirements.txt
    pip install https://github.com/aiandit/pyadi/archive/refs/heads/master.zip


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
