#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import re
from setuptools import setup, find_packages

def read_version():
    with open(os.path.join('src', 'pyfad', '__init__.py')) as f:
        m = re.search(r'''version\s*=\s*['"]([^'"]*)['"]''', f.read())
        if m:
            return m.group(1)
        raise ValueError("couldn't find version")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pyfad',
    version=read_version(),
    description='Forward mode AD Python',
    long_description=long_description,
    long_description_content_type="text/markdown",
    maintainer='Johannes Willkomm',
    maintainer_email='jwillkomm@ai-and-it.de',
    url='https://github.com/aiandit/pyfad',
    project_urls={
        "Bug Tracker": "https://github.com/aiandit/pyfad/issues",
    },
    package_dir={'': 'src'},
    packages=find_packages("src"),
    include_package_data=True,
#    package_data={'': ['lib/astunparse/xsl/xml2json.xsl']},
    entry_points={
        'console_scripts': [
            'pydiff.cmdline:pydiff',
        ]
    },
    license="GPLv3",
    keywords='pyfad',
    classifiers=[
        'Development Status :: 4 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GPLv3 License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Code Generators',
    ],
    test_suite='tests',
    tests_require=[],
)
