"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
Modified by Madoshakalaka@Github (dependency links added)
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="MLVT",  # Required
    version="0.1.0" # Required
    description="Machine Learning Visualization Tool",
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",
    author="Dawid Jakóbczak",  # Optional
    classifiers=[  # Optional
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="CNN, classification, Python, Flask, OpenAPI",  # Optional
    packages=find_packages(exclude=["contrib", "docs", "tests"]),  # Required
    python_requires=">=3.6",
    install_requires=[
        "attrs==20.3.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "certifi==2020.12.5",
        "chardet==4.0.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
        "click==7.1.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
        "clickclick==20.10.2",
        "confuse==1.4.0",
        "connexion[swagger-ui]==2.7.0",
        "dataclasses==0.8; python_version < '3.7'",
        "flask==1.1.2",
        "flask-executor==0.9.4",
        "idna==2.10; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "importlib-metadata==3.3.0; python_version < '3.8'",
        "inflection==0.5.1; python_version >= '3.5'",
        "itsdangerous==1.1.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "jinja2==2.11.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
        "jsonschema==3.2.0",
        "markupsafe==1.1.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "numpy==1.19.5",
        "openapi-spec-validator==0.2.9",
        "opencv-python==4.5.1.48",
        "pillow==8.1.0",
        "plotly==4.14.1",
        "pyrsistent==0.17.3; python_version >= '3.5'",
        "pyyaml==5.3.1",
        "requests==2.25.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
        "retrying==1.3.3",
        "six==1.15.0; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3'",
        "swagger-ui-bundle==0.0.8",
        "torch==1.7.1",
        "torchvision==0.8.2",
        "tqdm==4.55.1",
        "typing-extensions==3.7.4.3; python_version < '3.8'",
        "urllib3==1.26.2; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4' and python_version < '4'",
        "werkzeug==1.0.1; python_version >= '2.7' and python_version not in '3.0, 3.1, 3.2, 3.3, 3.4'",
        "zipp==3.4.0; python_version >= '3.6'",
    ],  # Optional
    extras_require={"dev": []},  # Optional
)
