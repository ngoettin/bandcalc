## IMPORTANT:
# This is a really rudimentary setup file. It does not cover
# dependencies (like ray, cupy ...) which you will have to
# install yourself.

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
        name="bandcalc",
        version="0.0.1",
        author="Niclas Götting",
        description="Calculate bandstrucuteres and moire properties",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/ngoettin/band-calc",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            ],
        python_requires=">=3.6"
)
