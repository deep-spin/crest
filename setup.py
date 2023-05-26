from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="CREST",
    version="0.0.1",
    description="CREST",
    long_description=readme,
    author="Marcos V Treviso",
    author_email="marcos.treviso@tecnico.ulisboa.com",
    url="https://github.com/deep-spin/crest",
    license=license,
    packages=find_packages(exclude=("tests", "docs")),
    data_files=["LICENSE"],
    zip_safe=False,
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
    ],
)
