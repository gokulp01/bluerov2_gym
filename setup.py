# setup.py
from setuptools import setup, find_packages

setup(
    name="bluerov_gym",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "bluerov_gym": ["assets/*"],
    },
)
