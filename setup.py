
from setuptools import setup, find_packages

setup(
    name='ai-dataset-generator',
    version='0.1',
    author='Humboldt University Berlin, deepset GmbH',
    description='Generate datasets with large language models.',
    package_dir={"": "src"},
    packages=find_packages("src"),
    license="Apache 2.0",
    python_requires=">=3.8.0",
)
