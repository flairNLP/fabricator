from pathlib import Path
from setuptools import setup, find_packages


def requirements():
    with open("requirements.txt", "r") as f:
        return f.read().splitlines()


setup(
    name='fabricator-ai',
    version='0.1.1',
    author='Humboldt University Berlin, deepset GmbH',
    author_email="goldejon@informatik.hu-berlin.de",
    description='Conveniently generating datasets with large language models.',
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages("src"),
    license="Apache 2.0",
    python_requires=">=3.8",
    install_requires=requirements(),
)
