from setuptools import setup, find_packages


def requirements():
    with open("requirements.txt", "r") as f:
        return f.read().splitlines()


setup(
    name='fabricator',
    version='0.1.1',
    author='Humboldt University Berlin, deepset GmbH',
    author_email="goldejon@informatik.hu-berlin.de",
    description='Conveniently generating datasets with large language models.',
    long_description="If you require textual datasets for specific tasks, you can utilize large language models "
                     "that possess immense generation capability. fabricator enables you to conveniently create "
                     "or annotate datasets to fine-tune your custom model. fabricator is constructed on "
                     "deepset's haystack and huggingface's datasets libraries, to seamlessly integrate "
                     "into existing NLP frameworks.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    license="Apache 2.0",
    python_requires=">=3.8",
    install_requires=requirements(),
)
