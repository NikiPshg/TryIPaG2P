from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="TryRuIpa",
    version="0.1.0",
    description="TryRuIpa: A package for grapheme to phoneme.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NikiPshg",
    url="https://github.com/NikiPshg/TryIPaG2P/",
    license="MIT",
    packages=find_packages(where="src/g2p"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements, 
    include_package_data=True, 
)