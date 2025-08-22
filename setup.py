from setuptools import setup, find_packages

setup(
    name="est_numpy_awm",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["numpy>=2.3.2"],
)