from setuptools import setup, find_packages

setup(
    name="adaptive_time",
    description="Adaptive Time Algorithms",
    version="0.1",
    python_requires=">=3.9",
    install_requires=[],
    extras_requires={},
    packages=find_packages(include=["adaptive_time"]),
    include_package_data=True,
)
