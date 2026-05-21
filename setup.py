from setuptools import setup, find_packages

setup(
    name="wsi-model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
    ],
    author="Your Name",
    description="WSI Model Project",
    python_requires=">=3.8",
)
