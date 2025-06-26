from setuptools import setup, find_packages

setup(
    name="data-gen",
    version="0.1.0",
    author="Fahad Abdullah",
    author_email="fahadai.co@gmail.com",
    description="Unified synthetic data generation for tabular, time-series, image, and text data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/FAbdullah17/data-gen",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "sdv>=1.6.1",
        "ydata-synthetic>=0.5.0",
        # gretel-synthetics must be manually installed with --no-deps
        "torch>=1.11.0",
        "torchvision>=0.12.0",
        "transformers>=4.40.0",   # For GPT models
        "accelerate>=0.22.0",     # Optional for performance
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.6.0",
        "plotly>=5.10.0",
        "pyyaml>=6.0",
        "jsonschema>=4.16.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
