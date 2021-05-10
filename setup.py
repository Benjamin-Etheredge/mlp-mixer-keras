import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlp-mixer-keras",
    version="0.0.1",
    author="Benjamin Etheredge",
    author_email="",
    description="An implementation of MLP-Mixer in Keras/Tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Benjamin-Etheredge/mlp-mixer-keras",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)