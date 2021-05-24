import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlp-mixer-keras",
    version="0.0.4",
    license='MIT',
    author="Benjamin Etheredge",
    author_email="",
    description="MLP-Mixer in Keras/Tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Benjamin-Etheredge/mlp-mixer-keras",
    packages=setuptools.find_packages(),
    install_requires=[
        'tensorflow>=2.1'
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
)
