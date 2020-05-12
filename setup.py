import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Group-43", # Replace with your own username
    version="0.0.1",
    author="Bear & Ning",
    author_email="gxie319@aucklanduni.ac.nz, ngon152@aucklanduni.ac.nz",
    description="A small models package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UOA-CS302-2020/CS302-Python-2020-Group43",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
