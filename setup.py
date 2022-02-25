
import setuptools

setuptools.setup(
    name="lifelog_utils",
    version="0.1",
    author="Allie Tran",
    author_email="ly.tran2@mail.dcu.ie",
    description="Various NLP tools for lifelog data",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(exclude=["tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
