from setuptools import setup, find_packages

setup(
    name="vlense",
    version="0.1",
    packages=find_packages("vlense"),
    description="A Python package to extract text from images and PDFs using Vision Language Model (VLM).",
    author="Aditya Miskin",
    author_email="adityamiskin98@gmail.com",
    url="https://github.com/adityamiskin/vlense",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "aiofiles==24.1.0",
        "aioshutil==1.5",
        "litellm==1.51.2",
        "pdf2image==1.17.0",
        "pydantic==2.9.2",
        "asyncio==3.4.3",
        "pdf2image==1.17.0",
    ],
)
