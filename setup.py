from setuptools import setup, find_packages

setup(
    name="hangman-eval",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "inspect-ai>=0.3.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0",
            "isort>=5.0",
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
    author="Matt Fisher",
    author_email="matt@example.com",
    description="Hangman game evaluation for AI models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hangman-eval",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
