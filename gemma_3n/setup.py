#!/usr/bin/env python3
"""
Setup script for Gemma 3 4B Kazakh Legal Expert Model
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gemma-3-4b-kazakh-legal",
    version="1.0.0",
    author="Nurlykhan",
    author_email="your.email@example.com",
    description="Специализированная языковая модель для казахского права на базе Gemma 3 4B",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gemma-3-4b-kazakh-legal",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Legal Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Natural Language :: Kazakh",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
        ],
        "monitoring": [
            "wandb>=0.16.0",
            "tensorboard>=2.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kazakh-legal-api=api_server:main",
            "kazakh-legal-test=test_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.jsonl"],
    },
    keywords=[
        "artificial intelligence",
        "natural language processing",
        "legal expert system",
        "kazakh law",
        "gemma",
        "mlx",
        "lora",
        "fine-tuning"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/gemma-3-4b-kazakh-legal/issues",
        "Source": "https://github.com/yourusername/gemma-3-4b-kazakh-legal",
        "Documentation": "https://github.com/yourusername/gemma-3-4b-kazakh-legal#readme",
    },
)
