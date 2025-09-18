#!/usr/bin/env python3
"""
Setup script for Kazakh Legal AI - Fine-tuning Gemma Models
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
    name="kazakh-legal-ai",
    version="1.0.0",
    author="Nurlykhan",
    author_email="your.email@example.com",
    description="Специализированные языковые модели для казахского права на базе Gemma",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kazakh-legal-ai",
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
        "Natural Language :: Russian",
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
        "database": [
            "sqlalchemy>=2.0.0",
            "alembic>=1.12.0",
        ],
        "caching": [
            "redis>=4.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kazakh-legal-api=gemma_3n.api_server:main",
            "kazakh-legal-test=gemma_3n.test_model:main",
            "kazakh-legal-1b=gemma_1b.quick_legal_model:main",
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
        "fine-tuning",
        "machine learning",
        "nlp",
        "legal ai",
        "kazakhstan"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/kazakh-legal-ai/issues",
        "Source": "https://github.com/yourusername/kazakh-legal-ai",
        "Documentation": "https://github.com/yourusername/kazakh-legal-ai#readme",
        "Changelog": "https://github.com/yourusername/kazakh-legal-ai/blob/main/CHANGELOG.md",
    },
)
