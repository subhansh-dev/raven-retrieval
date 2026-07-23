"""Raven-Retrieval: Late-interaction retrieval meets hierarchical summarization trees.

19 retrieval pipelines benchmarked head-to-head on BEIR datasets
with proper statistical significance testing.
"""

from setuptools import setup, find_packages

setup(
    name="raven-retrieval",
    version="0.3.0",
    description="Late-interaction retrieval meets hierarchical summarization trees",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="subhansh-dev",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "rank-bm25>=0.2.2",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "full": [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.4",
            "umap-learn>=0.5.3",
            "beir>=2.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "raven-benchmark=run_enhanced_benchmark:main",
            "raven-report=src.eval.report:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
    ],
)
