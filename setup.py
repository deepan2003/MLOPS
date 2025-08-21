from setuptools import setup, find_packages

setup(
    name="heart-attack-mlops",
    version="0.1.0",
    description="Heart Attack Prediction MLOps Project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
        "joblib>=1.0.0",
        "PyYAML>=5.4.0",
        "Flask>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.9",
            "black>=21.0",
            "bandit>=1.7",
            "tox>=3.0",
        ]
    }
)
