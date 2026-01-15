from setuptools import setup, find_packages

setup(
    name="falcon-core",
    version="0.2.0",
    description="Falcon Trading Platform - Core Libraries",
    author="TradingAsBuddies",
    url="https://github.com/TradingAsBuddies/falcon-core",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "requests>=2.32.3",
        "python-dotenv>=1.0.0",
        "pytz>=2023.3",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "postgresql": ["psycopg2-binary>=2.9.9"],
        "backtesting": [
            "bt>=0.2.9",          # MIT license - event-driven backtesting
            "ffn>=0.3.6",         # MIT license - financial functions
            "yfinance>=0.2.28",   # Apache 2.0 - historical data
        ],
        "full": [
            "psycopg2-binary>=2.9.9",
            "bt>=0.2.9",
            "ffn>=0.3.6",
            "yfinance>=0.2.28",
        ],
    },
    entry_points={
        "console_scripts": [
            "falcon-backtest=falcon_core.backtesting.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
