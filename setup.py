from setuptools import setup, find_packages

setup(
    name="falcon-core",
    version="0.1.0",
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
    ],
    extras_require={
        "postgresql": ["psycopg2-binary>=2.9.9"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
