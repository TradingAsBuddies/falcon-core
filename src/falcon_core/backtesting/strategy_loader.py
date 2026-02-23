#!/usr/bin/env python3
"""
Strategy Plugin Loader

Loads trading strategy plugins from multiple sources:
  - Database (strategy_roster.strategy_code)
  - File system (FALCON_STRATEGY_DIR, default /var/lib/falcon/strategies/)
  - Python packages (standard imports)

Strategies are self-contained Python modules that define a BaseStrategy subclass.
They import only from the public contract: base.py (BaseStrategy, StrategyParams,
Signal, SignalType) plus standard libs (pandas, numpy, datetime, etc.).
"""

import ast
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

from falcon_core.backtesting.strategies.base import (
    BaseStrategy,
    Signal,
    SignalType,
    StrategyParams,
)

logger = logging.getLogger(__name__)

# Modules allowed in strategy plugin code
ALLOWED_IMPORTS = frozenset({
    'pandas', 'pd',
    'numpy', 'np',
    'math',
    'datetime',
    'dataclasses',
    'typing',
    'collections',
    'itertools',
    'functools',
    'decimal',
    'statistics',
    'json',
    'falcon_core.backtesting.strategies.base',
    'falcon_core.backtesting.strategies',
})

# Modules explicitly forbidden in strategy plugins
FORBIDDEN_IMPORTS = frozenset({
    'subprocess', 'os', 'sys', 'shutil', 'pathlib',
    'socket', 'http', 'urllib', 'requests',
    'importlib', 'ctypes', 'multiprocessing',
    'signal', 'threading', 'asyncio',
    'pickle', 'shelve', 'marshal',
    'code', 'codeop', 'compile', 'compileall',
    'builtins', '__builtin__',
})

# Forbidden function calls
FORBIDDEN_CALLS = frozenset({
    'eval', 'exec', 'compile', '__import__',
    'getattr', 'setattr', 'delattr',
    'globals', 'locals', 'vars',
    'open', 'input', 'breakpoint',
})


def validate_strategy_code(code: str) -> Tuple[bool, str]:
    """Validate strategy plugin code for safety and correctness.

    Checks:
      - Valid Python syntax (AST parse)
      - No forbidden imports (subprocess, os, etc.)
      - No forbidden calls (eval, exec, open, etc.)
      - At least one BaseStrategy subclass defined

    Returns:
        (is_valid, error_message) — error_message is empty on success
    """
    # Parse the AST
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    has_strategy_subclass = False

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_root = alias.name.split('.')[0]
                if module_root in FORBIDDEN_IMPORTS:
                    return False, f"Forbidden import: {alias.name}"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_root = node.module.split('.')[0]
                if module_root in FORBIDDEN_IMPORTS:
                    return False, f"Forbidden import: from {node.module}"

        # Check for forbidden function calls
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                return False, f"Forbidden call: {func.id}()"
            elif isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_CALLS:
                return False, f"Forbidden call: .{func.attr}()"

        # Check for BaseStrategy subclass
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name == 'BaseStrategy':
                    has_strategy_subclass = True

    if not has_strategy_subclass:
        return False, "No BaseStrategy subclass found"

    return True, ""


def load_strategy_from_code(code: str, name: str) -> Optional[Type[BaseStrategy]]:
    """Load a strategy class from source code string.

    Executes code in a controlled namespace that provides the base classes
    and standard library modules strategies are allowed to use.

    Args:
        code: Python source code defining a BaseStrategy subclass
        name: Strategy name (for logging)

    Returns:
        The BaseStrategy subclass, or None on failure
    """
    import dataclasses
    import datetime
    import math
    import typing

    try:
        import numpy as np
    except ImportError:
        np = None

    try:
        import pandas as pd
    except ImportError:
        pd = None

    # Build the execution namespace
    namespace = {
        # Base classes (the public contract)
        'BaseStrategy': BaseStrategy,
        'StrategyParams': StrategyParams,
        'Signal': Signal,
        'SignalType': SignalType,
        # Standard library
        'math': math,
        'datetime': datetime,
        'dataclasses': dataclasses,
        'dataclass': dataclasses.dataclass,
        'field': dataclasses.field,
        'typing': typing,
        'Dict': typing.Dict,
        'List': typing.List,
        'Optional': typing.Optional,
        'Tuple': typing.Tuple,
        'Any': typing.Any,
        # Data science
        'pd': pd,
        'pandas': pd,
        'np': np,
        'numpy': np,
    }

    try:
        exec(code, namespace)  # noqa: S102
    except Exception as e:
        logger.error(f"Failed to execute strategy code for '{name}': {e}")
        return None

    # Find the BaseStrategy subclass
    for obj in namespace.values():
        if (isinstance(obj, type)
                and issubclass(obj, BaseStrategy)
                and obj is not BaseStrategy):
            return obj

    logger.error(f"No BaseStrategy subclass found in code for '{name}'")
    return None


def load_strategies_from_db(db) -> Dict[str, Type[BaseStrategy]]:
    """Load strategy plugins from the database.

    Reads strategy_code from strategy_roster where code is present,
    validates and loads each.

    Args:
        db: DatabaseManager instance

    Returns:
        Dict mapping strategy_name to strategy class
    """
    strategies = {}

    try:
        rows = db.execute(
            'SELECT strategy_name, strategy_code FROM strategy_roster '
            'WHERE strategy_code IS NOT NULL',
            fetch='all'
        )
    except Exception as e:
        logger.warning(f"Could not query strategy_roster for code: {e}")
        return strategies

    if not rows:
        return strategies

    for row in rows:
        name = row['strategy_name'] if isinstance(row, dict) else row[0]
        code = row['strategy_code'] if isinstance(row, dict) else row[1]

        if not code or not code.strip():
            continue

        is_valid, error = validate_strategy_code(code)
        if not is_valid:
            logger.warning(f"Strategy '{name}' failed validation: {error}")
            continue

        cls = load_strategy_from_code(code, name)
        if cls is not None:
            strategies[name] = cls
            logger.debug(f"Loaded strategy '{name}' from database")

    return strategies


def load_strategies_from_directory(path: str) -> Dict[str, Type[BaseStrategy]]:
    """Load strategy plugins from a directory of .py files.

    Args:
        path: Directory path containing *.py strategy files

    Returns:
        Dict mapping strategy_name to strategy class
    """
    strategies = {}
    dir_path = Path(path)

    if not dir_path.is_dir():
        logger.debug(f"Strategy directory does not exist: {path}")
        return strategies

    for py_file in sorted(dir_path.glob('*.py')):
        if py_file.name.startswith('_'):
            continue

        try:
            code = py_file.read_text()
        except Exception as e:
            logger.warning(f"Could not read {py_file}: {e}")
            continue

        is_valid, error = validate_strategy_code(code)
        if not is_valid:
            logger.warning(f"Strategy file '{py_file.name}' failed validation: {error}")
            continue

        cls = load_strategy_from_code(code, py_file.stem)
        if cls is not None:
            strategy_name = getattr(cls, 'name', py_file.stem)
            strategies[strategy_name] = cls
            logger.debug(f"Loaded strategy '{strategy_name}' from {py_file}")

    return strategies


def get_all_strategies(db=None) -> Dict[str, Type[BaseStrategy]]:
    """Load all strategy plugins from all sources.

    Sources (in order, later sources override on name conflict):
      1. Plugin directory (FALCON_STRATEGY_DIR)
      2. Database (strategy_roster.strategy_code) — takes precedence

    Args:
        db: DatabaseManager instance (optional)

    Returns:
        Dict mapping strategy_name to strategy class
    """
    strategies = {}

    # 1. Load from plugin directory
    strategy_dir = os.getenv(
        'FALCON_STRATEGY_DIR',
        '/var/lib/falcon/strategies/'
    )
    dir_strategies = load_strategies_from_directory(strategy_dir)
    strategies.update(dir_strategies)

    # 2. Load from database (takes precedence)
    if db is not None:
        db_strategies = load_strategies_from_db(db)
        strategies.update(db_strategies)

    logger.info(
        f"Loaded {len(strategies)} strategies "
        f"(dir: {len(dir_strategies)}, db: {len(db_strategies) if db else 0})"
    )

    return strategies
