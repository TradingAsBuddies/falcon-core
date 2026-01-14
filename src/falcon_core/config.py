#!/usr/bin/env python3
"""
Configuration Management for Falcon Trading Platform
Handles FHS-compliant configuration loading with fallbacks
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FalconConfig:
    """
    Falcon configuration manager with FHS compliance

    Configuration precedence (highest to lowest):
    1. Environment variables
    2. /etc/falcon/secrets.env (production secrets)
    3. /etc/falcon/config.conf (production config)
    4. ~/.config/falcon/config (user config)
    5. ./config (local development)
    6. Built-in defaults
    """

    # FHS-compliant default paths
    PRODUCTION_CONFIG_DIR = Path('/etc/falcon')
    PRODUCTION_DATA_DIR = Path('/var/lib/falcon')
    PRODUCTION_CACHE_DIR = Path('/var/cache/falcon')
    PRODUCTION_LOG_DIR = Path('/var/log/falcon')

    def __init__(self, env: str = 'auto'):
        """
        Initialize configuration

        Args:
            env: Environment mode - 'production', 'development', or 'auto' (detect)
        """
        self.env = self._detect_environment() if env == 'auto' else env
        self.config: Dict[str, Any] = {}
        self._load_config()

        logger.info(f"Configuration loaded: {self.env} mode")

    def _detect_environment(self) -> str:
        """
        Detect if running in production or development

        Returns 'production' if:
        - Running as 'falcon' user
        - /etc/falcon/ exists
        - Explicitly set via FALCON_ENV

        Otherwise returns 'development'
        """
        # Check environment variable
        if os.getenv('FALCON_ENV'):
            return os.getenv('FALCON_ENV')

        # Check if running as falcon service user
        if os.getuid() != 0:  # Not root
            try:
                import pwd
                username = pwd.getpwuid(os.getuid()).pw_name
                if username == 'falcon':
                    return 'production'
            except Exception:
                pass

        # Check if production config exists
        if self.PRODUCTION_CONFIG_DIR.exists():
            return 'production'

        return 'development'

    def _load_config(self):
        """Load configuration from all sources"""
        # Start with defaults
        self.config = self._get_defaults()

        # Load from files based on environment
        if self.env == 'production':
            self._load_production_config()
        else:
            self._load_development_config()

        # Environment variables always override (highest priority)
        self._load_from_env()

    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values"""
        if self.env == 'production':
            base_dir = self.PRODUCTION_DATA_DIR
            config_dir = self.PRODUCTION_CONFIG_DIR
            cache_dir = self.PRODUCTION_CACHE_DIR
            log_dir = self.PRODUCTION_LOG_DIR
        else:
            # Development mode - use local directories
            base_dir = Path.cwd()
            config_dir = base_dir
            cache_dir = base_dir / 'cache'
            log_dir = base_dir / 'logs'

        return {
            # Environment
            'env': self.env,

            # Directories
            'base_dir': str(base_dir),
            'config_dir': str(config_dir),
            'cache_dir': str(cache_dir),
            'log_dir': str(log_dir),
            'market_data_dir': str(base_dir / 'market_data'),

            # Database
            'db_type': 'sqlite',
            'db_path': str(base_dir / 'paper_trading.db'),
            'db_host': 'localhost',
            'db_port': 5432,
            'db_name': 'falcon',
            'db_user': 'falcon',
            'db_password': '',

            # API Keys (will be overridden from secrets file or env)
            'massive_api_key': '',
            'claude_api_key': '',
            'openai_api_key': '',
            'perplexity_api_key': '',

            # Trading
            'initial_balance': 10000.0,
            'max_positions': 10,
            'max_position_size': 0.1,  # 10% of portfolio

            # Screener
            'finviz_screener_url': '',

            # Server
            'flask_host': '127.0.0.1',
            'flask_port': 5000,
            'flask_debug': self.env == 'development',
        }

    def _load_production_config(self):
        """Load production configuration from /etc/falcon/"""

        # Load main config file
        config_file = self.PRODUCTION_CONFIG_DIR / 'config.conf'
        if config_file.exists():
            self._load_env_file(config_file)
            logger.info(f"Loaded config from {config_file}")

        # Load secrets file (higher priority)
        secrets_file = self.PRODUCTION_CONFIG_DIR / 'secrets.env'
        if secrets_file.exists():
            self._load_env_file(secrets_file)
            logger.info(f"Loaded secrets from {secrets_file}")
        else:
            logger.warning(
                f"Secrets file not found: {secrets_file}. "
                "API keys may not be configured."
            )

    def _load_development_config(self):
        """Load development configuration"""

        # Try local .env file
        local_env = Path.cwd() / '.env'
        if local_env.exists():
            self._load_env_file(local_env)
            logger.info(f"Loaded config from {local_env}")

        # Try ~/.local/.env (current development setup)
        user_env = Path.home() / '.local' / '.env'
        if user_env.exists():
            self._load_env_file(user_env)
            logger.info(f"Loaded config from {user_env}")

        # Try ~/.config/falcon/config
        user_config = Path.home() / '.config' / 'falcon' / 'config'
        if user_config.exists():
            self._load_env_file(user_config)
            logger.info(f"Loaded config from {user_config}")

    def _load_env_file(self, filepath: Path):
        """Load key=value pairs from a file"""
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue

                    # Parse key=value
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]

                        # Map environment variable names to config keys
                        config_key = self._env_to_config_key(key)
                        if config_key:
                            self.config[config_key] = self._cast_value(value)

        except Exception as e:
            logger.error(f"Error loading config file {filepath}: {e}")

    def _load_from_env(self):
        """Load configuration from environment variables (highest priority)"""
        env_mappings = {
            'FALCON_ENV': 'env',
            'DB_TYPE': 'db_type',
            'DB_PATH': 'db_path',
            'DB_HOST': 'db_host',
            'DB_PORT': 'db_port',
            'DB_NAME': 'db_name',
            'DB_USER': 'db_user',
            'DB_PASSWORD': 'db_password',
            'MASSIVE_API_KEY': 'massive_api_key',
            'CLAUDE_API_KEY': 'claude_api_key',
            'OPENAI_API_KEY': 'openai_api_key',
            'PERPLEXITY_API_KEY': 'perplexity_api_key',
            'FINVIZ_SCREENER_URL': 'finviz_screener_url',
            'FLASK_HOST': 'flask_host',
            'FLASK_PORT': 'flask_port',
            'FLASK_DEBUG': 'flask_debug',
        }

        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self.config[config_key] = self._cast_value(value)

    def _env_to_config_key(self, env_key: str) -> Optional[str]:
        """Convert environment variable name to config key"""
        # Map common environment variable names
        mappings = {
            'DB_TYPE': 'db_type',
            'DB_PATH': 'db_path',
            'DB_HOST': 'db_host',
            'DB_PORT': 'db_port',
            'DB_NAME': 'db_name',
            'DB_USER': 'db_user',
            'DB_PASSWORD': 'db_password',
            'MASSIVE_API_KEY': 'massive_api_key',
            'CLAUDE_API_KEY': 'claude_api_key',
            'OPENAI_API_KEY': 'openai_api_key',
            'PERPLEXITY_API_KEY': 'perplexity_api_key',
            'FINVIZ_SCREENER_URL': 'finviz_screener_url',
            'FLASK_HOST': 'flask_host',
            'FLASK_PORT': 'flask_port',
            'FLASK_DEBUG': 'flask_debug',
            'INITIAL_BALANCE': 'initial_balance',
        }

        return mappings.get(env_key, env_key.lower())

    def _cast_value(self, value: str) -> Any:
        """Cast string value to appropriate type"""
        # Boolean
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        if value.lower() in ('false', 'no', '0', 'off'):
            return False

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # String (default)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dict syntax"""
        return self.config[key]

    def __getattr__(self, key: str) -> Any:
        """Get configuration value using attribute syntax"""
        if key in self.config:
            return self.config[key]
        raise AttributeError(f"Configuration key '{key}' not found")

    def get_db_config(self) -> Dict[str, Any]:
        """Get database configuration for DatabaseManager"""
        return {
            'db_type': self.config['db_type'],
            'db_path': self.config['db_path'],
            'db_host': self.config['db_host'],
            'db_port': self.config['db_port'],
            'db_name': self.config['db_name'],
            'db_user': self.config['db_user'],
            'db_password': self.config['db_password'],
        }

    def validate(self) -> bool:
        """
        Validate configuration

        Returns:
            True if configuration is valid, False otherwise
        """
        errors = []

        # Check required API keys for production
        if self.env == 'production':
            if not self.config.get('massive_api_key'):
                errors.append("MASSIVE_API_KEY is required for trading")

        # Check database configuration
        if self.config['db_type'] == 'postgresql':
            required = ['db_host', 'db_name', 'db_user', 'db_password']
            for field in required:
                if not self.config.get(field):
                    errors.append(f"PostgreSQL requires {field.upper()}")

        # Check directories exist (production mode)
        if self.env == 'production':
            for dir_key in ['base_dir', 'config_dir']:
                dir_path = Path(self.config[dir_key])
                if not dir_path.exists():
                    errors.append(f"Directory does not exist: {dir_path}")

        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False

        logger.info("Configuration validation successful")
        return True

    def print_summary(self):
        """Print configuration summary (excluding secrets)"""
        print("Falcon Configuration Summary")
        print("=" * 50)
        print(f"Environment: {self.config['env']}")
        print(f"Base Directory: {self.config['base_dir']}")
        print(f"Config Directory: {self.config['config_dir']}")
        print(f"Database Type: {self.config['db_type']}")

        if self.config['db_type'] == 'sqlite':
            print(f"Database Path: {self.config['db_path']}")
        else:
            print(f"Database Host: {self.config['db_host']}")
            print(f"Database Name: {self.config['db_name']}")

        # API keys (show if configured, not actual values)
        print(f"MASSIVE_API_KEY: {'✓ configured' if self.config.get('massive_api_key') else '✗ not set'}")
        print(f"CLAUDE_API_KEY: {'✓ configured' if self.config.get('claude_api_key') else '✗ not set'}")
        print("=" * 50)


# Global config instance
_config: Optional[FalconConfig] = None


def get_config(env: str = 'auto', reload: bool = False) -> FalconConfig:
    """
    Get global configuration instance

    Args:
        env: Environment mode
        reload: Force reload configuration

    Returns:
        FalconConfig instance
    """
    global _config

    if _config is None or reload:
        _config = FalconConfig(env=env)

    return _config


if __name__ == '__main__':
    # Test configuration loading
    config = get_config()
    config.print_summary()

    if '--validate' in sys.argv:
        valid = config.validate()
        sys.exit(0 if valid else 1)
