# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration Manager for CVDP Benchmark

Handles environment variables, .env files, and configuration management
with support for type validation and secure API key handling.
"""

import os
import logging
from typing import Any, Dict, Optional, Union, Type, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Centralized configuration management for environment variables and .env files.
    
    Features:
    - .env file support for sensitive data
    - Dynamic environment variable registration
    - Type validation and conversion
    - Default value management
    - Security-first approach for API keys
    """
    
    def __init__(self, env_file: Optional[str] = ".env"):
        """
        Initialize ConfigManager with optional .env file loading.
        
        Args:
            env_file: Path to .env file (None to skip loading)
        """
        self.config: Dict[str, Any] = {}
        self._registered_keys: Dict[str, dict] = {}
        
        # Load .env file if it exists
        if env_file:
            self._load_env_file(env_file)
        
        # Setup default configurations from existing codebase
        self._setup_default_configs()
    
    def _load_env_file(self, env_file: str) -> None:
        """Load environment variables from .env file."""
        env_path = Path(env_file)
        if not env_path.exists():
            logger.info(f"No .env file found at {env_path}, skipping")
            return
        
        try:
            with open(env_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse KEY=VALUE format
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        
                        # Set environment variable if not already set
                        if key not in os.environ:
                            os.environ[key] = value
                            logger.debug(f"Loaded {key} from .env file")
                    else:
                        logger.warning(f"Invalid line format in {env_file}:{line_num}: {line}")
        
        except Exception as e:
            logger.error(f"Error loading .env file {env_file}: {e}")
    
    def _setup_default_configs(self) -> None:
        """Setup default configurations found in the existing codebase."""
        # Timeout configurations
        self.register_config("MODEL_TIMEOUT", default=60, type_cast=int, 
                           description="Timeout for model operations in seconds")
        self.register_config("TASK_TIMEOUT", default=300, type_cast=int,
                           description="Timeout for task operations in seconds")
        self.register_config("DOCKER_TIMEOUT", default=600, type_cast=int,
                           description="Timeout for Docker operations in seconds")
        self.register_config("DOCKER_TIMEOUT_AGENT", default=600, type_cast=int,
                           description="Timeout for Docker agent operations in seconds")
        self.register_config("QUEUE_TIMEOUT", default=None, type_cast=int,
                           description="Queue timeout in seconds")
        
        # API Keys (not required during initialization, validated when needed)
        self.register_config("OPENAI_USER_KEY", required=False, 
                           description="OpenAI API key for model access")
        self.register_config("ANTHROPIC_API_KEY", required=False,
                           description="Anthropic API key for Claude model access")
        self.register_config("NVIDIA_API_KEY", required=False,
                           description="NVIDIA API key for model access")
              
        # Application Configuration
        self.register_config("ENABLE_SUBJECTIVE_SCORING", default=False, type_cast=bool,
                           description="Enable LLM-based subjective scoring")
        self.register_config("BENCHMARK_THREADS", default=1, type_cast=int,
                           description="Number of parallel threads for benchmark execution")
        self.register_config("BENCHMARK_PREFIX", default="work", type_cast=str,
                           description="Prefix for output directories")
        self.register_config("DEFAULT_MODEL", default="o4-mini", type_cast=str,
                           description="Default model to use when none is specified")
        self.register_config("CUSTOM_MODEL_FACTORY", required=False,
                           description="Path to custom model factory implementation")
        
        # Docker and Resource Management
        self.register_config("DOCKER_QUOTA_THRESHOLD_MB", default=50, type_cast=int,
                           description="Docker quota threshold in MB")
        self.register_config("DOCKER_QUOTA_CHECK_INTERVAL", default=1, type_cast=int,
                           description="Docker quota check interval in seconds")
        self.register_config("DOCKER_QUOTA_MIN_COMPRESS_SIZE_MB", default=10, type_cast=int,
                           description="Minimum file size in MB before compression")
        
        # EDA Tool Infrastructure Configuration
        self.register_config("VERIF_EDA_IMAGE", default="cvdp-cadence-verif:latest", type_cast=str,
                           description="Docker image for verification tasks with commercial EDA tools")
        self.register_config("LICENSE_NETWORK", default="licnetwork", type_cast=str,
                           description="Docker network name for EDA license server connectivity")
        self.register_config("LICENSE_NETWORK_AUTO_CREATE", default=True, type_cast=bool,
                           description="Automatically create license network if it doesn't exist")
        self.register_config("OSS_SIM_IMAGE", default="ghcr.io/hdl/sim/osvb", type_cast=str,
                           description="Docker image for simulation tasks with open-source EDA tools")
        self.register_config("OSS_PNR_IMAGE", default="ghcr.io/hdl/impl/pnr", type_cast=str,
                           description="Docker image for place-and-route tasks with open-source EDA tools")
    
    def register_config(self, 
                       key: str, 
                       default: Any = None, 
                       type_cast: Type = str, 
                       required: bool = False,
                       description: str = "") -> None:
        """
        Register a configuration key with validation and type casting.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            type_cast: Type to cast the value to (int, str, bool, float)
            required: Whether this configuration is required
            description: Human-readable description of the config
        """
        self._registered_keys[key] = {
            'default': default,
            'type_cast': type_cast,
            'required': required,
            'description': description
        }
        
        # Immediately validate and cache the value
        self._validate_and_cache(key)
    
    def _validate_and_cache(self, key: str) -> None:
        """Validate and cache a configuration value."""
        config = self._registered_keys[key]
        raw_value = os.getenv(key)
        
        # Handle required keys
        if config['required'] and raw_value is None:
            raise ValueError(f"Required configuration '{key}' is missing. {config['description']}")
        
        # Use default if not found
        if raw_value is None:
            self.config[key] = config['default']
            return
        
        # Type casting with error handling
        try:
            if config['type_cast'] == bool:
                # Handle boolean conversion
                self.config[key] = raw_value.lower() in ('true', '1', 'yes', 'on')
            elif config['type_cast'] == int:
                self.config[key] = int(raw_value)
            elif config['type_cast'] == float:
                self.config[key] = float(raw_value)
            else:
                self.config[key] = config['type_cast'](raw_value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid value for {key}: '{raw_value}'. Using default: {config['default']}")
            self.config[key] = config['default']
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with optional runtime default.
        
        Args:
            key: Configuration key
            default: Runtime default (overrides registered default)
            
        Returns:
            Configuration value
        """
        if key in self.config:
            return self.config[key]
        
        # Handle unregistered keys
        raw_value = os.getenv(key, default)
        if raw_value is None:
            return default
        
        # Try to intelligently cast the value
        try:
            # Try int first
            if raw_value.isdigit():
                return int(raw_value)
            # Try float
            elif '.' in raw_value and raw_value.replace('.', '').isdigit():
                return float(raw_value)
            # Try boolean
            elif raw_value.lower() in ('true', 'false', '1', '0', 'yes', 'no'):
                return raw_value.lower() in ('true', '1', 'yes')
            else:
                return raw_value
        except (ValueError, AttributeError):
            return raw_value
    
    def get_api_keys(self) -> Dict[str, str]:
        """
        Get all registered API keys that have values.
        
        Returns:
            Dictionary of API key names to values
        """
        api_keys = {}
        for key, config in self._registered_keys.items():
            if 'KEY' in key:
                value = self.config.get(key)
                if value is not None and value != "":
                    api_keys[key] = value
        return api_keys
    
    def get_timeouts(self) -> Dict[str, int]:
        """
        Get all timeout configurations.
        
        Returns:
            Dictionary of timeout names to values
        """
        timeouts = {}
        for key, value in self.config.items():
            if 'TIMEOUT' in key and value is not None:
                timeouts[key] = value
        return timeouts
    
    def mark_required(self, *keys: str) -> None:
        """
        Mark specific configuration keys as required.
        
        Args:
            keys: Configuration keys to mark as required
        """
        for key in keys:
            if key in self._registered_keys:
                self._registered_keys[key]['required'] = True
                # Don't re-validate immediately, just mark as required
    
    def validate_required(self) -> List[str]:
        """
        Validate all required configurations are present.
        
        Returns:
            List of missing required configuration keys
        """
        missing = []
        for key, config in self._registered_keys.items():
            if config['required'] and self.config.get(key) is None:
                missing.append(key)
        return missing
    
    def summary(self) -> str:
        """
        Get a summary of all registered configurations.
        
        Returns:
            Human-readable configuration summary
        """
        lines = ["Configuration Summary:", "=" * 50]
        
        for key, config in self._registered_keys.items():
            value = self.config.get(key, "NOT_SET")
            
            # Mask API keys for security
            if 'KEY' in key and value is not None and value != "NOT_SET" and str(value) != "None":
                display_value = f"{str(value)[:8]}..." if len(str(value)) > 8 else "***"
            else:
                display_value = value
            
            status = "✓" if value != "NOT_SET" else "✗"
            required = " (REQUIRED)" if config['required'] else ""
            
            lines.append(f"{status} {key}: {display_value}{required}")
            if config['description']:
                lines.append(f"   {config['description']}")
            lines.append("")
        
        return "\n".join(lines)


# Global configuration manager instance
config = ConfigManager() 
