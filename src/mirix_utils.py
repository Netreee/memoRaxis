from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from .logger import get_logger
from .config import get_config

logger = get_logger()

def get_mirix_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load MIRIX agent configuration from config/mirix_config.yaml.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "mirix_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"MIRIX configuration file not found at {config_path}")
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            mirix_config = yaml.safe_load(f) or {}
        return mirix_config
    except Exception as e:
        logger.error(f"Error loading MIRIX config: {e}")
        return {}

def get_mirix_connection_info() -> Dict[str, str]:
    """
    Get MIRIX connection info (api_key, base_url) from config/config.yaml.
    """
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    credentials = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                app_config = yaml.safe_load(f) or {}
                credentials = app_config.get("mirix", {})
        except Exception as e:
            logger.warning(f"Failed to load credentials from config.yaml: {e}")
    return credentials
