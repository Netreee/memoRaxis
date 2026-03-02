from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from .logger import get_logger
from .config import get_config

logger = get_logger()

def get_mirix_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load MIRIX configuration.
    
    If config_path is not provided, tries to load from default location:
    config/mirix_config.yaml relative to project root.
    """
    if config_path is None:
        # Assuming project root structure from src/mirix_utils.py -> ../config/mirix_config.yaml
        config_path = Path(__file__).parent.parent / "config" / "mirix_config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"MIRIX configuration file not found at {config_path}")
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            mirix_config = yaml.safe_load(f) or {}
            
        # Optionally merge with API credentials from main config if needed
        # But user asked to remove mirix from config, so maybe we pull keys from here or main config?
        # The main config had 'mirix' section with api_key and base_url.
        # If we remove that from Config class properties, we might still access it via _app_config if we didn't remove it from the yaml file.
        # However, the user said "remove everything about mirix from config".
        # So we should probably expect everything to be in mirix_config.yaml or handled here.
        
        # Let's check config.yaml content again. It has a 'mirix' section.
        # If user wants to remove 'mirix' property from Config class, we can't access conf.mirix.
        # But we can still access conf._app_config if we really needed to, but better to encapsulate here.
        
        # For now, let's just return what's in mirix_config.yaml as the agent config.
        # The user might want to move api_key/base_url here too, or keep them in main config but process them here?
        # Let's stick to returning the agent config for now, as that's what initialize_meta_agent needs.
        
        return mirix_config
        
    except Exception as e:
        logger.error(f"Error loading MIRIX config: {e}")
        return {}

def get_mirix_connection_info() -> Dict[str, str]:
    """
    Get MIRIX connection info (api_key, base_url) from main config.
    Since we are removing 'mirix' property from Config, we might need to access it differently 
    or just recommend putting it in mirix_config.yaml?
    
    The user said "remove everything about mirix from config".
    If I remove the property `mirix` from `Config` class, `get_config().mirix` won't work.
    But `get_config()._app_config["mirix"]` would still work if the yaml has it.
    
    However, the cleanest way is if `mirix_config.yaml` contains everything or we look at `config.yaml` manually here.
    Let's assume we read `config/config.yaml` for connection info if strictly separated, 
    OR we assume `mirix_config.yaml` will also hold connection info?
    
    The user's previous request added `mirix_config.yaml` with agent config.
    `config.yaml` had api_key.
    
    I will retrieve `mirix` section from `config.yaml` manually here using `get_config()._app_config.get("mirix", {})`.
    Wait, `_app_config` is internal. 
    
    Let's look at `mem0_utils.py`. It calls `get_config()`.
    Use `get_config()` to get global config, but `Config` class won't have `mirix` property.
    We can rely on `_app_config` being accessible if we don't change `Config` visibility? 
    `_app_config` is private.
    
    Maybe I should act as if `config/config.yaml` still has the `mirix` section (data), 
    but the `Config` class (code) doesn't have the explicit property.
    So I can add a method or just parse the file again? Parsing file again seems redundant.
    
    Actually, if I remove `mirix` property from `Config`, I can't access it easily.
    Maybe I'll just load `config/config.yaml` inside `get_mirix_connection_info` or `get_mirix_config`.
    
    Or, I can combine everything into `mirix_config.yaml`.
    
    Let's stick to: `get_mirix_config` retrieves the agent configuration.
    And maybe `get_mirix_credentials` retrieves connection info.
    
    Let's assume the user wants `mirix_utils.py` to be the single point of truth for mirix config.
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
