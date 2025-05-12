from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Optional

class BaseAgent(ABC):
    """Base class for all agents in the social media content generation system."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent.
        
        Args:
            name (str): Name of the agent
            config (Dict[str, Any], optional): Configuration parameters for the agent
        """
        self.name = name
        self.config = config or {}
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging for the agent."""
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.name}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input data and return the results.
        
        Args:
            input_data (Dict[str, Any]): Input data to process
            
        Returns:
            Dict[str, Any]: Processing results
        """
        pass
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key (str): Configuration key
            default (Any, optional): Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        return self.config.get(key, default)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update the agent's configuration.
        
        Args:
            updates (Dict[str, Any]): Configuration updates
        """
        self.config.update(updates)
        self.logger.info(f"Configuration updated: {updates}")
    
    def validate_input(self, input_data: Dict[str, Any], required_fields: list) -> bool:
        """
        Validate that input data contains all required fields.
        
        Args:
            input_data (Dict[str, Any]): Input data to validate
            required_fields (list): List of required field names
            
        Returns:
            bool: True if valid, False otherwise
        """
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            self.logger.error(f"Missing required fields: {missing_fields}")
            return False
        return True
