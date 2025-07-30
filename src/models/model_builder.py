"""
Model Builder using Builder Architecture Pattern

This module provides a flexible model builder that allows creating LLM instances
with various parameters and configurations using a fluent interface.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
import logging

try:
    from openai import OpenAI
    from langchain_openai import ChatOpenAI
    from langchain_community.llms import Ollama
    from langchain_community.chat_models import ChatOllama
    OPENAI_AVAILABLE = True
    OLLAMA_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OLLAMA_AVAILABLE = False

# try:
#     from anthropic import Anthropic
#     from langchain_anthropic import ChatAnthropic
#     ANTHROPIC_AVAILABLE = True
# except ImportError:
#     ANTHROPIC_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration class for model parameters."""
    provider: str = "openai"
    model_name: str = "gpt-4.1-nano"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    retry_attempts: int = 3
    custom_headers: Dict[str, str] = field(default_factory=dict)
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    callbacks: List[Callable] = field(default_factory=list)
    tools: List[Callable] = field(default_factory=list)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def create_client(self, config: ModelConfig) -> Any:
        """Create and return the LLM client."""
        pass
    
    @abstractmethod
    def run(self, client: Any, prompt: str, config: ModelConfig) -> str:
        """Run the LLM with the given prompt and return the response."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""
    
    def create_client(self, config: ModelConfig) -> Any:
        """Create OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        return OpenAI(
            api_key=api_key,
            max_retries=config.retry_attempts
        )
    
    def run(self, client: Any, prompt: str, config: ModelConfig) -> str:
        """Run OpenAI model."""
        messages = []
        
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Build parameters dict with only available attributes
        params = {
            "model": config.model_name,
            "messages": messages,
            "temperature": config.temperature,
        }
        
        # Add optional parameters only if they exist
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens
        
        response = client.chat.completions.create(**params)
        
        return response.choices[0].message.content


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM provider implementation."""
    pass

class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider implementation."""
    
    def create_client(self, config: ModelConfig) -> Any:
        """Create Ollama client."""
        if not OLLAMA_AVAILABLE:
            raise ImportError("LangChain Ollama not available. Install with: pip install langchain-community")
        
        base_url = config.base_url or "http://localhost:11434"
        
        return ChatOllama(
            model=config.model_name,
            base_url=base_url,
            temperature=config.temperature
        )
    
    def run(self, client: Any, prompt: str, config: ModelConfig) -> str:
        """Run Ollama model."""
        messages = []
        
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = client.invoke(messages)
        return response.content


class ModelBuilder:
    """
    Model Builder using the Builder Architecture Pattern.
    
    """
    
    def __init__(self):
        self._config = ModelConfig()
        self._providers = {
            "openai": OpenAIProvider(),
            "ollama": OllamaProvider()
        }
        self._client = None
        self._is_built = False
    
    def with_provider(self, provider: str) -> 'ModelBuilder':
        """Set the LLM provider."""
        if provider not in self._providers:
            raise ValueError(f"Unsupported provider: {provider}. Available: {list(self._providers.keys())}")
        
        self._config.provider = provider
        return self
    
    def with_model(self, model_name: str) -> 'ModelBuilder':
        """Set the model name."""
        self._config.model_name = model_name
        return self
    
    def with_temperature(self, temperature: float) -> 'ModelBuilder':
        """Set the temperature parameter."""
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        self._config.temperature = temperature
        return self
    
    def with_max_tokens(self, max_tokens: int) -> 'ModelBuilder':
        """Set the maximum tokens parameter."""
        if max_tokens <= 0:
            raise ValueError("Max tokens must be positive")
        self._config.max_tokens = max_tokens
        return self
    

    
    def with_api_key(self, api_key: str) -> 'ModelBuilder':
        """Set the API key."""
        self._config.api_key = api_key
        return self
    
    def with_base_url(self, base_url: str) -> 'ModelBuilder':
        """Set the base URL for the API."""
        self._config.base_url = base_url
        return self
    

    
    def with_retry_attempts(self, attempts: int) -> 'ModelBuilder':
        """Set the number of retry attempts."""
        if attempts < 0:
            raise ValueError("Retry attempts must be non-negative")
        self._config.retry_attempts = attempts
        return self
    
    def with_custom_headers(self, headers: Dict[str, str]) -> 'ModelBuilder':
        """Set custom headers."""
        self._config.custom_headers.update(headers)
        return self
    
    def with_system_prompt(self, prompt: str) -> 'ModelBuilder':
        """Set the system prompt."""
        self._config.system_prompt = prompt
        return self
    
    def with_user_prompt(self, prompt: str) -> 'ModelBuilder':
        """Set the user prompt."""
        self._config.user_prompt = prompt
        return self
    
    def with_tools(self, tools: List[Callable]) -> 'ModelBuilder':
        """Set the tools."""
        self._config.tools = tools
        return self
    
    
    def build(self) -> 'ModelBuilder':
        """Build the model and create the client."""
        try:
            provider = self._providers[self._config.provider]
            self._client = provider.create_client(self._config)
            self._is_built = True
            logger.info(f"Model built successfully with provider: {self._config.provider}")
            return self
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise
    
    def run(self, prompt: Optional[str] = None) -> str:
        """
        Run the model with the given prompt.
        
        Args:
            prompt: The prompt to send to the model. If None, uses the configured user_prompt.
        
        Returns:
            The model's response as a string.
        """
        if not self._is_built:
            raise RuntimeError("Model must be built before running. Call .build() first.")
        
        if prompt is None:
            prompt = self._config.user_prompt
            if prompt is None:
                raise ValueError("No prompt provided. Either pass a prompt to run() or set user_prompt()")
        
        try:
            provider = self._providers[self._config.provider]
            response = provider.run(self._client, prompt, self._config)
            
            # Execute callbacks
            for callback in self._config.callbacks:
                try:
                    callback(response, prompt, self._config)
                except Exception as e:
                    logger.warning(f"Callback execution failed: {e}")
            
            return response
        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            raise
    
    def get_config(self) -> ModelConfig:
        """Get the current configuration."""
        return self._config
    
    def reset(self) -> 'ModelBuilder':
        """Reset the builder to initial state."""
        self._config = ModelConfig()
        self._client = None
        self._is_built = False
        return self


# Convenience functions for quick model creation
def create_openai_model(model_name: str = "gpt-4.1-nano", **kwargs) -> ModelBuilder:
    """Create an OpenAI model with default settings."""
    return ModelBuilder().with_provider("openai").with_model(model_name).build()


def create_anthropic_model(model_name: str = "claude-3-sonnet-20240229", **kwargs) -> ModelBuilder:
    """Create an Anthropic model with default settings."""
    return ModelBuilder().with_provider("anthropic").with_model(model_name).build()


def create_ollama_model(model_name: str = "llama2", **kwargs) -> ModelBuilder:
    """Create an Ollama model with default settings."""
    return ModelBuilder().with_provider("ollama").with_model(model_name).build()


# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage
    model = (ModelBuilder()
             .with_provider("openai")
             .with_model("gpt-4.1-nano")
             .with_temperature(0.7)
             .with_max_tokens(1000)
             .with_system_prompt("You are a helpful assistant.")
             .build())
    
    response = model.run("What is the capital of France?")
    print(response)
    
    # Example 2: Using convenience function
    model2 = create_openai_model("gpt-4.1-nano", temperature=0.5)
    response2 = model2.run("Explain quantum computing in simple terms.")
    print(response2)
