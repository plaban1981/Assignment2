"""
Prompt Manager - Task 1: Prompt Versioning System

Handles loading, versioning, and rollback of prompts from YAML files.
Supports 30-second rollback capability via version switching.
"""

import yaml
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class PromptManager:
    """
    Manages versioned prompts stored in YAML files.

    Features:
    - Load prompts by version or use current active version
    - Quick rollback by changing current.yaml
    - Caching for performance
    - Metadata tracking for audit
    """

    def __init__(self, prompts_dir: str = None):
        """
        Initialize the PromptManager.

        Args:
            prompts_dir: Path to prompts directory. Defaults to ./prompts
        """
        if prompts_dir is None:
            # Default to prompts directory relative to this file
            prompts_dir = Path(__file__).parent / "prompts"
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_timestamp: Dict[str, datetime] = {}

    def _get_prompt_path(self, agent_name: str, version: Optional[str] = None) -> Path:
        """
        Get the path to a prompt file.

        Args:
            agent_name: Name of the agent (e.g., 'customer_support')
            version: Specific version (e.g., 'v1.0.0') or None for current

        Returns:
            Path to the YAML file
        """
        agent_dir = self.prompts_dir / "agents" / agent_name

        if version:
            return agent_dir / f"{version}.yaml"

        # Read current version from current.yaml
        current_file = agent_dir / "current.yaml"
        if current_file.exists():
            with open(current_file, 'r') as f:
                current_config = yaml.safe_load(f)
                version = current_config.get('current_version', 'v1.0.0')
                return agent_dir / f"{version}.yaml"

        # Fallback to v1.0.0
        return agent_dir / "v1.0.0.yaml"

    def load_prompt(self, agent_name: str, version: Optional[str] = None,
                    use_cache: bool = True) -> Dict[str, Any]:
        """
        Load a prompt configuration from YAML.

        Args:
            agent_name: Name of the agent (e.g., 'customer_support')
            version: Specific version or None for current
            use_cache: Whether to use cached version

        Returns:
            Dictionary containing prompt configuration
        """
        cache_key = f"{agent_name}:{version or 'current'}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        prompt_path = self._get_prompt_path(agent_name, version)

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_data = yaml.safe_load(f)

        # Cache the loaded prompt
        self._cache[cache_key] = prompt_data
        self._load_timestamp[cache_key] = datetime.now()

        return prompt_data

    def get_system_prompt(self, agent_name: str, version: Optional[str] = None,
                          user_tier: str = "standard") -> str:
        """
        Get the complete system prompt for an agent.

        Args:
            agent_name: Name of the agent
            version: Specific version or None for current
            user_tier: 'vip' or 'standard' to include tier-specific instructions

        Returns:
            Complete system prompt string
        """
        prompt_data = self.load_prompt(agent_name, version)

        # Build the complete prompt
        parts = [prompt_data.get('system_prompt', '')]

        # Add tier-specific instructions
        if user_tier == 'vip':
            parts.append(prompt_data.get('vip_instructions', ''))
        else:
            parts.append(prompt_data.get('standard_instructions', ''))

        # Add safety instructions
        parts.append(prompt_data.get('safety_instructions', ''))

        return '\n\n'.join(filter(None, parts))

    def get_current_version(self, agent_name: str) -> str:
        """
        Get the current active version for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Version string (e.g., 'v1.1.0')
        """
        current_file = self.prompts_dir / "agents" / agent_name / "current.yaml"

        if current_file.exists():
            with open(current_file, 'r') as f:
                current_config = yaml.safe_load(f)
                return current_config.get('current_version', 'v1.0.0')

        return 'v1.0.0'

    def set_current_version(self, agent_name: str, version: str) -> None:
        """
        Set the current active version (enables 30-second rollback).

        Args:
            agent_name: Name of the agent
            version: Version to set as current
        """
        # Verify the version exists
        version_file = self.prompts_dir / "agents" / agent_name / f"{version}.yaml"
        if not version_file.exists():
            raise FileNotFoundError(f"Version {version} does not exist for {agent_name}")

        current_file = self.prompts_dir / "agents" / agent_name / "current.yaml"

        with open(current_file, 'w') as f:
            yaml.dump({'current_version': version}, f)

        # Clear cache for this agent
        keys_to_remove = [k for k in self._cache if k.startswith(f"{agent_name}:")]
        for key in keys_to_remove:
            del self._cache[key]

    def rollback(self, agent_name: str, version: str) -> None:
        """
        Quick rollback to a previous version.

        Args:
            agent_name: Name of the agent
            version: Version to rollback to
        """
        self.set_current_version(agent_name, version)

    def list_versions(self, agent_name: str) -> list:
        """
        List all available versions for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            List of version strings
        """
        agent_dir = self.prompts_dir / "agents" / agent_name

        if not agent_dir.exists():
            return []

        versions = []
        for file in agent_dir.glob("v*.yaml"):
            versions.append(file.stem)

        return sorted(versions)

    def get_metadata(self, agent_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata for a prompt version.

        Args:
            agent_name: Name of the agent
            version: Specific version or None for current

        Returns:
            Metadata dictionary
        """
        prompt_data = self.load_prompt(agent_name, version)
        return prompt_data.get('metadata', {})

    def load_shared_prompt(self, prompt_name: str) -> Dict[str, Any]:
        """
        Load a shared prompt component.

        Args:
            prompt_name: Name of the shared prompt (without .yaml)

        Returns:
            Prompt data dictionary
        """
        prompt_path = self.prompts_dir / "shared" / f"{prompt_name}.yaml"

        if not prompt_path.exists():
            raise FileNotFoundError(f"Shared prompt not found: {prompt_path}")

        with open(prompt_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)


# Example usage and testing
if __name__ == "__main__":
    pm = PromptManager()

    # List available versions
    print("Available versions:", pm.list_versions("customer_support"))

    # Get current version
    print("Current version:", pm.get_current_version("customer_support"))

    # Load and display system prompt
    print("\n--- System Prompt (VIP) ---")
    print(pm.get_system_prompt("customer_support", user_tier="vip"))

    # Get metadata
    print("\n--- Metadata ---")
    print(pm.get_metadata("customer_support"))
