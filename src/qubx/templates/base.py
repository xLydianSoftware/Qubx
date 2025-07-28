"""Base template management system for strategy generation."""

import os
import shutil
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, Template

from qubx import logger


class TemplateError(Exception):
    """Exception raised for template-related errors."""
    pass


class TemplateManager:
    """Manages strategy templates and generation."""
    
    def __init__(self):
        self.templates_dir = Path(__file__).parent
        self.built_in_templates = self._discover_built_in_templates()
    
    def _discover_built_in_templates(self) -> dict[str, Path]:
        """Discover all built-in templates."""
        templates = {}
        for template_dir in self.templates_dir.iterdir():
            if template_dir.is_dir() and not template_dir.name.startswith('_'):
                template_yml = template_dir / "template.yml"
                if template_yml.exists():
                    templates[template_dir.name] = template_dir
        return templates
    
    def list_templates(self) -> dict[str, dict]:
        """List all available built-in templates with their metadata."""
        templates_info = {}
        for name, template_path in self.built_in_templates.items():
            template_yml = template_path / "template.yml"
            if template_yml.exists():
                with open(template_yml) as f:
                    metadata = yaml.safe_load(f)
                templates_info[name] = metadata
            else:
                templates_info[name] = {"name": name, "description": "No description available"}
        return templates_info
    
    def generate_strategy(
        self,
        template_name: str | None = None,
        template_path: str | None = None,
        output_dir: str | Path = ".",
        **template_vars
    ) -> Path:
        """
        Generate a strategy from a template.
        
        Args:
            template_name: Name of built-in template to use
            template_path: Path to custom template directory
            output_dir: Directory to create strategy in
            **template_vars: Variables to substitute in templates
        
        Returns:
            Path to generated strategy directory
        """
        # Determine template directory
        if template_path:
            template_dir = Path(template_path)
            if not template_dir.exists():
                raise TemplateError(f"Template path does not exist: {template_path}")
        elif template_name:
            if template_name not in self.built_in_templates:
                available = list(self.built_in_templates.keys())
                raise TemplateError(f"Template '{template_name}' not found. Available: {available}")
            template_dir = self.built_in_templates[template_name]
        else:
            # Default to simple template
            template_name = "simple"
            if template_name not in self.built_in_templates:
                raise TemplateError("Default 'simple' template not found")
            template_dir = self.built_in_templates[template_name]
        
        # Load template metadata
        template_yml = template_dir / "template.yml"
        template_metadata = {}
        if template_yml.exists():
            with open(template_yml) as f:
                template_metadata = yaml.safe_load(f)
        
        # Set default template variables
        strategy_name = template_vars.get('name', 'my_strategy')
        default_vars = {
            'strategy_name': strategy_name,
            'strategy_class': self._to_class_name(strategy_name),
            'exchange': 'BINANCE.UM',
            'symbols': ['BTCUSDT'],
            'timeframe': '1h',
        }
        default_vars.update(template_vars)
        
        # Ensure symbols is a list
        if isinstance(default_vars['symbols'], str):
            default_vars['symbols'] = [s.strip() for s in default_vars['symbols'].split(',')]
        
        # Create output directory
        output_path = Path(output_dir).resolve() / strategy_name
        if output_path.exists():
            raise TemplateError(f"Directory already exists: {output_path}")
        
        logger.info(f"Creating strategy '{strategy_name}' from template '{template_dir.name}'")
        
        # Generate files
        self._render_template_directory(template_dir, output_path, default_vars)
        
        logger.info(f"Strategy created successfully at: {output_path}")
        return output_path
    
    def _render_template_directory(self, template_dir: Path, output_dir: Path, template_vars: dict):
        """Recursively render template directory to output directory."""
        # Create Jinja2 environment
        env = Environment(loader=FileSystemLoader(template_dir))
        
        # Walk through template directory
        for template_file in template_dir.rglob("*"):
            if template_file.is_file() and template_file.name != "template.yml":
                # Calculate relative path from template directory
                rel_path = template_file.relative_to(template_dir)
                
                # Create output file path with template variable substitution in path
                rel_path_str = str(rel_path)
                for var_name, var_value in template_vars.items():
                    rel_path_str = rel_path_str.replace(f"{{{{ {var_name} }}}}", str(var_value))
                
                output_file = output_dir / rel_path_str
                
                # Handle .j2 extension
                if output_file.suffix == '.j2':
                    output_file = output_file.with_suffix('')
                
                # Create parent directories
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Render template
                if template_file.suffix == '.j2':
                    # Render Jinja2 template
                    template = env.get_template(str(rel_path))
                    rendered_content = template.render(**template_vars)
                    with open(output_file, 'w') as f:
                        f.write(rendered_content)
                    
                    # Make shell scripts executable
                    if output_file.suffix == '.sh':
                        output_file.chmod(0o755)
                else:
                    # Copy non-template files as-is
                    shutil.copy2(template_file, output_file)
                
                logger.debug(f"Created: {output_file}")
    
    def _to_class_name(self, name: str) -> str:
        """Convert strategy name to proper class name."""
        # Convert snake_case or kebab-case to PascalCase
        words = name.replace('-', '_').split('_')
        return ''.join(word.capitalize() for word in words) + 'Strategy'