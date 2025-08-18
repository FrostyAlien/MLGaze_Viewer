"""Plugin manager for orchestrating sequential plugin execution with dependency management."""

import time
from typing import Dict, List, Any
from src.utils.logger import MLGazeLogger

from .registry import PluginRegistry
from .base import AnalyticsPlugin
from .exceptions import PluginExecutionError, MissingDependencyError


class PluginManager:
    """Manages plugin execution with automatic dependency resolution."""
    
    def __init__(self, logger=None):
        """Initialize plugin manager.
        
        Args:
            logger: Optional logger instance. If None, creates MLGazeLogger.
        """
        self.registry = PluginRegistry()
        self.logger = logger or MLGazeLogger().get_logger("PluginManager")
        self.execution_results = {}
    
    def register_plugin(self, plugin: AnalyticsPlugin) -> None:
        """Register a plugin with the manager.
        
        Args:
            plugin: AnalyticsPlugin instance to register
        """
        try:
            self.registry.register(plugin)
            plugin_name = plugin.__class__.__name__
            self.logger.info(f"Registered plugin: {plugin_name}")
            
            # Log dependencies
            deps = plugin.get_dependencies()
            if deps:
                self.logger.info(f"  Dependencies: {deps}")
            
            optional_deps = plugin.get_optional_dependencies()
            if optional_deps:
                self.logger.info(f"  Optional dependencies: {optional_deps}")
        except Exception as e:
            self.logger.error(f"Failed to register plugin {plugin.__class__.__name__}: {e}")
            raise
    
    def register_plugins(self, plugins: List[AnalyticsPlugin]) -> None:
        """Register multiple plugins.
        
        Args:
            plugins: List of AnalyticsPlugin instances to register
        """
        for plugin in plugins:
            self.register_plugin(plugin)
    
    def execute_plugins(self, session, config: Dict[str, Any], 
                       requested_plugins: List[str] = None) -> Dict[str, Any]:
        """Execute plugins in dependency order.
        
        Args:
            session: SessionData object containing all sensor data
            config: Configuration dictionary
            requested_plugins: Optional list of plugin names to execute.
                              If None, executes all enabled registered plugins.
        
        Returns:
            Dictionary mapping plugin names to their results
        """
        self.execution_results = {}
        
        try:
            # Determine which plugins to execute
            if requested_plugins is None:
                requested_plugins = [
                    name for name, plugin in self.registry.plugins.items() 
                    if plugin.enabled
                ]
            
            # Get execution order
            execution_order = self.registry.topological_sort(requested_plugins)
            self.logger.info(f"Executing {len(execution_order)} plugins in dependency order: {execution_order}")
            
            # Execute plugins sequentially
            for plugin_name in execution_order:
                self._execute_plugin(plugin_name, session, config)
            
            self.logger.info(f"Plugin execution completed. {len(self.execution_results)} plugins processed")
            return self.execution_results
            
        except Exception as e:
            self.logger.error(f"Plugin execution failed: {e}")
            self.execution_results["_system_error"] = str(e)
            return self.execution_results
    
    def _execute_plugin(self, plugin_name: str, session, config: Dict[str, Any]) -> None:
        """Execute a single plugin.
        
        Args:
            plugin_name: Name of plugin to execute
            session: SessionData object
            config: Configuration dictionary
        """
        try:
            plugin = self.registry.get_plugin(plugin_name)
            
            # Check if plugin is enabled
            if not plugin.enabled:
                self.logger.info(f"Skipping disabled plugin: {plugin_name}")
                self.execution_results[plugin_name] = {"status": "skipped", "reason": "disabled"}
                return
            
            # Validate data requirements
            if not plugin.validate_data(session):
                self.logger.warning(f"Plugin {plugin_name} validation failed - invalid data")
                self.execution_results[plugin_name] = {"error": "Data validation failed"}
                return
            
            # Check dependencies
            if not plugin.validate_dependencies(self.execution_results):
                missing_deps = [dep for dep in plugin.get_dependencies() 
                               if dep not in self.execution_results or 
                               "error" in self.execution_results.get(dep, {})]
                self.logger.error(f"Plugin {plugin_name} missing dependencies: {missing_deps}")
                self.execution_results[plugin_name] = {
                    "error": f"Missing dependencies: {missing_deps}"
                }
                return
            
            # Prepare config with dependency results
            enhanced_config = dict(config)
            
            # Add required dependency results
            if plugin.get_dependencies():
                enhanced_config["dependencies"] = {}
                for dep_name in plugin.get_dependencies():
                    enhanced_config["dependencies"][dep_name] = self.execution_results[dep_name]
                self.logger.debug(f"  Injected required dependencies: {plugin.get_dependencies()}")
            
            # Add optional dependency results if available
            optional_deps = plugin.get_optional_dependencies()
            available_optional = []
            missing_optional = []
            if optional_deps:
                if "dependencies" not in enhanced_config:
                    enhanced_config["dependencies"] = {}
                for dep_name in optional_deps:
                    if (dep_name in self.execution_results and 
                        "error" not in self.execution_results[dep_name]):
                        enhanced_config["dependencies"][dep_name] = self.execution_results[dep_name]
                        available_optional.append(dep_name)
                    else:
                        missing_optional.append(dep_name)
                
                if available_optional:
                    self.logger.debug(f"  Injected optional dependencies: {available_optional}")
                if missing_optional:
                    self.logger.debug(f"  Skipping missing optional dependencies: {missing_optional}")
            
            # Execute plugin with timing
            self.logger.info(f"Executing plugin: {plugin_name}")
            start_time = time.time()
            plugin_results = plugin.process(session, enhanced_config)
            execution_time = time.time() - start_time
            
            # Store results
            self.execution_results[plugin_name] = plugin_results
            
            # Store in session for backward compatibility and data sharing
            if hasattr(session, 'set_plugin_result'):
                session.set_plugin_result(plugin_name, plugin_results)
            
            # Log execution results
            self.logger.info(f"✓ {plugin_name} completed in {execution_time:.3f}s")
            
            # Log summary
            summary = plugin.get_summary(plugin_results)
            if summary:
                self.logger.info(f"  {summary}")
            else:
                # Basic results summary if plugin doesn't provide one
                if isinstance(plugin_results, dict):
                    result_keys = [k for k in plugin_results.keys() if k != 'session_data']
                    self.logger.debug(f"  Results keys: {result_keys}")
                
        except Exception as e:
            self.logger.error(f"Plugin {plugin_name} execution failed: {e}", exc_info=True)
            self.execution_results[plugin_name] = {
                "error": str(e),
                "plugin_class": plugin.__class__.__name__
            }
    
    def get_execution_plan(self, requested_plugins: List[str] = None) -> Dict[str, Any]:
        """Get execution plan for requested plugins.
        
        Args:
            requested_plugins: List of plugin names to plan for
            
        Returns:
            Dictionary with execution plan details
        """
        if requested_plugins is None:
            requested_plugins = [
                name for name, plugin in self.registry.plugins.items() 
                if plugin.enabled
            ]
        
        return self.registry.get_execution_plan(requested_plugins)
    
    def list_registered_plugins(self) -> List[str]:
        """Get list of all registered plugin names.
        
        Returns:
            List of plugin names
        """
        return self.registry.list_plugins()
    
    def get_plugin_info(self, plugin_name: str) -> Dict[str, Any]:
        """Get information about a registered plugin.
        
        Args:
            plugin_name: Name of plugin to get info for
            
        Returns:
            Dictionary with plugin information
        """
        try:
            plugin = self.registry.get_plugin(plugin_name)
            return {
                "name": plugin.name,
                "class_name": plugin.__class__.__name__,
                "enabled": plugin.enabled,
                "entity_path": plugin.entity_path,
                "dependencies": plugin.get_dependencies(),
                "optional_dependencies": plugin.get_optional_dependencies(),
                "required_columns": plugin.get_required_columns()
            }
        except KeyError:
            return {"error": f"Plugin '{plugin_name}' not found"}
    
    def clear_results(self) -> None:
        """Clear execution results cache."""
        self.execution_results = {}