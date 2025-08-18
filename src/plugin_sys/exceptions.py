"""Custom exceptions for the plugin system."""


class PluginSystemError(Exception):
    """Base exception for plugin system errors."""
    pass


class CircularDependencyError(PluginSystemError):
    """Raised when circular dependencies are detected between plugins."""
    
    def __init__(self, cycle_path):
        self.cycle_path = cycle_path
        super().__init__(f"Circular dependency detected: {' -> '.join(cycle_path)}")


class MissingDependencyError(PluginSystemError):
    """Raised when a required plugin dependency is missing."""
    
    def __init__(self, plugin_name, missing_dependency):
        self.plugin_name = plugin_name
        self.missing_dependency = missing_dependency
        super().__init__(f"Plugin '{plugin_name}' requires '{missing_dependency}' but it's not available")


class PluginExecutionError(PluginSystemError):
    """Raised when a plugin fails during execution."""
    
    def __init__(self, plugin_name, original_error):
        self.plugin_name = plugin_name
        self.original_error = original_error
        super().__init__(f"Plugin '{plugin_name}' failed: {original_error}")