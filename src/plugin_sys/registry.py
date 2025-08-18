"""Plugin registry with DAG-based dependency management."""

from typing import Dict, List, Set, Tuple
from collections import defaultdict, deque

from .base import AnalyticsPlugin
from .exceptions import CircularDependencyError, MissingDependencyError


class PluginRegistry:
    """Registry for managing plugins and their dependencies using DAG pattern."""
    
    def __init__(self):
        """Initialize empty registry."""
        self.plugins: Dict[str, AnalyticsPlugin] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.optional_dependencies: Dict[str, List[str]] = {}
    
    def register(self, plugin: AnalyticsPlugin) -> None:
        """Register a plugin and build dependency graph.
        
        Args:
            plugin: AnalyticsPlugin instance to register
            
        Raises:
            CircularDependencyError: If registering creates a cycle
        """
        plugin_name = plugin.__class__.__name__
        
        # Store plugin and its dependencies
        self.plugins[plugin_name] = plugin
        self.dependencies[plugin_name] = plugin.get_dependencies()
        self.optional_dependencies[plugin_name] = plugin.get_optional_dependencies()
        
        # Check for circular dependencies after registration
        cycles = self.detect_circular_dependencies()
        if cycles:
            # Remove the plugin we just added to restore consistency
            del self.plugins[plugin_name]
            del self.dependencies[plugin_name]
            del self.optional_dependencies[plugin_name]
            raise CircularDependencyError(cycles[0])
    
    def unregister(self, plugin_name: str) -> None:
        """Remove a plugin from the registry.
        
        Args:
            plugin_name: Name of plugin to remove
        """
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
            del self.dependencies[plugin_name]
            del self.optional_dependencies[plugin_name]
    
    def get_plugin(self, plugin_name: str) -> AnalyticsPlugin:
        """Get a registered plugin by name.
        
        Args:
            plugin_name: Name of plugin to retrieve
            
        Returns:
            AnalyticsPlugin instance
            
        Raises:
            KeyError: If plugin is not registered
        """
        if plugin_name not in self.plugins:
            raise KeyError(f"Plugin '{plugin_name}' not found in registry")
        return self.plugins[plugin_name]
    
    def list_plugins(self) -> List[str]:
        """Get list of all registered plugin names.
        
        Returns:
            List of plugin names
        """
        return list(self.plugins.keys())
    
    def topological_sort(self, requested_plugins: List[str] = None) -> List[str]:
        """Get plugins in execution order using Kahn's algorithm.
        
        Args:
            requested_plugins: Optional list of plugin names to include.
                              If None, includes all registered plugins.
        
        Returns:
            List of plugin names in dependency order
            
        Raises:
            MissingDependencyError: If a required dependency is not available
            CircularDependencyError: If circular dependencies are detected
        """
        if requested_plugins is None:
            requested_plugins = list(self.plugins.keys())
        
        # Build dependency graph including all required dependencies
        plugins_to_include = set()
        self._add_dependencies_recursive(requested_plugins, plugins_to_include)
        
        # Build graph for topological sort
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        # Initialize in_degree for all plugins
        for plugin in plugins_to_include:
            in_degree[plugin] = 0
        
        # Build graph and calculate in-degrees
        for plugin in plugins_to_include:
            deps = self.dependencies.get(plugin, [])
            for dep in deps:
                if dep not in plugins_to_include:
                    raise MissingDependencyError(plugin, dep)
                graph[dep].append(plugin)
                in_degree[plugin] += 1
        
        # Kahn's algorithm
        queue = deque([plugin for plugin in plugins_to_include if in_degree[plugin] == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            # Reduce in-degree for dependent plugins
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(result) != len(plugins_to_include):
            remaining = plugins_to_include - set(result)
            raise CircularDependencyError(list(remaining))
        
        return result
    
    def _add_dependencies_recursive(self, plugins: List[str], result_set: Set[str]) -> None:
        """Recursively add plugins and their dependencies to result set.
        
        Args:
            plugins: List of plugin names to process
            result_set: Set to add plugins to (modified in-place)
        """
        for plugin in plugins:
            if plugin not in result_set:
                if plugin not in self.plugins:
                    raise MissingDependencyError("requested", plugin)
                
                result_set.add(plugin)
                deps = self.dependencies.get(plugin, [])
                if deps:
                    self._add_dependencies_recursive(deps, result_set)
    
    def detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in the current graph.
        
        Returns:
            List of cycles, where each cycle is a list of plugin names
        """
        # Use DFS to detect cycles
        white = set(self.plugins.keys())  # Unvisited
        gray = set()   # Currently visiting
        black = set()  # Fully processed
        cycles = []
        
        def dfs(node: str, path: List[str]) -> None:
            if node in gray:
                # Found a cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in black:
                return
            
            white.discard(node)
            gray.add(node)
            path.append(node)
            
            # Visit dependencies
            for dep in self.dependencies.get(node, []):
                if dep in self.plugins:  # Only check registered plugins
                    dfs(dep, path)
            
            path.pop()
            gray.discard(node)
            black.add(node)
        
        # Start DFS from each unvisited node
        for plugin in list(white):
            if plugin in white:
                dfs(plugin, [])
        
        return cycles
    
    def get_execution_plan(self, requested_plugins: List[str]) -> Dict[str, any]:
        """Get detailed execution plan for requested plugins.
        
        Args:
            requested_plugins: List of plugin names to execute
            
        Returns:
            Dictionary with execution order and dependency information
        """
        try:
            execution_order = self.topological_sort(requested_plugins)
            
            return {
                "execution_order": execution_order,
                "total_plugins": len(execution_order),
                "dependency_graph": dict(self.dependencies),
                "optional_dependencies": dict(self.optional_dependencies),
                "status": "valid"
            }
        except (CircularDependencyError, MissingDependencyError) as e:
            return {
                "execution_order": [],
                "total_plugins": 0,
                "dependency_graph": dict(self.dependencies),
                "optional_dependencies": dict(self.optional_dependencies),
                "status": "error",
                "error": str(e)
            }