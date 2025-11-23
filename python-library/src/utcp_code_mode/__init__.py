"""UTCP Code Mode Client Extension.

This package provides a CodeModeUtcpClient that extends the base UtcpClient
with Python code execution capabilities. It allows executing Python code
that can directly call registered tools as functions.

Key Features:
    - Python code execution with tool access
    - Automatic Python type hint generation from JSON schemas
    - Console output capture
    - Tool introspection capabilities
    - Safe execution environment with timeout support

Usage:
    ```python
    from utcp_code_mode import CodeModeUtcpClient
    
    # Create a code mode client
    client = await CodeModeUtcpClient.create()
    
    # Execute Python code with tool access
    result = await client.call_tool_chain('''
    # Your Python code here
    tools = await search_tools("weather")
    result = await weather.get_current_weather(city="London")
    print(f"Weather in London: {result}")
    ''')
    
    print("Result:", result["result"])
    print("Logs:", result["logs"])
    ```
"""

from utcp_code_mode.code_mode_utcp_client import CodeModeUtcpClient

# Since this is a client extension rather than a communication protocol,
# we don't need to register with the plugin system in the same way.
# The CodeModeUtcpClient can be used directly by importing it.

__all__ = [
    "CodeModeUtcpClient",
]

__version__ = "1.0.0"
