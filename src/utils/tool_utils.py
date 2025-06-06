def transform_tool_definition(
    name: str, description: str, input_schema: dict, annotations: dict = None
) -> dict:
    """
    Transform a tool definition into the required format for the OpenAI API.

    Args:
        name (str): The name of the tool
        description (str): The description of the tool
        input_schema (dict): The input schema for the tool
        annotations (dict, optional): Additional annotations for the tool

    Returns:
        dict: The transformed tool definition in the required format
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": input_schema,
        },
    }


def transform_schema_to_parameters(schema: dict) -> dict:
    """
    Transform a schema format into the parameters format.

    Args:
        schema (dict): The input schema in the format:
            {
                'properties': {
                    'field_name': {
                        'title': 'Field Title',
                        'type': 'field_type',
                        'default': default_value  # optional
                    }
                },
                'required': ['field_name'],
                'type': 'object'
            }

    Returns:
        dict: The transformed parameters format
    """
    # Create a new properties dict with descriptions instead of titles
    properties = {}
    for field_name, field_info in schema["properties"].items():
        properties[field_name] = {
            "type": field_info["type"],
            "description": field_info["title"],
        }
        # Add default value if it exists
        if "default" in field_info:
            properties[field_name]["default"] = field_info["default"]

    return {"type": "object", "properties": properties, "required": schema["required"]}


def create_tools_list(*tool_definitions) -> list:
    """
    Create a list of tools from multiple tool definitions.

    Args:
        *tool_definitions: Variable number of tool definitions

    Returns:
        list: List of transformed tool definitions
    """
    return [transform_tool_definition(**tool_def) for tool_def in tool_definitions]


# Example usage:
if __name__ == "__main__":
    # Example tool definition
    tool_def = {
        "name": "process_video",
        "description": "Process a video file and prepare it for searching.",
        "input_schema": {
            "type": "object",
            "properties": {"video_path": {"type": "string", "title": "Video Path"}},
            "required": ["video_path"],
        },
        "annotations": None,
    }

    # Transform single tool
    transformed_tool = transform_tool_definition(**tool_def)
    print("Single tool:", transformed_tool)

    # Create tools list
    tools = create_tools_list(tool_def)
    print("\nTools list:", tools)

    # Example schema transformation
    schema = {
        "properties": {
            "video_name": {"title": "Video Name", "type": "string"},
            "user_query": {"title": "User Query", "type": "string"},
            "top_k": {"default": 3, "title": "Top K", "type": "integer"},
        },
        "required": ["video_name", "user_query"],
        "type": "object",
    }

    parameters = transform_schema_to_parameters(schema)
    print("\nTransformed parameters:", parameters)
