# Handle both enum objects and string values safely
def get_enum_value(field_value):
    """Safely extract value from enum or string."""
    if hasattr(field_value, "value"):
        return field_value.value
    return str(field_value)
