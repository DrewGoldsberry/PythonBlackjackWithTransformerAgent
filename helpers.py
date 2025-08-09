def is_lambda(v):
    """
    Checks if a variable 'v' is a lambda function.
    """
    return callable(v) and getattr(v, '__name__', None) == '<lambda>'