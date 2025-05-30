def log_function_call(func):
    """Decorator to log function calls only when self.verbose is True (if applicable)."""
    def wrapper(*args, **kwargs):
        should_log = False

        # Check if method is bound to a class with self.verbose = True
        if args and hasattr(args[0], 'verbose'):
            should_log = getattr(args[0], 'verbose', False)

        if should_log:
            print(f"\nFUNCTION CALL: {func.__module__}.{func.__name__}")
        return func(*args, **kwargs)
    return wrapper
