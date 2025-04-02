import logging
import functools
from typing import Callable, Any, TypeVar, ParamSpec, Optional
from enum import Enum

# Set up logging
logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Enum for different logging levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    NONE = None  # For functions that should not be logged

P = ParamSpec('P')
T = TypeVar('T')

def log_and_handle_errors(
    log_level: LogLevel = LogLevel.INFO,
    skip_logging: bool = False
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that handles logging and error handling for functions."""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            func_name = func.__name__
            if not skip_logging and log_level != LogLevel.NONE:
                logger.log(log_level.value, f"Starting {func_name}")
            try:
                result = func(*args, **kwargs)
                if not skip_logging and log_level != LogLevel.NONE:
                    logger.log(log_level.value, f"Completed {func_name} successfully")
                return result
            except Exception as e:
                logger.error(f"Error in {func_name}: {str(e)}")
                raise
        return wrapper
    return decorator

def log_class_methods(
    default_log_level: LogLevel = LogLevel.INFO,
    skip_methods: Optional[set[str]] = None
) -> Callable[[type], type]:
    """Decorator that applies log_and_handle_errors to all methods of a class."""
    if skip_methods is None:
        skip_methods = {
            # PyTorch frequently called methods
            'forward', 'backward', '__call__', 'step', 'zero_grad',
            # Initialization and utility methods
            '_init_weights', 'apply', 'to', 'train', 'eval',
            # Data processing methods
            'cpu', 'numpy', 'detach', 'clone', 'copy_',
            # Model specific methods
            'log_prior'
        }
    
    def decorator(cls: type) -> type:
        for name, method in cls.__dict__.items():
            if callable(method) and not name.startswith('__'):
                # Skip logging for frequently called methods
                skip_logging = name in skip_methods
                setattr(cls, name, log_and_handle_errors(
                    log_level=LogLevel.DEBUG if skip_logging else default_log_level,
                    skip_logging=skip_logging
                )(method))
        return cls
    return decorator 