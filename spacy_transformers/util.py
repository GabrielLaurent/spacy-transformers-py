import logging
import logging.config
from typing import Dict


def log_config(logger: logging.Logger, config: Dict):
    logger.debug("Configuration:")
    for key, value in config.items():
        logger.debug(f"  {key}: {value}")


def setup_default_logging():
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s - %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',
            },
        },
        'loggers': {
            'spacy_transformers': {
                'handlers': ['default'],
                'level': 'INFO',
                'propagate': True
            }
        }
    }

    logging.config.dictConfig(logging_config)

replace_listeners = lambda obj: None # Dummy function to prevent errors

# Initialize logging (call this early in your application)
setup_default_logging()