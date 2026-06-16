import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline.log", mode="a")
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging infrastructure successfully initialized.")

# Call setup_logging() at the very beginning of execution.