"""
    Register feature definitions with Feast.

    Run once after setting up the feature repo:
        python scripts/feast_apply.py

    This runs `feast apply` which:
    1. Reads feature_repo/features.py, entities.py, data_sources.py
    2. Registers all entities, feature views, feature services
    3. Creates the local registry (feature_repo/data/registry.db)
    4. Does NOT push data to Redis — that is done by feast_materialize.py

"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    
    feature_repo = Path("feature_repo")
    
    if not feature_repo.exists():
        logger.error("feature_repo/ directory not found. Run this script from the project root.")
        sys.exit(1)
        
    logger.info("Running feast apply to register feature definitions...")
    
    result = subprocess.run(
        ["feast", "apply"],
        cwd=str(feature_repo),
        capture_output=False,  # show output directly
        text=True,
    )
    
    if result.returncode != 0:
        logger.error("feast apply failed.")
        sys.exit(1)

    logger.info("feast apply complete. Feature definitions registered.")

if __name__ == "__main__":
    main()
