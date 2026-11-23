"""Select parameters on the three development folds, never final targets."""
import sys
from route_recovery.cli import main

if __name__ == "__main__":
    main(["tune", *sys.argv[1:]])
