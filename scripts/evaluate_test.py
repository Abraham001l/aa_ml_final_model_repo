"""Score a fixed configuration on a separate 24-month target window."""
import sys
from route_recovery.cli import main

if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
