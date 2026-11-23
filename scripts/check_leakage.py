"""Run 36 adversarial checks plus four positive controls on the supplied panel."""
import sys
from route_recovery.cli import main

if __name__ == "__main__":
    main(["verify", *sys.argv[1:]])
