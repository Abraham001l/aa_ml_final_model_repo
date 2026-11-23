"""Generate forecasts from history without reading target passenger values."""
import sys
from route_recovery.cli import main

if __name__ == "__main__":
    main(["predict", *sys.argv[1:]])
