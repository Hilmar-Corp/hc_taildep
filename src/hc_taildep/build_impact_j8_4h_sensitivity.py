from __future__ import annotations

import sys

from hc_taildep.build_impact_j7_var_es import main as j7_main


def main() -> int:
    # Forward args to J7 (J7 parses sys.argv itself)
    sys.argv[0] = "hc_taildep.build_impact_j8_4h_sensitivity"
    return j7_main()


if __name__ == "__main__":
    raise SystemExit(main())