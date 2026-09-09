"""Reserved entrypoint for future dual-object regression testing."""

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    parser.exit(
        status=2,
        message=(
            "Multi-object regression is not implemented: a real Pipe2 CAD and "
            "validated dual-object test procedure are still required.\n"
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
