"""Current application entrypoint.

During this migration it deliberately delegates to the verified single-object
orchestration. Dual-object execution is not claimed until a real second CAD and
regression procedure exist.
"""

from scripts.test_single_object import main


if __name__ == "__main__":
    raise SystemExit(main())
