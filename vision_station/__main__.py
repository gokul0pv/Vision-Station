r"""python -m vision_station  -- start the application.

    python -m vision_station                   the application
    python -m vision_station D:\projects       ... with its projects elsewhere
    python -m vision_station --classic         the classic single-window editor
"""
import sys


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    if "--classic" in argv[1:]:
        from vision_station.ui.classic_window import main as classic
        return classic()
    from vision_station.ui.main_window import main as app
    return app(argv)


if __name__ == "__main__":
    sys.exit(main())
