"""Module-execution bridge for ``python -m classical_conditioning``.

Review note: this module deliberately contains no workflow logic.  It makes
``python -m classical_conditioning`` delegate to the same command-line entry
point as the installed console command, keeping both invocation styles aligned.
"""

# Reuse the installed-command implementation so both invocation styles behave
# identically and need only one argument-parsing entry point.
from classical_conditioning.cli import main


# Invoke only when Python runs this module directly, never when it is imported.
if __name__ == "__main__":
    main()
