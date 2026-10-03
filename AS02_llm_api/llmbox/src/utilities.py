from pathlib import Path
import os
import sys

class Utilities:

    def __init__(self) -> None:
        pass


    def use_sibling_directory(
        self,
        directory_name: str,
        *,
        change_working_directory: bool = False,
    ) -> Path:
        """
        Enable imports from a sibling directory's `src` package.

        Paths are resolved relative to this Python file, not the terminal's
        current working directory.

        Optionally change the working directory so relative file reads also
        resolve from the sibling directory.
        """
        script_directory = Path(__file__).resolve().parent
        target_directory = (
            script_directory.parent / directory_name
        ).resolve()

        if not target_directory.is_dir():
            raise FileNotFoundError(
                f"Directory does not exist: {target_directory}"
            )

        if not (target_directory / "src").is_dir():
            raise FileNotFoundError(
                f"No src directory found inside: {target_directory}"
            )

        # Add the directory CONTAINING src so `from src...` works.
        import_path = str(target_directory)
        if import_path not in sys.path:
            sys.path.insert(0, import_path)

        if change_working_directory:
            os.chdir(target_directory)

        return target_directory
