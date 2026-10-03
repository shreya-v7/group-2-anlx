"""
LLMBox -- A Software Application for Building Customized and Affordable AI Solutions.
Copyright (C) 2026  Sara Kingsley

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from pathlib import Path

from pydantic import BaseModel
from src.pydantic_models import personrecords


def save_model_json(
    model: BaseModel,
    directory: str | Path,
    filename: str = "config.json",
) -> Path:
    """Save a Pydantic model as JSON, creating the directory if needed."""
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    # Keep the output inside the specified directory.
    if not filename or Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError("filename must be a file name, not a path")

    output_path = directory / filename
    output_path.write_text(
        model.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


"""
# Example model
class ModelConfig(BaseModel):
    model: str
    mode: str = "chat"
    temperature: float = 0.7

"""

if __name__ == "__main__":
    # config = ModelConfig(model="gemma3_270m")
    # config = src.personrecords.PersonRecord()

    output = save_model_json(
        # config,
        model=personrecords.PersonRecord(firstname="sara", lastname="king"),
        directory="./data/pydantic_models/",
        filename="structuredoutput.json",
    )
    print(f"Saved to {output.resolve()}")
