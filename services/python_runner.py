"""
Python Runner.

Executes Python files and captures the results.
"""

import subprocess
import sys
from pathlib import Path


class PythonRunnerService:

    TIMEOUT = 30

    def run(self, path: str):

        try:

            file = Path(path)

            if not file.exists():

                return {
                    "success": False,
                    "error": "File does not exist."
                }

            if not file.is_file():

                return {
                    "success": False,
                    "error": "Path is not a file."
                }

            if file.suffix != ".py":

                return {
                    "success": False,
                    "error": "Not a Python file."
                }

            result = subprocess.run(
                [
                    sys.executable,
                    str(file)
                ],
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT
            )

            return {
                "success": True,
                "data": {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                }
            }

        except subprocess.TimeoutExpired:

            return {
                "success": False,
                "error": "Execution timed out."
            }

        except Exception as e:

            return {
                "success": False,
                "error": str(e)
            }