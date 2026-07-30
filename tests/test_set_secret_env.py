import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "set_secret_env.py"


class SetSecretEnvTests(unittest.TestCase):
    def test_sets_and_replaces_without_duplication(self):
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text("EXISTING=ok\nLGI_READONLY_DATABASE_URL=old\n")
            subprocess.run(
                [sys.executable, str(SCRIPT), str(env_path), "LGI_READONLY_DATABASE_URL"],
                input="postgresql://reader:new@example/neondb\n",
                text=True,
                check=True,
            )
            contents = env_path.read_text()
            self.assertIn("EXISTING=ok", contents)
            self.assertEqual(contents.count("LGI_READONLY_DATABASE_URL="), 1)
            self.assertIn(
                "LGI_READONLY_DATABASE_URL=postgresql://reader:new@example/neondb",
                contents,
            )
            mode = stat.S_IMODE(os.stat(env_path).st_mode)
            self.assertEqual(mode, 0o600)


if __name__ == "__main__":
    unittest.main()
