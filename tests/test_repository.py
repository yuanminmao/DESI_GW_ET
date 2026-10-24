from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class RepositoryIntegrityTests(unittest.TestCase):
    def test_catalogue_sizes_and_header(self) -> None:
        for count in (100, 500, 1000, 3000):
            path = ROOT / "results" / "catalogs" / f"N{count:04d}" / "realization_0042.txt"
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "z dL_obs[Mpc] sigma_dL[Mpc]")
            self.assertEqual(len(lines) - 1, count)

    def test_json_products_are_valid(self) -> None:
        products = sorted((ROOT / "results" / "tables").glob("*.json"))
        self.assertGreaterEqual(len(products), 10)
        for path in products:
            with self.subTest(path=path.name):
                json.loads(path.read_text(encoding="utf-8"))

    def test_manuscript_figures_exist(self) -> None:
        source = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
        names = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source)
        self.assertEqual(len(names), 7)
        for name in names:
            with self.subTest(figure=name):
                self.assertTrue((ROOT / "paper" / "figures" / name).is_file())

    def test_no_personal_absolute_paths_in_text_files(self) -> None:
        suffixes = {".cff", ".csv", ".json", ".md", ".py", ".tex", ".txt", ".toml", ".yml"}
        excluded = {".git", ".cache", "build", "dist"}
        offenders: list[str] = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if any(part in excluded for part in path.relative_to(ROOT).parts):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            unix_home_marker = "/" + "Users/"
            windows_home_marker = "C:\\" + "Users\\"
            if unix_home_marker in text or windows_home_marker in text:
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_no_github_hostile_large_files(self) -> None:
        limit = 50 * 1024 * 1024
        offenders = [
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*")
            if path.is_file() and ".git" not in path.parts and path.stat().st_size > limit
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
