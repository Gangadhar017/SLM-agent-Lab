"""Inject the tables from results/report/summary.md into README.md between the RESULTS markers, so the README
never drifts from the logged results.

  python scripts/make_report.py && python scripts/update_readme_results.py
"""

import re
from pathlib import Path

import _bootstrap  # noqa: F401

from slm_agent_lab.paths import RESULTS_DIR, ROOT


def main() -> None:
    summary = (RESULTS_DIR / "report" / "summary.md").read_text(encoding="utf-8")
    body = summary.split("\n", 1)[1].strip()  # drop the "# Evaluation summary" title
    body = body.replace("\n## ", "\n#### ")
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    new = re.sub(r"<!-- RESULTS:BEGIN -->.*?<!-- RESULTS:END -->",
                 "<!-- RESULTS:BEGIN -->\n" + body + "\n<!-- RESULTS:END -->", readme, flags=re.DOTALL)
    readme_path.write_text(new, encoding="utf-8")
    print("README results section updated")


if __name__ == "__main__":
    main()
