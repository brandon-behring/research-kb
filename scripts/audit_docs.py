#!/usr/bin/env python3
"""
Documentation drift detector for research-kb.

Run: python scripts/audit_docs.py

Performs 7 prioritized checks:
1. [Critical] CLI commands in code vs CLAUDE.md/README.md
2. [Critical] Packages in packages/ have README.md
3. [Critical] External paths in docs still exist
4. [High] Extraction backends in code vs README comparison table
5. [High] S2-client commands documented
6. [Medium] API routes in code vs README
7. [Medium] CURRENT_STATUS.md freshness

Exit codes:
- 0: All checks pass
- 1: Warnings found
- 2: Errors found
"""

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Repository root
REPO_ROOT = Path(__file__).parent.parent


# Colors for output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def success(msg: str) -> None:
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")


def warning(msg: str) -> None:
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")


def error(msg: str) -> None:
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")


def header(msg: str) -> None:
    print(f"\n{Colors.BOLD}=== {msg} ==={Colors.RESET}")


def check_cli_commands() -> tuple[bool, list[str]]:
    """Check CLI commands in code vs CLAUDE.md."""
    header("Check 1: CLI Commands Documentation")

    issues = []

    # Find commands in CLI code
    cli_main = REPO_ROOT / "packages/cli/src/research_kb_cli/main.py"
    if not cli_main.exists():
        error("CLI main.py not found")
        return False, ["CLI main.py not found"]

    content = cli_main.read_text()

    # Extract @app.command decorators
    code_commands = set(re.findall(r'@app\.command\(["\'](\w+)["\']', content))

    # Also check for subcommand groups
    subgroups = re.findall(r"(\w+)_app\s*=\s*typer\.Typer", content)
    for group in subgroups:
        group_file = REPO_ROOT / f"packages/cli/src/research_kb_cli/{group}.py"
        if group_file.exists():
            group_content = group_file.read_text()
            sub_commands = re.findall(r'@\w+\.command\(["\'](\w+)["\']', group_content)
            for cmd in sub_commands:
                code_commands.add(f"{group} {cmd}")

    # Check CLAUDE.md
    claude_md = REPO_ROOT / "CLAUDE.md"
    claude_content = claude_md.read_text()

    # Check README.md
    readme = REPO_ROOT / "README.md"
    readme_content = readme.read_text()

    # Commands that should be documented
    critical_commands = {"query", "sources", "stats", "concepts", "graph", "path"}

    for cmd in critical_commands:
        if f"research-kb {cmd}" not in claude_content:
            issues.append(f"Command '{cmd}' missing from CLAUDE.md")
        if f"research-kb {cmd}" not in readme_content:
            issues.append(f"Command '{cmd}' missing from README.md")

    if not issues:
        success(f"Found {len(code_commands)} commands, critical commands documented")
        return True, []
    else:
        for issue in issues:
            warning(issue)
        return False, issues


def check_package_readmes() -> tuple[bool, list[str]]:
    """Check that all packages have README.md."""
    header("Check 2: Package READMEs")

    issues = []
    packages_dir = REPO_ROOT / "packages"

    if not packages_dir.exists():
        error("packages/ directory not found")
        return False, ["packages/ directory not found"]

    for pkg_dir in packages_dir.iterdir():
        if pkg_dir.is_dir() and not pkg_dir.name.startswith("."):
            readme = pkg_dir / "README.md"
            if not readme.exists():
                issues.append(f"Package '{pkg_dir.name}' missing README.md")

    if not issues:
        success("All packages have README.md")
        return True, []
    else:
        for issue in issues:
            warning(issue)
        return False, issues


def check_external_paths() -> tuple[bool, list[str]]:
    """Check that external paths referenced in docs exist."""
    header("Check 3: External Path References")

    issues = []

    # Paths to check (from docs)
    external_paths = [
        Path.home() / "Claude/lever_of_archimedes/hooks/lib/research_kb.sh",
        Path.home() / "Claude/lever_of_archimedes/docs/brain/ideas/research_kb_full_design.md",
        Path.home() / "Claude/lever_of_archimedes/services/health/research_kb_status.jl",
        Path.home() / "Claude/lever_of_archimedes/knowledge/master_bibliography/AVAILABLE.md",
    ]

    for path in external_paths:
        if not path.exists():
            issues.append(f"External path missing: {path}")

    if not issues:
        success("All external paths exist")
        return True, []
    else:
        for issue in issues:
            warning(issue)
        return False, issues


def check_extraction_backends() -> tuple[bool, list[str]]:
    """Check extraction backends in code vs README."""
    header("Check 4: Extraction Backends Documentation")

    issues = []

    # Check for backend files
    extraction_dir = REPO_ROOT / "packages/extraction/src/research_kb_extraction"
    backend_files = {
        "ollama_client.py": "OllamaClient",
        "instructor_client.py": "InstructorOllamaClient",
        "llama_cpp_client.py": "LlamaCppClient",
        "anthropic_client.py": "AnthropicClient",
    }

    code_backends = []
    for filename, classname in backend_files.items():
        if (extraction_dir / filename).exists():
            code_backends.append(classname)

    # Check README
    readme = REPO_ROOT / "packages/extraction/README.md"
    if readme.exists():
        content = readme.read_text()
        for backend in code_backends:
            if backend not in content:
                issues.append(f"Backend '{backend}' not documented in extraction README")
    else:
        issues.append("packages/extraction/README.md missing")

    if not issues:
        success(f"All {len(code_backends)} backends documented")
        return True, []
    else:
        for issue in issues:
            warning(issue)
        return False, issues


def check_s2_commands() -> tuple[bool, list[str]]:
    """Check S2-client commands are documented."""
    header("Check 5: S2-Client Commands")

    issues = []

    # Check for discover and enrich commands
    discover_file = REPO_ROOT / "packages/cli/src/research_kb_cli/discover.py"
    enrich_file = REPO_ROOT / "packages/cli/src/research_kb_cli/enrich.py"

    readme = REPO_ROOT / "README.md"
    readme_content = readme.read_text() if readme.exists() else ""

    if discover_file.exists() and "discover" not in readme_content.lower():
        issues.append("discover commands not in README")

    if enrich_file.exists() and "enrich" not in readme_content.lower():
        issues.append("enrich commands not in README")

    if not issues:
        success("S2 commands documented")
        return True, []
    else:
        for issue in issues:
            warning(issue)
        return False, issues


def check_api_routes() -> tuple[bool, list[str]]:
    """Check API routes are documented."""
    header("Check 6: API Routes Documentation")

    issues = []

    api_dir = REPO_ROOT / "packages/api/src/research_kb_api"
    if not api_dir.exists():
        warning("API package not found")
        return True, []  # Not critical

    readme = REPO_ROOT / "packages/api/README.md"
    if not readme.exists():
        issues.append("packages/api/README.md missing")

    if not issues:
        success("API package has README")
        return True, []
    else:
        for issue in issues:
            warning(issue)
        return False, issues


def check_status_freshness() -> tuple[bool, list[str]]:
    """Check CURRENT_STATUS.md freshness."""
    header("Check 7: Status Documentation Freshness")

    issues = []

    status_file = REPO_ROOT / "docs/status/CURRENT_STATUS.md"
    if not status_file.exists():
        issues.append("CURRENT_STATUS.md not found")
        error(issues[0])
        return False, issues

    # Check modification time
    mtime = datetime.fromtimestamp(status_file.stat().st_mtime)
    age = datetime.now() - mtime

    if age > timedelta(days=7):
        issues.append(f"CURRENT_STATUS.md is {age.days} days old (>7 days)")
        warning(issues[0])
        return False, issues

    success(f"CURRENT_STATUS.md updated {age.days} days ago")
    return True, []


def main() -> int:
    """Run all documentation audits."""
    print(f"{Colors.BOLD}Documentation Audit - research-kb{Colors.RESET}")
    print(f"Repository: {REPO_ROOT}")
    print(f"Time: {datetime.now().isoformat()}")

    all_passed = True
    all_issues = []

    checks = [
        ("Critical", check_cli_commands),
        ("Critical", check_package_readmes),
        ("Critical", check_external_paths),
        ("High", check_extraction_backends),
        ("High", check_s2_commands),
        ("Medium", check_api_routes),
        ("Medium", check_status_freshness),
    ]

    critical_failed = False

    for priority, check_fn in checks:
        passed, issues = check_fn()
        if not passed:
            all_passed = False
            all_issues.extend(issues)
            if priority == "Critical":
                critical_failed = True

    # Summary
    header("Summary")
    if all_passed:
        success("All checks passed!")
        return 0
    else:
        print(f"\nTotal issues found: {len(all_issues)}")
        if critical_failed:
            error("Critical issues found - please fix")
            return 2
        else:
            warning("Warnings found - consider fixing")
            return 1


if __name__ == "__main__":
    sys.exit(main())
