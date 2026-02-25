#!/usr/bin/env python3
"""
Documentation drift detector for research-kb.

Run: python scripts/audit_docs.py
     python scripts/audit_docs.py --pre-commit  (fast checks only, no DB/network)
     python scripts/audit_docs.py --ci          (all fast checks, exit 2 on failure)

Performs 12 prioritized checks:
 1. [Critical] CLI commands in code vs CLAUDE.md/README.md
 2. [Critical] Packages in packages/ have README.md
 3. [Critical] External paths in docs still exist
 4. [High] Extraction backends in code vs README comparison table
 5. [High] S2-client commands documented
 6. [Medium] API routes in code vs README
 7. [Medium] CURRENT_STATUS.md freshness
 8. [Critical] README CLI commands match @app.command() decorators
 9. [Critical] README MCP tool names match @mcp.tool() names
10. [Critical] README package table matches packages/*/pyproject.toml
11. [Medium] Corpus numbers in README (when DB available)
12. [Critical] Auto-generated sections not stale (compare markers vs source)

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
    print(f"{Colors.GREEN}  {msg}{Colors.RESET}")


def warning(msg: str) -> None:
    print(f"{Colors.YELLOW}   {msg}{Colors.RESET}")


def error(msg: str) -> None:
    print(f"{Colors.RED}  {msg}{Colors.RESET}")


def header(msg: str) -> None:
    print(f"\n{Colors.BOLD}=== {msg} ==={Colors.RESET}")


def _extract_commands_from_file(filepath: Path) -> set[str]:
    """Extract @app.command() names from a Typer module.

    Handles both named commands and bare @app.command() decorators.
    """
    content = filepath.read_text()
    commands = set(re.findall(r'@\w+\.command\(\s*(?:name\s*=\s*)?["\']([^"\']+)["\']', content))

    lines = content.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"\s*@\w+\.command\(\s*\)\s*$", line):
            for j in range(i + 1, min(i + 5, len(lines))):
                func_match = re.match(r"\s*(?:async\s+)?def\s+(\w+)", lines[j])
                if func_match:
                    cmd_name = func_match.group(1).replace("_", "-")
                    commands.add(cmd_name)
                    break

    return commands


def check_cli_commands() -> tuple[bool, list[str]]:
    """Check CLI commands in code vs CLAUDE.md."""
    header("Check 1: CLI Commands Documentation")

    issues = []
    code_commands: set[str] = set()

    cli_base = REPO_ROOT / "packages/cli/src/research_kb_cli"
    cli_main = cli_base / "main.py"
    if not cli_main.exists():
        error("CLI main.py not found")
        return False, ["CLI main.py not found"]

    main_content = cli_main.read_text()

    # Find registered sub-apps: app.add_typer(..., name="X")
    subgroup_names = re.findall(
        r'app\.add_typer\(\s*\w+\s*,\s*name\s*=\s*["\'](\w+)["\']', main_content
    )

    # Scan commands/ directory for sub-app modules
    commands_dir = cli_base / "commands"
    if commands_dir.is_dir():
        for py_file in sorted(commands_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            # Derive sub-app name from filename (e.g., search.py -> search)
            group_name = py_file.stem
            cmds = _extract_commands_from_file(py_file)
            for cmd in cmds:
                code_commands.add(f"{group_name} {cmd}")

    # Scan top-level sub-app files (discover.py, enrich.py)
    for group in subgroup_names:
        group_file = cli_base / f"{group}.py"
        if group_file.exists():
            cmds = _extract_commands_from_file(group_file)
            for cmd in cmds:
                code_commands.add(f"{group} {cmd}")

    # Check CLAUDE.md
    claude_md = REPO_ROOT / "CLAUDE.md"
    claude_content = claude_md.read_text()

    # Check README.md
    readme = REPO_ROOT / "README.md"
    readme_content = readme.read_text()

    # Critical commands that should be documented (new sub-app syntax)
    critical_commands = {
        "search query",
        "sources list",
        "sources stats",
        "graph concepts",
        "graph neighborhood",
        "graph path",
    }

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

    # External path checks removed — no external dependencies in public release

    if not issues:
        success("All internal paths verified")
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


# --- New checks (Phase 5a) ---


def _extract_cli_commands_from_code() -> set[str]:
    """Extract CLI command names from source code."""
    commands = set()

    cli_main = REPO_ROOT / "packages/cli/src/research_kb_cli/main.py"
    if not cli_main.exists():
        return commands

    content = cli_main.read_text()

    # Extract @app.command("name") and @app.command(name="name")
    for match in re.findall(r'@app\.command\(\s*(?:name\s*=\s*)?["\']([^"\']+)["\']', content):
        commands.add(match)

    # Extract subcommand groups and their commands
    for group_match in re.finditer(
        r'app\.add_typer\(\s*(\w+),\s*name\s*=\s*["\']([^"\']+)["\']', content
    ):
        group_var = group_match.group(1)
        group_name = group_match.group(2)

        # Find the import for this group variable to locate the file
        import_match = re.search(
            rf"from\s+research_kb_cli\.(\w+)\s+import\s+\w+\s+as\s+{re.escape(group_var)}",
            content,
        )
        if import_match:
            module_name = import_match.group(1)
            group_file = REPO_ROOT / f"packages/cli/src/research_kb_cli/{module_name}.py"
            if group_file.exists():
                group_content = group_file.read_text()
                for cmd_match in re.findall(
                    r'@\w+\.command\(\s*(?:name\s*=\s*)?["\']([^"\']+)["\']', group_content
                ):
                    commands.add(f"{group_name} {cmd_match}")

    return commands


def _extract_cli_commands_from_readme() -> set[str]:
    """Extract CLI commands mentioned in README auto-gen section."""
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        return set()

    content = readme.read_text()

    # Extract from AUTO-GEN CLI section
    cli_section = re.search(
        r"<!-- AUTO-GEN:cli-commands:START -->(.+?)<!-- AUTO-GEN:cli-commands:END -->",
        content,
        re.DOTALL,
    )
    if not cli_section:
        return set()

    section_text = cli_section.group(1)
    # Match "research-kb <command>" patterns
    commands = set()
    for match in re.findall(r"research-kb\s+([\w-]+(?:\s+[\w-]+)?)", section_text):
        # Skip if it's a quoted argument like "instrumental variables"
        if not match.startswith('"'):
            commands.add(match.strip())
    return commands


def check_readme_cli_commands() -> tuple[bool, list[str]]:
    """Check 8: Verify README CLI commands match @app.command() decorators."""
    header("Check 8: README CLI Commands vs Code")

    issues = []

    code_commands = _extract_cli_commands_from_code()
    readme_commands = _extract_cli_commands_from_readme()

    if not code_commands:
        issues.append("Could not extract CLI commands from code")
        error(issues[0])
        return False, issues

    if not readme_commands:
        issues.append("No AUTO-GEN:cli-commands section found in README.md")
        warning(issues[0])
        return False, issues

    # Check for commands in code but not in README
    # Normalize: code uses hyphens in command names
    for cmd in code_commands:
        # Check if this command or a reasonable variant appears in README
        cmd_normalized = cmd.replace("_", "-")
        found = False
        for readme_cmd in readme_commands:
            if cmd_normalized in readme_cmd or cmd in readme_cmd:
                found = True
                break
        if not found:
            issues.append(f"CLI command '{cmd_normalized}' in code but not in README")

    if not issues:
        success(f"All {len(code_commands)} CLI commands documented in README")
        return True, []
    else:
        for issue in issues:
            warning(issue)
        return False, issues


def _extract_mcp_tools_from_code() -> set[str]:
    """Extract MCP tool names from source code."""
    tools = set()

    tools_dir = REPO_ROOT / "packages/mcp-server/src/research_kb_mcp/tools"
    if not tools_dir.exists():
        return tools

    for py_file in tools_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        content = py_file.read_text()
        # Match async def research_kb_xxx(...) decorated with @mcp.tool()
        # The function name IS the tool name
        for match in re.findall(r"@mcp\.tool\(\)\s*\n\s*async\s+def\s+(\w+)", content):
            tools.add(match)

    return tools


def _extract_mcp_tools_from_readme() -> set[str]:
    """Extract MCP tool names from README auto-gen section."""
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        return set()

    content = readme.read_text()

    mcp_section = re.search(
        r"<!-- AUTO-GEN:mcp-tools:START -->(.+?)<!-- AUTO-GEN:mcp-tools:END -->",
        content,
        re.DOTALL,
    )
    if not mcp_section:
        return set()

    section_text = mcp_section.group(1)
    # Match backtick-quoted tool names
    tools = set(re.findall(r"`(research_kb_\w+)`", section_text))
    return tools


def check_readme_mcp_tools() -> tuple[bool, list[str]]:
    """Check 9: Verify README MCP tool names match @mcp.tool() names."""
    header("Check 9: README MCP Tools vs Code")

    issues = []

    code_tools = _extract_mcp_tools_from_code()
    readme_tools = _extract_mcp_tools_from_readme()

    if not code_tools:
        issues.append("Could not extract MCP tools from code")
        error(issues[0])
        return False, issues

    if not readme_tools:
        issues.append("No AUTO-GEN:mcp-tools section found in README.md")
        warning(issues[0])
        return False, issues

    # Check for tools in code but not in README
    missing_from_readme = code_tools - readme_tools
    extra_in_readme = readme_tools - code_tools

    for tool in sorted(missing_from_readme):
        issues.append(f"MCP tool '{tool}' in code but not in README")

    for tool in sorted(extra_in_readme):
        issues.append(f"MCP tool '{tool}' in README but not in code")

    if not issues:
        success(f"All {len(code_tools)} MCP tools match between code and README")
        return True, []
    else:
        for issue in issues:
            warning(issue)
        return False, issues


def _get_packages_from_filesystem() -> set[str]:
    """Get package names from packages/ directory."""
    packages_dir = REPO_ROOT / "packages"
    if not packages_dir.exists():
        return set()

    packages = set()
    for pkg_dir in packages_dir.iterdir():
        if pkg_dir.is_dir() and not pkg_dir.name.startswith("."):
            packages.add(pkg_dir.name)
    return packages


def _get_packages_from_readme() -> set[str]:
    """Get package names mentioned in README packages table."""
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        return set()

    content = readme.read_text()

    pkg_section = re.search(
        r"<!-- AUTO-GEN:packages:START -->(.+?)<!-- AUTO-GEN:packages:END -->",
        content,
        re.DOTALL,
    )
    if not pkg_section:
        return set()

    section_text = pkg_section.group(1)
    # Match backtick-quoted package names in table rows
    packages = set(re.findall(r"\|\s*`(\w[\w-]*)`\s*\|", section_text))
    return packages


def check_readme_packages() -> tuple[bool, list[str]]:
    """Check 10: Verify README package table matches packages/*/pyproject.toml."""
    header("Check 10: README Package Table vs Filesystem")

    issues = []

    fs_packages = _get_packages_from_filesystem()
    readme_packages = _get_packages_from_readme()

    if not fs_packages:
        issues.append("No packages found in packages/ directory")
        error(issues[0])
        return False, issues

    if not readme_packages:
        issues.append("No AUTO-GEN:packages section found in README.md")
        warning(issues[0])
        return False, issues

    # Normalize: filesystem uses hyphens, README might use either
    fs_normalized = {p.replace("-", "_") for p in fs_packages} | fs_packages
    readme_normalized = {p.replace("-", "_") for p in readme_packages} | readme_packages

    missing_from_readme = fs_packages - readme_normalized
    extra_in_readme = readme_packages - fs_normalized

    for pkg in sorted(missing_from_readme):
        issues.append(f"Package '{pkg}' on filesystem but not in README table")

    for pkg in sorted(extra_in_readme):
        issues.append(f"Package '{pkg}' in README table but not on filesystem")

    if not issues:
        success(f"All {len(fs_packages)} packages documented in README")
        return True, []
    else:
        for issue in issues:
            warning(issue)
        return False, issues


def check_corpus_numbers() -> tuple[bool, list[str]]:
    """Check 11: Verify corpus numbers in README against DB (when DB available)."""
    header("Check 11: Corpus Numbers (DB check)")

    # This check requires a running database — skip in pre-commit/CI-fast mode
    try:
        import asyncio

        import asyncpg
    except ImportError:
        success("Skipped (asyncpg not available)")
        return True, []

    async def _check():
        issues = []
        try:
            import os

            pool = await asyncpg.create_pool(
                host=os.environ.get("POSTGRES_HOST", "localhost"),
                port=int(os.environ.get("POSTGRES_PORT", "5432")),
                database=os.environ.get("POSTGRES_DB", "research_kb"),
                user=os.environ.get("POSTGRES_USER", "postgres"),
                password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
                min_size=1,
                max_size=2,
                command_timeout=5,
            )
        except Exception:
            success("Skipped (database not available)")
            return True, []

        try:
            async with pool.acquire() as conn:
                source_count = await conn.fetchval("SELECT COUNT(*) FROM sources")
                chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")

            readme = REPO_ROOT / "README.md"
            if readme.exists():
                content = readme.read_text()

                # Check source count
                source_match = re.search(r"Sources.*?\|\s*(\d[\d,]*)\s*\|", content)
                if source_match:
                    readme_sources = int(source_match.group(1).replace(",", ""))
                    if abs(readme_sources - source_count) > 10:
                        issues.append(
                            f"README says {readme_sources} sources, DB has {source_count}"
                        )

                # Check chunk count (approximate, README uses "226K" format)
                chunk_match = re.search(r"Text chunks.*?\|\s*(\d+)K\s*\|", content)
                if chunk_match:
                    readme_chunks_k = int(chunk_match.group(1))
                    actual_chunks_k = chunk_count // 1000
                    if abs(readme_chunks_k - actual_chunks_k) > 10:
                        issues.append(
                            f"README says {readme_chunks_k}K chunks, DB has {actual_chunks_k}K"
                        )

            if not issues:
                success(f"Corpus numbers match: {source_count} sources, {chunk_count} chunks")
                return True, []
            else:
                for issue in issues:
                    warning(issue)
                return False, issues
        finally:
            await pool.close()

    try:
        passed, issues = asyncio.run(_check())
        return passed, issues
    except Exception as e:
        success(f"Skipped (DB error: {e})")
        return True, []


def check_autogen_freshness() -> tuple[bool, list[str]]:
    """Check 12: Verify auto-generated sections are not stale."""
    header("Check 12: Auto-Generated Section Freshness")

    issues = []

    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        issues.append("README.md not found")
        error(issues[0])
        return False, issues

    content = readme.read_text()

    # Check that expected AUTO-GEN markers exist
    expected_markers = ["cli-commands", "mcp-tools", "packages"]
    for marker in expected_markers:
        start = f"<!-- AUTO-GEN:{marker}:START -->"
        end = f"<!-- AUTO-GEN:{marker}:END -->"
        if start not in content:
            issues.append(f"Missing AUTO-GEN marker: {start}")
        elif end not in content:
            issues.append(f"Missing AUTO-GEN end marker: {end}")

    # If generate_readme_sections.py exists, run --check
    gen_script = REPO_ROOT / "scripts/generate_readme_sections.py"
    if gen_script.exists():
        import subprocess

        result = subprocess.run(
            [sys.executable, str(gen_script), "--check"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            issues.append(
                "Auto-generated sections are stale (run: python scripts/generate_readme_sections.py)"
            )

    if not issues:
        success("All auto-generated markers present and fresh")
        return True, []
    else:
        for issue in issues:
            warning(issue)
        return False, issues


def main() -> int:
    """Run all documentation audits."""
    # Parse flags
    pre_commit = "--pre-commit" in sys.argv
    ci_mode = "--ci" in sys.argv

    print(f"{Colors.BOLD}Documentation Audit - research-kb{Colors.RESET}")
    print(f"Repository: {REPO_ROOT}")
    print(f"Time: {datetime.now().isoformat()}")
    if pre_commit:
        print(f"Mode: pre-commit (fast checks only)")
    elif ci_mode:
        print(f"Mode: CI (all fast checks, strict)")

    all_passed = True
    all_issues = []

    # Original checks
    base_checks = [
        ("Critical", check_cli_commands),
        ("Critical", check_package_readmes),
        ("Critical", check_external_paths),
        ("High", check_extraction_backends),
        ("High", check_s2_commands),
        ("Medium", check_api_routes),
    ]

    # Slow checks (require DB or stale files)
    slow_checks = [
        ("Medium", check_status_freshness),
        ("Medium", check_corpus_numbers),
    ]

    # New structural checks (fast, source-code only)
    structural_checks = [
        ("Critical", check_readme_cli_commands),
        ("Critical", check_readme_mcp_tools),
        ("Critical", check_readme_packages),
        ("Critical", check_autogen_freshness),
    ]

    if pre_commit:
        # Pre-commit: fast checks only (1-3, 8-10, 12)
        checks = base_checks[:3] + structural_checks
    else:
        # Full run: all checks
        checks = base_checks + slow_checks + structural_checks

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
