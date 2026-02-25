#!/usr/bin/env python3
"""
Auto-generate README sections from source code introspection.

Generates content for sections marked with AUTO-GEN markers in README.md:
  <!-- AUTO-GEN:cli-commands:START -->  ...  <!-- AUTO-GEN:cli-commands:END -->
  <!-- AUTO-GEN:mcp-tools:START -->     ...  <!-- AUTO-GEN:mcp-tools:END -->
  <!-- AUTO-GEN:packages:START -->      ...  <!-- AUTO-GEN:packages:END -->

Usage:
  python scripts/generate_readme_sections.py           # Update README in-place
  python scripts/generate_readme_sections.py --check   # Check if README is stale (exit 1 if stale)
  python scripts/generate_readme_sections.py --dry-run # Print generated sections without writing
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
README_PATH = REPO_ROOT / "README.md"


def extract_cli_commands() -> list[tuple[str, str]]:
    """Extract CLI commands from source code.

    Returns list of (command_string, description) tuples.
    """
    commands: list[tuple[str, str]] = []

    cli_main = REPO_ROOT / "packages/cli/src/research_kb_cli/main.py"
    if not cli_main.exists():
        return commands

    content = cli_main.read_text()

    # Extract main commands: @app.command("name") with docstring
    for match in re.finditer(
        r'@app\.command\(\s*(?:name\s*=\s*)?["\']([^"\']+)["\']\s*\)\s*\n'
        r"(?:@[^\n]+\n)*"
        r"\s*(?:async\s+)?def\s+\w+\([^)]*\)(?:\s*->\s*[^:]+)?:\s*\n"
        r'\s*"""([^"]*?)"""',
        content,
        re.MULTILINE,
    ):
        cmd_name = match.group(1)
        docstring = match.group(2).strip().split("\n")[0]  # First line only
        commands.append((f"research-kb {cmd_name}", docstring))

    # Extract subcommand groups
    for group_match in re.finditer(
        r'app\.add_typer\(\s*(\w+),\s*name\s*=\s*["\']([^"\']+)["\']', content
    ):
        group_var = group_match.group(1)
        group_name = group_match.group(2)

        # Find import to locate module file
        import_match = re.search(
            rf"from\s+research_kb_cli\.(\w+)\s+import\s+\w+\s+as\s+{re.escape(group_var)}",
            content,
        )
        if import_match:
            module_name = import_match.group(1)
            group_file = REPO_ROOT / f"packages/cli/src/research_kb_cli/{module_name}.py"
            if group_file.exists():
                group_content = group_file.read_text()
                for cmd_match in re.finditer(
                    r'@\w+\.command\(\s*(?:name\s*=\s*)?["\']([^"\']+)["\']\s*\)\s*\n'
                    r"(?:@[^\n]+\n)*"
                    r"\s*(?:async\s+)?def\s+\w+\([^)]*\)(?:\s*->\s*[^:]+)?:\s*\n"
                    r'\s*"""([^"]*?)"""',
                    group_content,
                    re.MULTILINE,
                ):
                    cmd_name = cmd_match.group(1)
                    docstring = cmd_match.group(2).strip().split("\n")[0]
                    commands.append((f"research-kb {group_name} {cmd_name}", docstring))

    return commands


def extract_mcp_tools() -> list[tuple[str, str]]:
    """Extract MCP tool names and descriptions from source code.

    Returns list of (tool_name, description) tuples.
    """
    tools: list[tuple[str, str]] = []

    tools_dir = REPO_ROOT / "packages/mcp-server/src/research_kb_mcp/tools"
    if not tools_dir.exists():
        return tools

    for py_file in sorted(tools_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        content = py_file.read_text()

        for match in re.finditer(
            r"@mcp\.tool\(\)\s*\n"
            r"\s*async\s+def\s+(\w+)\([^)]*\)(?:\s*->\s*[^:]+)?:\s*\n"
            r'\s*"""([^"]*?)"""',
            content,
            re.MULTILINE,
        ):
            tool_name = match.group(1)
            docstring = match.group(2).strip().split("\n")[0]
            tools.append((tool_name, docstring))

    return tools


def extract_packages() -> list[tuple[str, str, str]]:
    """Extract package info from pyproject.toml files.

    Returns list of (name, description, key_dependency) tuples.
    """
    packages: list[tuple[str, str, str]] = []

    packages_dir = REPO_ROOT / "packages"
    if not packages_dir.exists():
        return packages

    for pkg_dir in sorted(packages_dir.iterdir()):
        if not pkg_dir.is_dir() or pkg_dir.name.startswith("."):
            continue

        pyproject = pkg_dir / "pyproject.toml"
        if not pyproject.exists():
            continue

        content = pyproject.read_text()

        # Extract description
        desc_match = re.search(r'description\s*=\s*"([^"]*)"', content)
        description = desc_match.group(1) if desc_match else ""

        packages.append((pkg_dir.name, description, ""))

    return packages


def generate_section(section_name: str) -> str | None:
    """Generate content for a named section.

    Returns None if section type is unknown.
    """
    if section_name == "cli-commands":
        commands = extract_cli_commands()
        if not commands:
            return None

        lines = [
            "## CLI Reference",
            "",
            "Full command reference with examples: [`docs/CLI.md`](docs/CLI.md)",
            "",
            "Quick reference:",
            "",
            "```bash",
        ]

        # Group commands by prefix for readability
        current_group = ""
        for cmd, desc in commands:
            parts = cmd.split()
            group = parts[1] if len(parts) >= 2 else ""

            # Add blank line between groups
            if group != current_group and current_group:
                lines.append("")
            current_group = group

            # Format: command  # description
            padding = max(1, 55 - len(cmd))
            lines.append(f"{cmd}{' ' * padding}# {desc}")

        lines.append("```")
        return "\n".join(lines)

    elif section_name == "mcp-tools":
        tools = extract_mcp_tools()
        if not tools:
            return None

        # Group tools by category (based on filename they came from)
        lines = [
            "## MCP Server",
            "",
            f"{len(tools)} tools organized by function, designed for conversational use in Claude Code:",
            "",
            "| Tool | Description |",
            "|------|-------------|",
        ]

        for tool_name, desc in tools:
            lines.append(f"| `{tool_name}` | {desc} |")

        return "\n".join(lines)

    elif section_name == "packages":
        packages = extract_packages()
        if not packages:
            return None

        lines = [
            "### Packages",
            "",
            "| Package | Purpose |",
            "|---------|---------|",
        ]

        for name, desc, _ in packages:
            lines.append(f"| `{name}` | {desc} |")

        return "\n".join(lines)

    return None


def check_section_data_points(section_name: str, section_text: str) -> list[str]:
    """Check that all key data points from code appear in a README section.

    Instead of exact string matching, verifies that all tool names, command names,
    or package names extracted from code are present in the section text.
    This allows the README to have richer formatting while catching missing items.

    Returns list of issues (empty = all good).
    """
    issues = []

    if section_name == "cli-commands":
        commands = extract_cli_commands()
        for cmd, _desc in commands:
            # Check the command name appears (e.g., "research-kb query")
            if cmd not in section_text:
                issues.append(f"CLI command '{cmd}' missing from README section")

    elif section_name == "mcp-tools":
        tools = extract_mcp_tools()
        for tool_name, _desc in tools:
            if tool_name not in section_text:
                issues.append(f"MCP tool '{tool_name}' missing from README section")

    elif section_name == "packages":
        packages = extract_packages()
        for name, _desc, _ in packages:
            if name not in section_text:
                issues.append(f"Package '{name}' missing from README section")

    return issues


def update_readme(dry_run: bool = False, check: bool = False) -> bool:
    """Update or check README auto-generated sections.

    Args:
        dry_run: Print generated content without writing.
        check: Return False if README is stale. Uses data-point presence checking
               rather than exact string match, so the README can have richer
               formatting while still catching missing tools/commands/packages.

    Returns:
        True if README is up-to-date (or was successfully updated).
    """
    if not README_PATH.exists():
        print("ERROR: README.md not found")
        return False

    content = README_PATH.read_text()
    updated_content = content
    is_stale = False

    sections = ["cli-commands", "mcp-tools", "packages"]

    for section_name in sections:
        start_marker = f"<!-- AUTO-GEN:{section_name}:START -->"
        end_marker = f"<!-- AUTO-GEN:{section_name}:END -->"

        if start_marker not in content:
            print(f"WARNING: Missing {start_marker} in README.md")
            continue

        if end_marker not in content:
            print(f"WARNING: Missing {end_marker} in README.md")
            continue

        # Extract current content between markers
        pattern = re.escape(start_marker) + r"\n(.*?)\n" + re.escape(end_marker)
        match = re.search(pattern, updated_content, re.DOTALL)
        if not match:
            continue

        current_section = match.group(1)

        if check:
            # In check mode: verify data points are present (not exact match)
            section_issues = check_section_data_points(section_name, current_section)
            if section_issues:
                is_stale = True
                for issue in section_issues:
                    print(f"STALE: {issue}")
        elif dry_run:
            generated = generate_section(section_name)
            if generated:
                print(f"\n--- Generated: {section_name} ---")
                print(generated)
                print(f"--- End: {section_name} ---\n")
        else:
            generated = generate_section(section_name)
            if generated is None:
                print(f"WARNING: Could not generate content for section '{section_name}'")
                continue

            generated_stripped = generated.strip()
            if current_section.strip() != generated_stripped:
                replacement = f"{start_marker}\n{generated}\n{end_marker}"
                updated_content = re.sub(pattern, replacement, updated_content, flags=re.DOTALL)
                print(f"Updated section: {section_name}")

    if not dry_run and not check and updated_content != content:
        README_PATH.write_text(updated_content)
        print("README.md updated.")

    if check and is_stale:
        print("\nREADME has stale auto-generated sections.")
        print("Run: python scripts/generate_readme_sections.py")
        return False

    if check and not is_stale:
        print("All auto-generated sections are up-to-date.")

    return True


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    check = "--check" in sys.argv

    if check:
        return 0 if update_readme(check=True) else 1
    elif dry_run:
        update_readme(dry_run=True)
        return 0
    else:
        update_readme()
        return 0


if __name__ == "__main__":
    sys.exit(main())
