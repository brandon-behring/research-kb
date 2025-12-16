#!/usr/bin/env python3
"""
Auto-generate package READMEs from code.

Run: python scripts/generate_package_docs.py

Generates:
- packages/api/README.md (from FastAPI routes)
- packages/dashboard/README.md (from Streamlit pages)

Uses hybrid approach: templates + grep extraction (not full AST).
"""

import re
import sys
from pathlib import Path
from datetime import datetime

# Repository root
REPO_ROOT = Path(__file__).parent.parent


def extract_fastapi_routes(api_dir: Path) -> list[dict]:
    """Extract FastAPI route information using grep patterns."""
    routes = []

    # Look for router files
    routers_dir = api_dir / "src/research_kb_api/routers"
    if not routers_dir.exists():
        # Try alternative structure
        src_dir = api_dir / "src/research_kb_api"
        if src_dir.exists():
            router_files = list(src_dir.glob("*.py"))
        else:
            return routes
    else:
        router_files = list(routers_dir.glob("*.py"))

    for router_file in router_files:
        if router_file.name.startswith("_"):
            continue

        content = router_file.read_text()
        module_name = router_file.stem

        # Find route decorators
        # Patterns: @router.get("/path"), @app.post("/path")
        route_pattern = r'@(?:router|app)\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']'
        matches = re.finditer(route_pattern, content)

        for match in matches:
            method, path = match.groups()

            # Try to find the function name and docstring after the decorator
            func_match = re.search(
                rf'{re.escape(match.group(0))}.*?\n(?:async\s+)?def\s+(\w+)\([^)]*\)(?:\s*->\s*[^:]+)?:\s*(?:"""([^"]*?)""")?',
                content,
                re.DOTALL
            )

            func_name = func_match.group(1) if func_match else "unknown"
            docstring = func_match.group(2).strip() if func_match and func_match.group(2) else ""

            routes.append({
                "module": module_name,
                "method": method.upper(),
                "path": path,
                "function": func_name,
                "description": docstring.split('\n')[0] if docstring else "",
            })

    return routes


def extract_streamlit_pages(dashboard_dir: Path) -> list[dict]:
    """Extract Streamlit page information."""
    pages = []

    # Look for pages directory
    pages_dir = dashboard_dir / "src/research_kb_dashboard/pages"
    if not pages_dir.exists():
        # Try alternative
        src_dir = dashboard_dir / "src/research_kb_dashboard"
        if src_dir.exists():
            page_files = list(src_dir.glob("*.py"))
        else:
            return pages
    else:
        page_files = list(pages_dir.glob("*.py"))

    for page_file in page_files:
        if page_file.name.startswith("_"):
            continue

        content = page_file.read_text()

        # Look for st.set_page_config or st.title
        title_match = re.search(r'st\.(?:set_page_config|title)\([^)]*["\']([^"\']+)["\']', content)
        title = title_match.group(1) if title_match else page_file.stem.replace("_", " ").title()

        # Look for module docstring
        docstring_match = re.match(r'^["\'"]{3}(.*?)["\'"]{3}', content, re.DOTALL)
        description = docstring_match.group(1).strip().split('\n')[0] if docstring_match else ""

        pages.append({
            "file": page_file.name,
            "title": title,
            "description": description,
        })

    return pages


def generate_api_readme(api_dir: Path) -> str:
    """Generate README for API package."""
    routes = extract_fastapi_routes(api_dir)

    # Group routes by module
    by_module: dict[str, list[dict]] = {}
    for route in routes:
        module = route["module"]
        if module not in by_module:
            by_module[module] = []
        by_module[module].append(route)

    # Generate markdown
    lines = [
        "# Research KB API",
        "",
        "FastAPI REST API for the research knowledge base.",
        "",
        "## Quick Start",
        "",
        "```bash",
        "# Start the API server",
        "uvicorn research_kb_api.main:app --reload",
        "",
        "# Or with Docker",
        "docker-compose up api",
        "```",
        "",
        "## Endpoints",
        "",
    ]

    if by_module:
        for module, module_routes in sorted(by_module.items()):
            lines.append(f"### {module.replace('_', ' ').title()}")
            lines.append("")
            lines.append("| Method | Path | Description |")
            lines.append("|--------|------|-------------|")
            for route in module_routes:
                desc = route["description"] or route["function"]
                lines.append(f"| {route['method']} | `{route['path']}` | {desc} |")
            lines.append("")
    else:
        lines.append("_No routes found. Run extraction to populate._")
        lines.append("")

    lines.extend([
        "## Configuration",
        "",
        "Environment variables:",
        "",
        "| Variable | Default | Description |",
        "|----------|---------|-------------|",
        "| `DATABASE_URL` | `postgresql://...` | Database connection string |",
        "| `EMBEDDING_SERVER_URL` | `http://localhost:8080` | Embedding server URL |",
        "",
        "---",
        f"*Generated by scripts/generate_package_docs.py on {datetime.now().strftime('%Y-%m-%d')}*",
    ])

    return "\n".join(lines)


def generate_dashboard_readme(dashboard_dir: Path) -> str:
    """Generate README for dashboard package."""
    pages = extract_streamlit_pages(dashboard_dir)

    lines = [
        "# Research KB Dashboard",
        "",
        "Streamlit visualization dashboard for the research knowledge base.",
        "",
        "## Quick Start",
        "",
        "```bash",
        "# Start the dashboard",
        "streamlit run packages/dashboard/src/research_kb_dashboard/app.py",
        "",
        "# Or with Docker",
        "docker-compose up dashboard",
        "```",
        "",
        "## Pages",
        "",
    ]

    if pages:
        lines.append("| Page | Description |")
        lines.append("|------|-------------|")
        for page in pages:
            desc = page["description"] or page["title"]
            lines.append(f"| {page['title']} | {desc} |")
        lines.append("")
    else:
        lines.append("_No pages found._")
        lines.append("")

    lines.extend([
        "## Features",
        "",
        "- **Search**: Hybrid search with FTS + vector + graph",
        "- **Graph Explorer**: Interactive concept graph visualization (PyVis)",
        "- **Citation Network**: Paper citation relationships",
        "- **Statistics**: Corpus metrics and health checks",
        "",
        "## Configuration",
        "",
        "Environment variables:",
        "",
        "| Variable | Default | Description |",
        "|----------|---------|-------------|",
        "| `DATABASE_URL` | `postgresql://...` | Database connection string |",
        "| `EMBEDDING_SERVER_URL` | `http://localhost:8080` | Embedding server URL |",
        "",
        "---",
        f"*Generated by scripts/generate_package_docs.py on {datetime.now().strftime('%Y-%m-%d')}*",
    ])

    return "\n".join(lines)


def main() -> int:
    """Generate package READMEs."""
    print("Generating package documentation...")

    # API README
    api_dir = REPO_ROOT / "packages/api"
    if api_dir.exists():
        readme_content = generate_api_readme(api_dir)
        readme_path = api_dir / "README.md"
        readme_path.write_text(readme_content)
        print(f"  ✅ Generated {readme_path}")
    else:
        print(f"  ⚠️  API package not found at {api_dir}")

    # Dashboard README
    dashboard_dir = REPO_ROOT / "packages/dashboard"
    if dashboard_dir.exists():
        readme_content = generate_dashboard_readme(dashboard_dir)
        readme_path = dashboard_dir / "README.md"
        readme_path.write_text(readme_content)
        print(f"  ✅ Generated {readme_path}")
    else:
        print(f"  ⚠️  Dashboard package not found at {dashboard_dir}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
