#!/usr/bin/env python3
"""Validate the Bayesian/statistics curriculum scaffold."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CURRICULUM_DIR = REPO_ROOT / "docs" / "curriculum"
REGISTRY_PATH = CURRICULUM_DIR / "material_registry.json"
MANIFEST_PATH = CURRICULUM_DIR / "module_manifest.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def validate() -> list[str]:
    errors: list[str] = []

    if not REGISTRY_PATH.exists():
        return [f"Missing registry: {REGISTRY_PATH}"]
    if not MANIFEST_PATH.exists():
        return [f"Missing manifest: {MANIFEST_PATH}"]

    registry = _read_json(REGISTRY_PATH)
    manifest = _read_json(MANIFEST_PATH)

    allowed_states = set(registry["source_states"])
    allowed_roles = set(registry["role_tags"])
    modules = manifest["modules"]
    module_ids = [module["id"] for module in modules]
    module_map = {module["id"]: module for module in modules}
    module_order = {module_id: idx for idx, module_id in enumerate(module_ids)}
    material_ids = {item["id"] for item in registry["materials"]}

    for material in registry["materials"]:
        if material["source_state"] not in allowed_states:
            errors.append(f"Invalid source_state for {material['id']}: {material['source_state']}")
        if material["role"] not in allowed_roles:
            errors.append(f"Invalid role for {material['id']}: {material['role']}")
        for module_id in material["module_ids"]:
            if module_id not in module_map:
                errors.append(f"Material {material['id']} points to unknown module {module_id}")

    for item in registry.get("explicit_out_of_scope", []):
        if item["role"] != "out_of_scope":
            errors.append(f"Out-of-scope item has wrong role: {item['title']}")
        if item["source_state"] not in allowed_states:
            errors.append(f"Out-of-scope item has invalid state: {item['title']}")

    if manifest["trunk_modules"] != [
        "01_probability_and_stat_baseline",
        "02_core_bayesian_inference",
        "03_computational_bayes",
        "04_bayesian_workflow_and_model_checks",
        "05_regression_and_glms",
        "06_hierarchical_models",
        "07_time_series_and_forecasting",
    ]:
        errors.append("Trunk modules do not match the expected Bayes/stats trunk order")

    for module in modules:
        for prereq in module["prerequisites"]:
            if prereq not in module_map:
                errors.append(f"Module {module['id']} has unknown prerequisite {prereq}")
                continue
            if module_order[prereq] >= module_order[module["id"]]:
                errors.append(
                    f"Module {module['id']} depends on {prereq}, which is not earlier in order"
                )

        artifacts = module["artifacts"]
        required_files = [
            artifacts["readme"],
            artifacts["case_study"],
            artifacts["worked_exercises"],
            artifacts["unworked_exercises"],
            artifacts["references"]["primary"],
            artifacts["references"]["support"],
            artifacts["references"]["optional"],
        ]
        for rel_path in required_files:
            if not (REPO_ROOT / rel_path).exists():
                errors.append(f"Missing artifact for {module['id']}: {rel_path}")

        notebook_paths = artifacts["notebooks"]
        if len(notebook_paths) < 2:
            errors.append(f"Module {module['id']} has fewer than 2 notebooks")
        for rel_path in notebook_paths:
            path = REPO_ROOT / rel_path
            if not path.exists():
                errors.append(f"Missing notebook for {module['id']}: {rel_path}")
                continue
            try:
                notebook = _read_json(path)
                if notebook.get("nbformat") != 4:
                    errors.append(f"Notebook has wrong nbformat: {rel_path}")
                code_cell = next(
                    (cell for cell in notebook["cells"] if cell.get("cell_type") == "code"),
                    None,
                )
                if code_cell is None:
                    errors.append(f"Notebook has no code cell: {rel_path}")
                    continue
                code = "".join(code_cell.get("source", []))
                compile(code, rel_path, "exec")
            except Exception as exc:  # pragma: no cover - validated through CLI/test
                errors.append(f"Invalid notebook {rel_path}: {exc}")

        for material_group in (
            module["primary_material_ids"],
            module["support_material_ids"],
            module["optional_material_ids"],
        ):
            for material_id in material_group:
                if material_id not in material_ids:
                    errors.append(
                        f"Module {module['id']} references unknown material {material_id}"
                    )

    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("Curriculum validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Bayesian curriculum validation passed.")
    print(f"Registry: {REGISTRY_PATH.relative_to(REPO_ROOT)}")
    print(f"Manifest: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
