"""
Project-aware API routes for Enso Atlas.

Provides endpoints to list, inspect, and query projects. These are additive —
all existing routes continue to work unchanged using the default project.

Routes:
    GET  /api/projects                        — list all projects
    GET  /api/projects/{project_id}           — get project details
    GET  /api/projects/{project_id}/slides    — slides for this project
    GET  /api/projects/{project_id}/status    — project readiness status
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from .projects import ProjectConfig, ProjectRegistry

logger = logging.getLogger(__name__)

# The registry is injected at startup via `set_registry()`
_registry: Optional[ProjectRegistry] = None


def set_registry(registry: ProjectRegistry):
    """Called from main.py during startup to inject the loaded registry."""
    global _registry
    _registry = registry


def get_registry() -> ProjectRegistry:
    """Get the project registry, raising 503 if not yet loaded."""
    if _registry is None:
        raise HTTPException(
            status_code=503,
            detail="Project registry not initialized yet. Server is starting up.",
        )
    return _registry


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("")
async def list_projects():
    """
    List all configured projects.

    Returns a list of project summaries (id, name, cancer_type, description, classes).
    """
    reg = get_registry()
    projects = reg.list_projects()
    result = []
    for pid, proj in projects.items():
        result.append({
            "id": proj.id,
            "name": proj.name,
            "cancer_type": proj.cancer_type,
            "prediction_target": proj.prediction_target,
            "classes": proj.classes,
            "positive_class": proj.positive_class,
            "description": proj.description,
            "is_default": pid == reg.default_project_id,
        })
    return {
        "projects": result,
        "default_project": reg.default_project_id,
        "count": len(result),
    }


@router.get("/{project_id}")
async def get_project(project_id: str):
    """
    Get full details for a specific project.

    Includes dataset paths, model configuration, threshold, and class definitions.
    """
    reg = get_registry()
    proj = reg.get_project(project_id)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    return {
        "project": proj.to_dict(),
        "is_default": project_id == reg.default_project_id,
    }


@router.get("/{project_id}/slides")
async def get_project_slides(project_id: str):
    """
    Get slides associated with a project.

    Currently queries the slides table filtered by project_id (if populated),
    falling back to returning all slides for the default project for backward
    compatibility.
    """
    reg = get_registry()
    proj = reg.get_project(project_id)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    # Try to get slides from database filtered by project_id
    try:
        from . import database as db

        pool = await db.get_pool()
        async with pool.acquire() as conn:
            # Check if project_id column exists (migration may not have run)
            col_check = await conn.fetchval(
                """
                SELECT COUNT(*) FROM information_schema.columns
                WHERE table_name = 'slides' AND column_name = 'project_id'
                """
            )
            if col_check > 0:
                rows = await conn.fetch(
                    """
                    SELECT s.slide_id, s.patient_id, s.filename, s.width, s.height,
                           s.label, s.has_embeddings, s.has_level0_embeddings, s.num_patches
                    FROM slides s
                    WHERE s.project_id = $1 OR s.project_id IS NULL
                    ORDER BY s.slide_id
                    """,
                    project_id,
                )
            else:
                # project_id column not yet added — return all slides
                rows = await conn.fetch(
                    """
                    SELECT s.slide_id, s.patient_id, s.filename, s.width, s.height,
                           s.label, s.has_embeddings, s.has_level0_embeddings, s.num_patches
                    FROM slides s
                    ORDER BY s.slide_id
                    """
                )

            return {
                "project_id": project_id,
                "slides": [dict(r) for r in rows],
                "count": len(rows),
            }
    except Exception as e:
        logger.warning(f"Database query failed for project slides: {e}")
        # Fallback: return basic project info without slide list
        return {
            "project_id": project_id,
            "slides": [],
            "count": 0,
            "error": "Database not available",
        }


@router.get("/{project_id}/status")
async def get_project_status(project_id: str):
    """
    Get readiness status of a project — are the required models and data available?
    """
    reg = get_registry()
    proj = reg.get_project(project_id)
    if proj is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

    slides_dir = Path(proj.dataset.slides_dir)
    embeddings_dir = Path(proj.dataset.embeddings_dir)
    mil_checkpoint = Path(proj.models.mil_checkpoint)
    labels_file = Path(proj.dataset.labels_file)

    def _count_files(p: Path, glob: str = "*.npy") -> int:
        if not p.exists():
            return 0
        return sum(1 for _ in p.glob(glob))

    exts = ("*.svs", "*.tiff", "*.tif", "*.ndpi")
    slide_count = sum(_count_files(slides_dir, g) for g in exts) if slides_dir.exists() else 0
    embedding_count = _count_files(embeddings_dir) if embeddings_dir.exists() else 0

    return {
        "project_id": project_id,
        "name": proj.name,
        "ready": {
            "slides_dir": slides_dir.exists(),
            "embeddings_dir": embeddings_dir.exists(),
            "mil_checkpoint": mil_checkpoint.exists(),
            "labels_file": labels_file.exists(),
        },
        "counts": {
            "slides": slide_count,
            "embeddings": embedding_count,
        },
        "threshold": proj.threshold,
    }
