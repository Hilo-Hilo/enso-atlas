from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_model_heatmap_proxy_forwards_project_id_to_backend():
    src = _read("frontend/src/app/api/heatmap/[slideId]/[modelId]/route.ts")

    assert 'searchParams.get("project_id")' in src
    assert 'backendParams.set("project_id", projectId)' in src


def test_slide_heatmap_proxy_forwards_project_and_alpha_params_to_backend():
    src = _read("frontend/src/app/api/heatmap/[slideId]/route.ts")

    assert 'searchParams.get("project_id")' in src
    assert 'backendParams.set("project_id", projectId)' in src
    assert 'searchParams.get("alpha_power")' in src
    assert 'backendParams.set("alpha_power", alphaPower)' in src


def test_api_client_uses_project_scoped_models_endpoint_and_heatmap_query_param():
    src = _read("frontend/src/lib/api.ts")

    assert "/api/models" in src
    assert "project_id" in src
    assert "getProjectAvailableModels" in src
    assert "getHeatmapUrl" in src
    assert "params.set('project_id', projectId)" in src or 'params.set("project_id", projectId)' in src


def test_api_client_heatmap_url_supports_analysis_run_nonce_for_model_heatmaps():
    src = _read("frontend/src/lib/api.ts")

    assert "analysisRunId" in src
    assert "analysis_run_id" in src
    assert "params.set('refresh', 'true')" in src or 'params.set("refresh", "true")' in src


def test_model_heatmap_proxy_preserves_backend_error_payload_for_ui_messages():
    src = _read("frontend/src/app/api/heatmap/[slideId]/[modelId]/route.ts")

    assert "const errorBody = await response.text();" in src
    assert "return new NextResponse(" in src
    assert '"Content-Type": response.headers.get("Content-Type") || "application/json"' in src


def test_model_heatmap_proxy_forwards_analysis_nonce_and_refresh_controls():
    src = _read("frontend/src/app/api/heatmap/[slideId]/[modelId]/route.ts")

    assert 'searchParams.get("analysis_run_id")' in src
    assert 'backendParams.set("analysis_run_id", analysisRunId)' in src
    assert 'backendParams.set("refresh", "true")' in src


def test_model_heatmap_proxy_disables_response_caching():
    src = _read("frontend/src/app/api/heatmap/[slideId]/[modelId]/route.ts")

    assert 'cache: "no-store"' in src
    assert '"Cache-Control": "no-store, max-age=0"' in src


def test_slide_heatmap_proxy_preserves_backend_error_payload_for_ui_messages():
    src = _read("frontend/src/app/api/heatmap/[slideId]/route.ts")

    assert "const errorBody = await response.text();" in src
    assert "return new NextResponse(" in src
    assert '"Content-Type": response.headers.get("Content-Type") || "application/json"' in src


def test_page_heatmap_model_options_are_guarded_by_current_project_scope():
    src = _read("frontend/src/app/page.tsx")

    assert "projectAvailableModelsScopeId" in src
    assert "projectAvailableModelsScopeId === currentProject.id" in src
    assert "normalizedHeatmapModel" in src
    assert "!scopedProjectModelIds.has(heatmapModel)" in src


def test_page_bumps_analysis_run_nonce_and_threads_it_into_heatmap_url():
    src = _read("frontend/src/app/page.tsx")

    assert "const [analysisRunId, setAnalysisRunId] = useState<number>(0);" in src
    assert "bumpAnalysisRunId();" in src
    assert "analysisRunId," in src
