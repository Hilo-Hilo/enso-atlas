"use client";

import React, { useState, useCallback, useEffect } from "react";
import dynamic from "next/dynamic";
import { Header } from "@/components/layout/Header";
import { Footer } from "@/components/layout/Footer";
import {
  SlideSelector,
  PredictionPanel,
  EvidencePanel,
  SimilarCasesPanel,
  ReportPanel,
  SemanticSearchPanel,
  CaseNotesPanel,
  QuickStatsPanel,
  recordAnalysis,
} from "@/components/panels";
import { PatchZoomModal } from "@/components/modals";
import { useAnalysis } from "@/hooks/useAnalysis";
import { getDziUrl, healthCheck, exportReportPdf, semanticSearch } from "@/lib/api";
import type { SlideInfo, PatchCoordinates, SemanticSearchResult, EvidencePatch } from "@/types";

// Dynamically import WSIViewer to prevent SSR issues with OpenSeadragon
const WSIViewer = dynamic(
  () => import("@/components/viewer/WSIViewer").then((mod) => mod.WSIViewer),
  { ssr: false, loading: () => <div className="h-full flex items-center justify-center bg-gray-100 rounded-lg">Loading viewer...</div> }
);

export default function HomePage() {
  // State
  const [selectedSlide, setSelectedSlide] = useState<SlideInfo | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [selectedPatchId, setSelectedPatchId] = useState<string | undefined>();

  // Patch zoom modal state
  const [zoomModalOpen, setZoomModalOpen] = useState(false);
  const [zoomedPatch, setZoomedPatch] = useState<EvidencePatch | null>(null);

  // Semantic search state
  const [semanticResults, setSemanticResults] = useState<SemanticSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  // Analysis hook
  const {
    isAnalyzing,
    isGeneratingReport,
    analysisResult,
    report,
    error,
    analyze,
    generateSlideReport,
    clearResults,
    clearError,
  } = useAnalysis();

  // Check backend connection
  useEffect(() => {
    const checkConnection = async () => {
      try {
        await healthCheck();
        setIsConnected(true);
      } catch {
        setIsConnected(false);
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, []);

  // Handle slide selection
  const handleSlideSelect = useCallback(
    (slide: SlideInfo) => {
      setSelectedSlide(slide);
      clearResults();
      setSelectedPatchId(undefined);
      setSemanticResults([]);
      setSearchError(null);
    },
    [clearResults]
  );

  // Handle semantic search
  const handleSemanticSearch = useCallback(
    async (query: string, topK: number) => {
      if (!selectedSlide) return;

      setIsSearching(true);
      setSearchError(null);

      try {
        const response = await semanticSearch(selectedSlide.id, query, topK);
        setSemanticResults(response.results);
      } catch (err) {
        console.error("Semantic search failed:", err);
        setSearchError(
          err instanceof Error ? err.message : "Search failed. Please try again."
        );
        setSemanticResults([]);
      } finally {
        setIsSearching(false);
      }
    },
    [selectedSlide]
  );

  // Handle analyze button
  const handleAnalyze = useCallback(async () => {
    if (!selectedSlide) return;

    const startTime = Date.now();
    const result = await analyze({
      slideId: selectedSlide.id,
      patchBudget: 8000,
      magnification: 20,
    });

    // Record stats for the quick stats panel
    if (result) {
      const processingTime = result.processingTimeMs || (Date.now() - startTime);
      recordAnalysis(
        selectedSlide.id,
        result.prediction.label,
        result.prediction.confidence,
        processingTime
      );
    }
  }, [selectedSlide, analyze]);

  // Handle patch click - navigate viewer
  const handlePatchClick = useCallback((coords: PatchCoordinates) => {
    // The WSIViewer will handle navigation internally via state/props
    setSelectedPatchId(`${coords.x}_${coords.y}`);
  }, []);

  // Handle patch zoom - open modal with enlarged view
  const handlePatchZoom = useCallback((patch: EvidencePatch) => {
    setZoomedPatch(patch);
    setZoomModalOpen(true);
  }, []);

  // Handle patch modal navigation
  const handlePatchModalNavigate = useCallback(
    (direction: "prev" | "next") => {
      if (!zoomedPatch || !analysisResult) return;

      const patches = analysisResult.evidencePatches;
      const currentIndex = patches.findIndex((p) => p.id === zoomedPatch.id);

      if (direction === "prev" && currentIndex > 0) {
        setZoomedPatch(patches[currentIndex - 1]);
      } else if (direction === "next" && currentIndex < patches.length - 1) {
        setZoomedPatch(patches[currentIndex + 1]);
      }
    },
    [zoomedPatch, analysisResult]
  );

  // Handle similar case click
  const handleCaseClick = useCallback((caseId: string) => {
    // Open similar case in new tab (placeholder - would navigate to case detail view)
    window.open(`/case/${caseId}`, "_blank");
  }, []);

  // Handle report generation
  const handleGenerateReport = useCallback(async () => {
    if (!selectedSlide || !analysisResult) return;

    await generateSlideReport({
      slideId: selectedSlide.id,
      evidencePatchIds: analysisResult.evidencePatches.map((p) => p.id),
      includeDetails: true,
    });
  }, [selectedSlide, analysisResult, generateSlideReport]);

  // Handle PDF export
  const handleExportPdf = useCallback(async () => {
    if (!selectedSlide) return;
    try {
      const blob = await exportReportPdf(selectedSlide.id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `atlas-report-${selectedSlide.id}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("PDF export failed:", err);
    }
  }, [selectedSlide]);

  // Handle JSON export
  const handleExportJson = useCallback(async () => {
    if (!selectedSlide || !report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `atlas-report-${selectedSlide.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [selectedSlide, report]);

  // Get DZI and heatmap URLs
  const dziUrl = selectedSlide ? getDziUrl(selectedSlide.id) : undefined;
  const heatmapData = analysisResult?.heatmap;

  return (
    <div className="flex flex-col h-screen bg-surface-secondary">
      {/* Header */}
      <Header
        isConnected={isConnected}
        isProcessing={isAnalyzing || isGeneratingReport}
        version="0.1.0"
        institutionName="Enso Labs"
        userName="Clinician"
      />

      {/* Main Content */}
      <main className="flex-1 flex overflow-hidden">
        {/* Left Sidebar - Slide Selection */}
        <aside className="w-80 border-r border-surface-border bg-white p-4 overflow-y-auto shrink-0 space-y-4">
          <SlideSelector
            selectedSlideId={selectedSlide?.id ?? null}
            onSlideSelect={handleSlideSelect}
            onAnalyze={handleAnalyze}
            isAnalyzing={isAnalyzing}
          />

          {/* Error Display */}
          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-700">{error}</p>
              <button
                onClick={clearError}
                className="mt-2 text-xs text-red-600 underline"
              >
                Dismiss
              </button>
            </div>
          )}

          {/* Quick Stats Dashboard */}
          <QuickStatsPanel />

          {/* Case Notes */}
          <CaseNotesPanel slideId={selectedSlide?.id ?? null} />
        </aside>

        {/* Center - WSI Viewer */}
        <section className="flex-1 p-4 overflow-hidden">
          {selectedSlide && dziUrl ? (
            <WSIViewer
              slideId={selectedSlide.id}
              dziUrl={dziUrl}
              heatmap={heatmapData}
              mpp={selectedSlide.mpp}
              onRegionClick={handlePatchClick}
              className="h-full"
            />
          ) : (
            <div className="h-full flex items-center justify-center bg-gray-100 rounded-lg border-2 border-dashed border-gray-300">
              <div className="text-center">
                <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg
                    className="w-8 h-8 text-gray-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                    />
                  </svg>
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-1">
                  No Slide Selected
                </h3>
                <p className="text-sm text-gray-500 max-w-sm">
                  Select a whole-slide image from the sidebar to begin analysis.
                  The slide will be displayed here with interactive viewing and
                  heatmap overlay capabilities.
                </p>
              </div>
            </div>
          )}
        </section>

        {/* Right Sidebar - Results */}
        <aside className="w-96 border-l border-surface-border bg-white p-4 overflow-y-auto shrink-0 space-y-4">
          {/* Prediction Results */}
          <PredictionPanel
            prediction={analysisResult?.prediction ?? null}
            isLoading={isAnalyzing}
            processingTime={analysisResult?.processingTimeMs}
          />

          {/* Evidence Patches */}
          <EvidencePanel
            patches={analysisResult?.evidencePatches ?? []}
            isLoading={isAnalyzing}
            onPatchClick={handlePatchClick}
            onPatchZoom={handlePatchZoom}
            selectedPatchId={selectedPatchId}
          />

          {/* Semantic Search */}
          <SemanticSearchPanel
            slideId={selectedSlide?.id ?? null}
            isAnalyzed={!!analysisResult}
            onSearch={handleSemanticSearch}
            results={semanticResults}
            isSearching={isSearching}
            error={searchError}
            onPatchClick={handlePatchClick}
            selectedPatchId={selectedPatchId}
          />

          {/* Similar Cases */}
          <SimilarCasesPanel
            cases={analysisResult?.similarCases ?? []}
            isLoading={isAnalyzing}
            onCaseClick={handleCaseClick}
          />

          {/* Clinical Report */}
          <ReportPanel
            report={report}
            isLoading={isGeneratingReport}
            onGenerateReport={
              analysisResult && !report ? handleGenerateReport : undefined
            }
            onExportPdf={report ? handleExportPdf : undefined}
            onExportJson={report ? handleExportJson : undefined}
          />
        </aside>
      </main>

      {/* Footer */}
      <Footer version="0.1.0" />

      {/* Patch Zoom Modal */}
      <PatchZoomModal
        isOpen={zoomModalOpen}
        onClose={() => setZoomModalOpen(false)}
        patch={zoomedPatch}
        allPatches={analysisResult?.evidencePatches ?? []}
        onNavigate={handlePatchModalNavigate}
        slideId={selectedSlide?.id}
      />
    </div>
  );
}
