// Enso Atlas - Analysis Hook
// State management for slide analysis workflow

import { useState, useCallback, useRef } from "react";
import type {
  AnalysisResponse,
  AnalysisRequest,
  StructuredReport,
  ReportRequest,
} from "@/types";
import { analyzeSlide, generateReport } from "@/lib/api";

// Analysis progress steps
export const ANALYSIS_STEPS = [
  { id: "embeddings", label: "Loading slide embeddings...", description: "Retrieving pre-computed feature vectors" },
  { id: "prediction", label: "Running CLAM prediction...", description: "Multiple instance learning inference" },
  { id: "similar", label: "Finding similar cases...", description: "FAISS vector similarity search" },
  { id: "evidence", label: "Generating evidence...", description: "Extracting top attention regions" },
] as const;

export type AnalysisStepId = typeof ANALYSIS_STEPS[number]["id"];

interface UseAnalysisState {
  isAnalyzing: boolean;
  isGeneratingReport: boolean;
  analysisResult: AnalysisResponse | null;
  report: StructuredReport | null;
  error: string | null;
  // Progress tracking
  analysisStep: number; // -1 = not started, 0-3 = step index
  analysisStepId: AnalysisStepId | null;
}

interface UseAnalysisReturn extends UseAnalysisState {
  analyze: (request: AnalysisRequest) => Promise<AnalysisResponse | null>;
  generateSlideReport: (request: ReportRequest) => Promise<void>;
  clearResults: () => void;
  clearError: () => void;
  retryAnalysis: () => Promise<AnalysisResponse | null>;
  retryReport: () => Promise<void>;
}

export function useAnalysis(): UseAnalysisReturn {
  const [state, setState] = useState<UseAnalysisState>({
    isAnalyzing: false,
    isGeneratingReport: false,
    analysisResult: null,
    report: null,
    error: null,
    analysisStep: -1,
    analysisStepId: null,
  });

  // Store last requests for retry functionality
  const lastAnalysisRequest = useRef<AnalysisRequest | null>(null);
  const lastReportRequest = useRef<ReportRequest | null>(null);

  // Simulate progress steps during analysis
  // In a real implementation, the backend would send progress updates via SSE/WebSocket
  const simulateProgress = useCallback((onComplete: () => void) => {
    const stepDurations = [400, 800, 600, 500]; // ms per step
    let currentStep = 0;

    const advanceStep = () => {
      if (currentStep < ANALYSIS_STEPS.length) {
        setState((prev) => ({
          ...prev,
          analysisStep: currentStep,
          analysisStepId: ANALYSIS_STEPS[currentStep].id,
        }));
        currentStep++;
        setTimeout(advanceStep, stepDurations[currentStep - 1] || 500);
      } else {
        onComplete();
      }
    };

    advanceStep();
  }, []);

  const analyze = useCallback(async (request: AnalysisRequest): Promise<AnalysisResponse | null> => {
    lastAnalysisRequest.current = request;

    setState((prev) => ({
      ...prev,
      isAnalyzing: true,
      error: null,
      analysisStep: 0,
      analysisStepId: ANALYSIS_STEPS[0].id,
    }));

    // Start progress simulation
    let progressComplete = false;
    const progressPromise = new Promise<void>((resolve) => {
      simulateProgress(() => {
        progressComplete = true;
        resolve();
      });
    });

    try {
      // Run actual API call in parallel with progress simulation
      const [result] = await Promise.all([
        analyzeSlide(request),
        progressPromise,
      ]);

      // If API finished before progress, wait for progress to complete
      if (!progressComplete) {
        await progressPromise;
      }

      setState((prev) => ({
        ...prev,
        isAnalyzing: false,
        analysisResult: result,
        report: result.report || null,
        analysisStep: ANALYSIS_STEPS.length, // Completed
        analysisStepId: null,
      }));
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Analysis failed";
      setState((prev) => ({
        ...prev,
        isAnalyzing: false,
        error: message,
        analysisStep: -1,
        analysisStepId: null,
      }));
      return null;
    }
  }, [simulateProgress]);

  const retryAnalysis = useCallback(async (): Promise<AnalysisResponse | null> => {
    if (lastAnalysisRequest.current) {
      return analyze(lastAnalysisRequest.current);
    }
    return null;
  }, [analyze]);

  const generateSlideReport = useCallback(async (request: ReportRequest) => {
    lastReportRequest.current = request;

    setState((prev) => ({
      ...prev,
      isGeneratingReport: true,
      error: null,
    }));

    try {
      const report = await generateReport(request);
      setState((prev) => ({
        ...prev,
        isGeneratingReport: false,
        report,
      }));
    } catch (err) {
      const message = err instanceof Error ? err.message : "Report generation failed";
      setState((prev) => ({
        ...prev,
        isGeneratingReport: false,
        error: message,
      }));
    }
  }, []);

  const retryReport = useCallback(async () => {
    if (lastReportRequest.current) {
      return generateSlideReport(lastReportRequest.current);
    }
  }, [generateSlideReport]);

  const clearResults = useCallback(() => {
    setState({
      isAnalyzing: false,
      isGeneratingReport: false,
      analysisResult: null,
      report: null,
      error: null,
      analysisStep: -1,
      analysisStepId: null,
    });
    lastAnalysisRequest.current = null;
    lastReportRequest.current = null;
  }, []);

  const clearError = useCallback(() => {
    setState((prev) => ({
      ...prev,
      error: null,
    }));
  }, []);

  return {
    ...state,
    analyze,
    generateSlideReport,
    clearResults,
    clearError,
    retryAnalysis,
    retryReport,
  };
}
