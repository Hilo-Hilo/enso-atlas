"use client";

import React from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { InlineProgress } from "@/components/ui/ProgressStepper";
import { ANALYSIS_STEPS } from "@/hooks/useAnalysis";
import { Play } from "lucide-react";

interface AnalysisControlsProps {
  selectedSlideId: string | null;
  selectedSlideHasLevel0: boolean;
  selectedModels: string[];
  onModelsChange: (models: string[]) => void;
  resolutionLevel: number;
  onResolutionChange: (level: number) => void;
  forceReembed: boolean;
  onForceReembedChange: (force: boolean) => void;
  onAnalyze: () => void;
  onGenerateEmbeddings?: () => void;
  isAnalyzing?: boolean;
  analysisStep?: number;
  isGeneratingEmbeddings?: boolean;
  embeddingProgress?: {
    phase: string;
    progress: number;
    message: string;
  } | null;
  embeddingStatus?: {
    hasLevel0: boolean;
    hasLevel1: boolean;
  };
}

export function AnalysisControls({
  selectedSlideId,
  onAnalyze,
  isAnalyzing = false,
  analysisStep = -1,
  isGeneratingEmbeddings = false,
}: AnalysisControlsProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Play className="h-4 w-4 text-clinical-600" />
          Run Analysis
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {isAnalyzing && analysisStep >= 0 ? (
          <div className="p-3 bg-clinical-50 border border-clinical-200 rounded-md">
            <InlineProgress
              steps={ANALYSIS_STEPS.map((s) => s.label)}
              currentStep={analysisStep}
            />
          </div>
        ) : selectedSlideId ? (
          <Button
            variant="secondary"
            size="lg"
            onClick={onAnalyze}
            disabled={isAnalyzing || isGeneratingEmbeddings}
            isLoading={isAnalyzing}
            className="w-full border-sky-300 bg-sky-100 text-sky-900 hover:bg-sky-200"
            data-action="run-analysis"
          >
            {isGeneratingEmbeddings ? "Generating Embeddings..." : "Run Analysis"}
          </Button>
        ) : (
          <p className="text-xs text-gray-500">
            Select a case to enable analysis.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
