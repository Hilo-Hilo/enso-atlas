"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { Check, Loader2 } from "lucide-react";

export interface ProgressStep {
  id: string;
  label: string;
  description?: string;
}

interface ProgressStepperProps {
  steps: ProgressStep[];
  currentStep: number; // 0-indexed, -1 means not started
  error?: string | null;
  className?: string;
}

export function ProgressStepper({
  steps,
  currentStep,
  error,
  className,
}: ProgressStepperProps) {
  return (
    <div className={cn("space-y-3", className)}>
      {steps.map((step, index) => {
        const isCompleted = index < currentStep;
        const isCurrent = index === currentStep;
        const isPending = index > currentStep;
        const hasError = error && isCurrent;

        return (
          <div
            key={step.id}
            className={cn(
              "flex items-start gap-3 transition-all duration-300",
              isPending && "opacity-50"
            )}
          >
            {/* Step indicator */}
            <div className="flex flex-col items-center">
              <div
                className={cn(
                  "w-7 h-7 rounded-full flex items-center justify-center transition-all duration-300 shrink-0",
                  isCompleted && "bg-green-500 text-white",
                  isCurrent && !hasError && "bg-clinical-600 text-white",
                  isCurrent && hasError && "bg-red-500 text-white",
                  isPending && "bg-gray-200 text-gray-400"
                )}
              >
                {isCompleted ? (
                  <Check className="h-4 w-4" />
                ) : isCurrent && !hasError ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <span className="text-xs font-semibold">{index + 1}</span>
                )}
              </div>
              {/* Connector line */}
              {index < steps.length - 1 && (
                <div
                  className={cn(
                    "w-0.5 h-6 mt-1 transition-all duration-300",
                    isCompleted ? "bg-green-500" : "bg-gray-200"
                  )}
                />
              )}
            </div>

            {/* Step content */}
            <div className="pt-0.5 min-w-0 flex-1">
              <p
                className={cn(
                  "text-sm font-medium transition-colors",
                  isCompleted && "text-green-700",
                  isCurrent && !hasError && "text-clinical-700",
                  isCurrent && hasError && "text-red-700",
                  isPending && "text-gray-400"
                )}
              >
                {step.label}
              </p>
              {step.description && isCurrent && (
                <p className="text-xs text-gray-500 mt-0.5">{step.description}</p>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Compact inline progress for headers/buttons
interface InlineProgressProps {
  steps: string[];
  currentStep: number;
  className?: string;
}

export function InlineProgress({
  steps,
  currentStep,
  className,
}: InlineProgressProps) {
  const safeStep = Math.min(Math.max(currentStep, 0), Math.max(steps.length - 1, 0));
  const progressPercent = Math.min(100, ((safeStep + 1) / Math.max(steps.length, 1)) * 100);
  const currentLabel = steps[safeStep] || steps[0] || "Processing...";

  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center justify-between text-xs">
        <span className="text-gray-600 font-medium">{currentLabel}</span>
        <span className="text-gray-400">
          Step {safeStep + 1} of {steps.length}
        </span>
      </div>
      <div className="h-1.5 bg-gray-200 rounded-full overflow-hidden">
        <div
          className="h-full bg-clinical-600 rounded-full transition-all duration-500 ease-out"
          style={{ width: `${progressPercent}%` }}
        />
      </div>
    </div>
  );
}
