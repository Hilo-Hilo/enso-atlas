"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className }: SkeletonProps) {
  return (
    <div
      className={cn("animate-pulse rounded bg-gray-200", className)}
    />
  );
}

// Skeleton for text lines
interface SkeletonTextProps {
  lines?: number;
  className?: string;
}

export function SkeletonText({ lines = 3, className }: SkeletonTextProps) {
  return (
    <div className={cn("space-y-2", className)}>
      {[...Array(lines)].map((_, i) => (
        <Skeleton
          key={i}
          className={cn(
            "h-4",
            i === lines - 1 ? "w-3/4" : "w-full"
          )}
        />
      ))}
    </div>
  );
}

// Skeleton for prediction panel
export function SkeletonPrediction() {
  return (
    <div className="space-y-4 animate-pulse">
      {/* Main prediction box */}
      <div className="p-4 rounded-xl border-2 border-gray-200 bg-gray-50">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Skeleton className="w-6 h-6 rounded-full" />
            <Skeleton className="h-6 w-24" />
          </div>
          <Skeleton className="h-5 w-20 rounded-full" />
        </div>
        <div className="flex items-baseline gap-1">
          <Skeleton className="h-10 w-16" />
          <Skeleton className="h-6 w-4" />
          <Skeleton className="h-4 w-16 ml-2" />
        </div>
      </div>

      {/* Probability bar */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-4 w-12" />
        </div>
        <Skeleton className="h-3 w-full rounded-full" />
      </div>

      {/* Confidence */}
      <div className="p-3 bg-gray-50 rounded-lg">
        <div className="flex justify-between mb-2">
          <Skeleton className="h-3 w-24" />
          <Skeleton className="h-4 w-10" />
        </div>
        <Skeleton className="h-2 w-full rounded-full" />
      </div>
    </div>
  );
}

// Skeleton for evidence patches grid
export function SkeletonEvidenceGrid() {
  return (
    <div className="space-y-3 animate-pulse">
      <div className="grid grid-cols-3 gap-2">
        {[...Array(6)].map((_, i) => (
          <div key={i} className="relative aspect-square">
            <Skeleton className="absolute inset-0 rounded-lg" />
            <div className="absolute top-1 left-1">
              <Skeleton className="h-4 w-6 rounded" />
            </div>
            <div className="absolute top-1 right-1">
              <Skeleton className="h-6 w-6 rounded-full" />
            </div>
          </div>
        ))}
      </div>
      <Skeleton className="h-4 w-40 mx-auto" />
    </div>
  );
}

// Skeleton for similar cases list
export function SkeletonSimilarCases() {
  return (
    <div className="space-y-3 animate-pulse">
      {[...Array(3)].map((_, i) => (
        <div
          key={i}
          className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border border-gray-100"
        >
          <Skeleton className="w-12 h-12 rounded-lg shrink-0" />
          <div className="flex-1 space-y-2">
            <div className="flex items-center justify-between">
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-5 w-16 rounded-full" />
            </div>
            <div className="flex items-center gap-2">
              <Skeleton className="h-3 w-16" />
              <Skeleton className="h-1.5 w-20 rounded-full" />
              <Skeleton className="h-3 w-8" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// Skeleton for report panel
export function SkeletonReport() {
  return (
    <div className="space-y-4 animate-pulse">
      {/* Summary section */}
      <div className="border rounded-lg overflow-hidden">
        <div className="p-3 bg-gray-50 border-b flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Skeleton className="h-4 w-4 rounded" />
            <Skeleton className="h-4 w-28" />
          </div>
          <Skeleton className="h-4 w-4 rounded" />
        </div>
        <div className="p-4">
          <SkeletonText lines={4} />
        </div>
      </div>

      {/* Evidence section */}
      <div className="border rounded-lg overflow-hidden">
        <div className="p-3 bg-gray-50 border-b flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Skeleton className="h-4 w-4 rounded" />
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-5 w-16 rounded-full" />
          </div>
          <Skeleton className="h-4 w-4 rounded" />
        </div>
      </div>
    </div>
  );
}
