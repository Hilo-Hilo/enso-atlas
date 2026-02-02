"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface PredictionGaugeProps {
  value: number; // 0-1 probability
  size?: "sm" | "md" | "lg";
  showLabel?: boolean;
  className?: string;
}

export function PredictionGauge({
  value,
  size = "md",
  showLabel = true,
  className,
}: PredictionGaugeProps) {
  // Clamp value between 0 and 1
  const clampedValue = Math.max(0, Math.min(1, value));
  const percentage = Math.round(clampedValue * 100);
  const isResponder = clampedValue >= 0.5;
  
  // SVG parameters
  const sizeMap = {
    sm: { width: 80, stroke: 6, fontSize: "text-lg" },
    md: { width: 120, stroke: 8, fontSize: "text-2xl" },
    lg: { width: 160, stroke: 10, fontSize: "text-3xl" },
  };
  
  const { width, stroke, fontSize } = sizeMap[size];
  const radius = (width - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  
  // The gauge spans from 180° to 360° (bottom half) for non-responder
  // and from 0° to 180° (top half) for responder
  // We'll use a full circle but color-code it
  const offset = circumference * (1 - clampedValue);
  
  // Color based on prediction
  const primaryColor = isResponder ? "#059669" : "#dc2626"; // green-600 / red-600
  const secondaryColor = isResponder ? "#d1fae5" : "#fee2e2"; // green-100 / red-100
  
  return (
    <div className={cn("flex flex-col items-center gap-2", className)}>
      <div className="relative" style={{ width, height: width }}>
        {/* Background circle */}
        <svg
          className="transform -rotate-90"
          width={width}
          height={width}
          viewBox={`0 0 ${width} ${width}`}
        >
          {/* Gradient definitions */}
          <defs>
            <linearGradient id="gauge-gradient-green" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#10b981" />
              <stop offset="100%" stopColor="#059669" />
            </linearGradient>
            <linearGradient id="gauge-gradient-red" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#ef4444" />
              <stop offset="100%" stopColor="#dc2626" />
            </linearGradient>
            <filter id="gauge-glow">
              <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>

          {/* Background track */}
          <circle
            cx={width / 2}
            cy={width / 2}
            r={radius}
            fill="none"
            stroke="#f1f5f9"
            strokeWidth={stroke}
          />
          
          {/* Threshold indicator at 50% */}
          <circle
            cx={width / 2}
            cy={width / 2}
            r={radius}
            fill="none"
            stroke="#cbd5e1"
            strokeWidth={stroke + 2}
            strokeDasharray={`${circumference * 0.015} ${circumference * 0.985}`}
            strokeDashoffset={-circumference * 0.5 + circumference * 0.0075}
            strokeLinecap="round"
          />
          
          {/* Value arc with gradient and glow */}
          <circle
            cx={width / 2}
            cy={width / 2}
            r={radius}
            fill="none"
            stroke={isResponder ? "url(#gauge-gradient-green)" : "url(#gauge-gradient-red)"}
            strokeWidth={stroke}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            filter="url(#gauge-glow)"
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        
        {/* Center value with animation */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span 
            className={cn("font-bold font-mono tabular-nums transition-all duration-700", fontSize)} 
            style={{ color: primaryColor }}
          >
            {percentage}%
          </span>
          {showLabel && (
            <span className="text-xs text-gray-400 mt-0.5 font-medium uppercase tracking-wide">score</span>
          )}
        </div>
      </div>
    </div>
  );
}

interface ConfidenceGaugeProps {
  value: number; // 0-1 confidence
  level: "high" | "moderate" | "low";
  className?: string;
}

export function ConfidenceGauge({
  value,
  level,
  className,
}: ConfidenceGaugeProps) {
  const clampedValue = Math.max(0, Math.min(1, value));
  const percentage = Math.round(clampedValue * 100);
  
  const colorMap = {
    high: { primary: "#059669", bg: "#d1fae5" },
    moderate: { primary: "#d97706", bg: "#fef3c7" },
    low: { primary: "#6b7280", bg: "#f3f4f6" },
  };
  
  const { primary, bg } = colorMap[level];
  
  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-gray-600 uppercase tracking-wide">
          Model Confidence
        </span>
        <span className="text-sm font-mono font-semibold" style={{ color: primary }}>
          {percentage}%
        </span>
      </div>
      
      {/* Segmented confidence bar */}
      <div className="flex gap-1 h-2">
        {[...Array(10)].map((_, i) => {
          const segmentThreshold = (i + 1) * 10;
          const isFilled = percentage >= segmentThreshold;
          
          return (
            <div
              key={i}
              className={cn(
                "flex-1 rounded-sm transition-all duration-300",
                isFilled ? "" : "bg-gray-200"
              )}
              style={isFilled ? { backgroundColor: primary } : undefined}
            />
          );
        })}
      </div>
      
      {/* Labels */}
      <div className="flex justify-between text-2xs text-gray-400">
        <span>Low</span>
        <span>Moderate</span>
        <span>High</span>
      </div>
    </div>
  );
}

interface UncertaintyDisplayProps {
  uncertainty: number; // Standard deviation
  confidenceInterval: [number, number];
  samples?: number[];
  className?: string;
}

export function UncertaintyDisplay({
  uncertainty,
  confidenceInterval,
  samples,
  className,
}: UncertaintyDisplayProps) {
  const [ciLower, ciUpper] = confidenceInterval;
  const ciWidth = (ciUpper - ciLower) * 100;
  
  // Determine uncertainty level
  const uncertaintyLevel = uncertainty < 0.1 ? "low" : uncertainty < 0.2 ? "moderate" : "high";
  
  const colorMap = {
    low: { primary: "#059669", bg: "#d1fae5", text: "Low uncertainty" },
    moderate: { primary: "#d97706", bg: "#fef3c7", text: "Moderate uncertainty" },
    high: { primary: "#dc2626", bg: "#fee2e2", text: "High uncertainty" },
  };
  
  const { primary, bg, text } = colorMap[uncertaintyLevel];
  
  return (
    <div className={cn("p-3 rounded-lg border", className)} style={{ backgroundColor: bg, borderColor: primary + "40" }}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold" style={{ color: primary }}>
          {text}
        </span>
        <span className="text-xs font-mono text-gray-600">
          σ = {(uncertainty * 100).toFixed(1)}%
        </span>
      </div>
      
      {/* Confidence interval visualization */}
      <div className="space-y-1.5">
        <div className="text-2xs text-gray-500">95% Confidence Interval</div>
        <div className="relative h-3 bg-white rounded-full overflow-hidden">
          {/* Full range background */}
          <div className="absolute inset-0 bg-gray-200" />
          
          {/* CI range */}
          <div
            className="absolute h-full rounded-full transition-all duration-500"
            style={{
              left: `${ciLower * 100}%`,
              width: `${ciWidth}%`,
              backgroundColor: primary,
            }}
          />
          
          {/* 50% threshold marker */}
          <div className="absolute top-0 bottom-0 w-0.5 bg-gray-400" style={{ left: "50%" }} />
        </div>
        
        {/* CI bounds labels */}
        <div className="flex justify-between text-2xs">
          <span className="text-gray-500">{(ciLower * 100).toFixed(0)}%</span>
          <span className="text-gray-400">50%</span>
          <span className="text-gray-500">{(ciUpper * 100).toFixed(0)}%</span>
        </div>
      </div>
      
      {/* Mini histogram of MC samples */}
      {samples && samples.length > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <div className="text-2xs text-gray-500 mb-1.5">MC Dropout Distribution ({samples.length} samples)</div>
          <div className="flex items-end h-8 gap-px">
            {/* Create histogram bins */}
            {(() => {
              const bins = 20;
              const histogram = new Array(bins).fill(0);
              samples.forEach(s => {
                const binIdx = Math.min(Math.floor(s * bins), bins - 1);
                histogram[binIdx]++;
              });
              const maxCount = Math.max(...histogram);
              
              return histogram.map((count, i) => (
                <div
                  key={i}
                  className="flex-1 rounded-t-sm transition-all"
                  style={{
                    height: `${(count / maxCount) * 100}%`,
                    backgroundColor: i < bins / 2 ? "#fee2e2" : "#d1fae5",
                    minHeight: count > 0 ? "2px" : "0",
                  }}
                />
              ));
            })()}
          </div>
        </div>
      )}
    </div>
  );
}
