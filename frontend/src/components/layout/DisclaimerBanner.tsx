"use client";

import React, { useState } from "react";
import { AlertTriangle, X } from "lucide-react";

export function DisclaimerBanner() {
  const [dismissed, setDismissed] = useState(false);

  if (dismissed) return null;

  return (
    <div className="fixed top-3 left-1/2 -translate-x-1/2 z-[9999] w-[calc(100vw-1rem)] sm:w-[66vw] sm:max-w-4xl">
      <div className="bg-amber-50/95 backdrop-blur-sm border border-amber-200 shadow-xl rounded-2xl px-3 py-2 sm:px-4 sm:py-2.5 flex items-start gap-2.5">
        <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
        <p className="text-xs sm:text-sm text-amber-800 leading-snug flex-1">
          <span className="font-semibold">Research Demonstration Only.</span>{" "}
          Downstream models use Path Foundation embeddings and are trained on minimal data;
          they are for demo use only and not validated for clinical decision-making.
        </p>
        <button
          onClick={() => setDismissed(true)}
          className="shrink-0 p-1 rounded-full hover:bg-amber-100 text-amber-600 hover:text-amber-800 transition-colors"
          aria-label="Dismiss disclaimer"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
