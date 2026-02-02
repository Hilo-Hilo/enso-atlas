"use client";

import React from "react";
import { Shield, Lock, FileWarning, ExternalLink, HelpCircle } from "lucide-react";

interface FooterProps {
  version?: string;
  buildDate?: string;
}

export function Footer({ version = "0.1.0", buildDate }: FooterProps) {
  // Note: currentYear removed - was unused and would cause hydration mismatch
  // If needed in future, use useEffect to set it client-side

  return (
    <footer className="h-12 bg-gradient-to-r from-surface-secondary via-white to-surface-secondary border-t border-surface-border px-6 flex items-center justify-between shrink-0">
      {/* Left: Regulatory Disclaimers */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2 px-3 py-1 bg-amber-50 rounded-full border border-amber-200">
          <FileWarning className="h-3.5 w-3.5 text-amber-600" />
          <span className="text-xs font-semibold text-amber-700 hidden md:inline">
            For Research Use Only
          </span>
          <span className="text-xs font-semibold text-amber-700 md:hidden">
            Research Only
          </span>
        </div>
        <div className="h-4 w-px bg-gray-200 hidden lg:block" />
        <div className="hidden lg:flex items-center gap-1.5 text-gray-500">
          <Lock className="h-3.5 w-3.5 text-clinical-600" />
          <span className="text-xs font-medium">Local Processing</span>
        </div>
      </div>

      {/* Center: Powered By */}
      <div className="hidden sm:flex items-center gap-2 px-3 py-1 bg-gray-50 rounded-full border border-gray-200">
        <span className="text-xs text-gray-400">Powered by</span>
        <span className="text-xs font-semibold bg-gradient-to-r from-clinical-600 to-violet-600 bg-clip-text text-transparent">
          Path Foundation + MedGemma
        </span>
      </div>

      {/* Right: Version, Help, and Security */}
      <div className="flex items-center gap-3 text-xs text-gray-500">
        <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 bg-green-50 rounded-full border border-green-200">
          <Shield className="h-3.5 w-3.5 text-status-positive" />
          <span className="text-green-700 font-medium">HIPAA Ready</span>
        </div>
        <div className="h-4 w-px bg-gray-200 hidden sm:block" />
        <a
          href="https://github.com/Hilo-Hilo/med-gemma-hackathon#readme"
          target="_blank"
          rel="noopener noreferrer"
          className="hidden sm:flex items-center gap-1.5 hover:text-clinical-600 transition-colors duration-150 group"
        >
          <HelpCircle className="h-3.5 w-3.5" />
          <span className="font-medium">Docs</span>
          <ExternalLink className="h-3 w-3 opacity-50 group-hover:opacity-100 transition-opacity" />
        </a>
        <div className="h-4 w-px bg-gray-200 hidden sm:block" />
        <span className="font-mono text-gray-400 bg-gray-100 px-2 py-0.5 rounded">
          v{version}
        </span>
      </div>
    </footer>
  );
}
