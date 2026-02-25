"use client";

import React, { useEffect, useCallback, useState } from "react";
import { Button } from "@/components/ui/Button";
import {
  X,
  Settings,
  Moon,
  Sun,
  Monitor,
  RefreshCw,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

type ThemeMode = "light" | "dark" | "system";

/**
 * Applies the theme class to the document and persists to localStorage
 */
function applyTheme(theme: ThemeMode, save = true) {
  if (typeof window === "undefined") return;
  
  const root = document.documentElement;
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  
  if (theme === "dark" || (theme === "system" && prefersDark)) {
    root.classList.add("dark");
  } else {
    root.classList.remove("dark");
  }
  
  if (save) {
    localStorage.setItem("atlas-theme", theme);
  }
}

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const [theme, setTheme] = useState<ThemeMode>("system");

  // Load settings from localStorage on mount and apply theme
  useEffect(() => {
    if (typeof window === "undefined") return;

    const savedTheme = localStorage.getItem("atlas-theme") as ThemeMode | null;
    const initialTheme = savedTheme || "system";
    setTheme(initialTheme);
    applyTheme(initialTheme, false); // Apply but don't re-save
    
    // Listen for system preference changes when in system mode
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = () => {
      const currentTheme = localStorage.getItem("atlas-theme") as ThemeMode | null;
      if (currentTheme === "system" || !currentTheme) {
        applyTheme("system", false);
      }
    };
    
    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, []);

  // Handle escape key
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    },
    [isOpen, onClose]
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  // Prevent scroll when modal is open
  useEffect(() => {
    if (typeof window === "undefined") return;
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [isOpen]);

  const handleThemeChange = (newTheme: ThemeMode) => {
    setTheme(newTheme);
    applyTheme(newTheme);
  };

  const handleReset = () => {
    setTheme("system");
    applyTheme("system");
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fade-in"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative z-10 w-full max-w-lg mx-4 bg-white dark:bg-navy-800 rounded-xl shadow-2xl overflow-hidden animate-scale-in">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-navy-600 bg-gray-50 dark:bg-navy-900">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-clinical-600 flex items-center justify-center">
              <Settings className="h-5 w-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Settings</h2>
              <p className="text-sm text-gray-500 dark:text-gray-400">Configure your preferences</p>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose} className="p-2 dark:text-gray-300 dark:hover:bg-navy-700">
            <X className="h-5 w-5" />
          </Button>
        </div>

        {/* Content */}
        <div className="p-6 max-h-[60vh] overflow-y-auto">
          <div className="space-y-6">
            <div>
              <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">Theme</h3>
              <div className="grid grid-cols-3 gap-3">
                {[
                  { id: "light" as const, label: "Light", icon: Sun },
                  { id: "dark" as const, label: "Dark", icon: Moon },
                  { id: "system" as const, label: "System", icon: Monitor },
                ].map((option) => (
                  <button
                    key={option.id}
                    onClick={() => handleThemeChange(option.id)}
                    className={cn(
                      "flex flex-col items-center gap-2 p-4 rounded-lg border-2 transition-all",
                      theme === option.id
                        ? "border-clinical-500 bg-clinical-50 dark:bg-clinical-900/30 text-clinical-700 dark:text-clinical-300"
                        : "border-gray-200 dark:border-navy-600 hover:border-gray-300 dark:hover:border-navy-500 text-gray-600 dark:text-gray-300"
                    )}
                  >
                    <option.icon className="h-6 w-6" />
                    <span className="text-sm font-medium">{option.label}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-gray-200 dark:border-navy-600 bg-gray-50 dark:bg-navy-900">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Reset to Defaults
          </Button>
          <Button variant="primary" size="sm" onClick={onClose}>
            Done
          </Button>
        </div>
      </div>
    </div>
  );
}
