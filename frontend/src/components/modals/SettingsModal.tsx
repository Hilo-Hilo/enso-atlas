"use client";

import React, { useEffect, useCallback, useState } from "react";
import { Button } from "@/components/ui/Button";
import { Toggle } from "@/components/ui/Toggle";
import {
  X,
  Settings,
  Moon,
  Sun,
  Monitor,
  Server,
  Eye,
  RefreshCw,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

type ThemeMode = "light" | "dark" | "system";

interface DisplaySettings {
  showHeatmapByDefault: boolean;
  showConfidenceScores: boolean;
  compactMode: boolean;
  animationsEnabled: boolean;
}

interface ApiSettings {
  endpoint: string;
  timeout: number;
}

const DEFAULT_DISPLAY_SETTINGS: DisplaySettings = {
  showHeatmapByDefault: true,
  showConfidenceScores: true,
  compactMode: false,
  animationsEnabled: true,
};

const DEFAULT_API_SETTINGS: ApiSettings = {
  endpoint: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  timeout: 30000,
};

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const [activeTab, setActiveTab] = useState<"appearance" | "display" | "api">("appearance");
  const [theme, setTheme] = useState<ThemeMode>("system");
  const [displaySettings, setDisplaySettings] = useState<DisplaySettings>(DEFAULT_DISPLAY_SETTINGS);
  const [apiSettings, setApiSettings] = useState<ApiSettings>(DEFAULT_API_SETTINGS);
  const [isDirty, setIsDirty] = useState(false);

  // Load settings from localStorage on mount
  useEffect(() => {
    if (typeof window === "undefined") return;
    
    const savedTheme = localStorage.getItem("atlas-theme") as ThemeMode | null;
    if (savedTheme) setTheme(savedTheme);

    const savedDisplay = localStorage.getItem("atlas-display-settings");
    if (savedDisplay) {
      try {
        setDisplaySettings(JSON.parse(savedDisplay));
      } catch {
        // Ignore parse errors
      }
    }

    const savedApi = localStorage.getItem("atlas-api-settings");
    if (savedApi) {
      try {
        setApiSettings(JSON.parse(savedApi));
      } catch {
        // Ignore parse errors
      }
    }
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
    setIsDirty(true);
    if (typeof window !== "undefined") {
      localStorage.setItem("atlas-theme", newTheme);
      // Apply theme to document
      const root = document.documentElement;
      if (newTheme === "dark") {
        root.classList.add("dark");
      } else if (newTheme === "light") {
        root.classList.remove("dark");
      } else {
        // System preference
        const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
        if (prefersDark) {
          root.classList.add("dark");
        } else {
          root.classList.remove("dark");
        }
      }
    }
  };

  const handleDisplayChange = (key: keyof DisplaySettings, value: boolean) => {
    const newSettings = { ...displaySettings, [key]: value };
    setDisplaySettings(newSettings);
    setIsDirty(true);
    if (typeof window !== "undefined") {
      localStorage.setItem("atlas-display-settings", JSON.stringify(newSettings));
    }
  };

  const handleApiChange = (key: keyof ApiSettings, value: string | number) => {
    const newSettings = { ...apiSettings, [key]: value };
    setApiSettings(newSettings);
    setIsDirty(true);
    if (typeof window !== "undefined") {
      localStorage.setItem("atlas-api-settings", JSON.stringify(newSettings));
    }
  };

  const handleReset = () => {
    setTheme("system");
    setDisplaySettings(DEFAULT_DISPLAY_SETTINGS);
    setApiSettings(DEFAULT_API_SETTINGS);
    if (typeof window !== "undefined") {
      localStorage.removeItem("atlas-theme");
      localStorage.removeItem("atlas-display-settings");
      localStorage.removeItem("atlas-api-settings");
      document.documentElement.classList.remove("dark");
    }
    setIsDirty(false);
  };

  if (!isOpen) return null;

  const tabs = [
    { id: "appearance" as const, label: "Appearance", icon: Sun },
    { id: "display" as const, label: "Display", icon: Eye },
    { id: "api" as const, label: "API", icon: Server },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm animate-fade-in"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative z-10 w-full max-w-lg mx-4 bg-white rounded-xl shadow-2xl overflow-hidden animate-scale-in">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-gray-50">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-clinical-600 flex items-center justify-center">
              <Settings className="h-5 w-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">Settings</h2>
              <p className="text-sm text-gray-500">Configure your preferences</p>
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose} className="p-2">
            <X className="h-5 w-5" />
          </Button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-200">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={cn(
                "flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-colors",
                activeTab === tab.id
                  ? "text-clinical-600 border-b-2 border-clinical-600 bg-clinical-50/50"
                  : "text-gray-500 hover:text-gray-700 hover:bg-gray-50"
              )}
            >
              <tab.icon className="h-4 w-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="p-6 max-h-[60vh] overflow-y-auto">
          {activeTab === "appearance" && (
            <div className="space-y-6">
              <div>
                <h3 className="text-sm font-semibold text-gray-900 mb-3">Theme</h3>
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
                          ? "border-clinical-500 bg-clinical-50 text-clinical-700"
                          : "border-gray-200 hover:border-gray-300 text-gray-600"
                      )}
                    >
                      <option.icon className="h-6 w-6" />
                      <span className="text-sm font-medium">{option.label}</span>
                    </button>
                  ))}
                </div>
              </div>
              <p className="text-xs text-gray-500">
                Note: Dark mode support is limited in this version.
              </p>
            </div>
          )}

          {activeTab === "display" && (
            <div className="space-y-4">
              <div className="flex items-center justify-between py-2">
                <div>
                  <p className="text-sm font-medium text-gray-900">Show heatmap by default</p>
                  <p className="text-xs text-gray-500">Display probability heatmap overlay on slides</p>
                </div>
                <Toggle
                  checked={displaySettings.showHeatmapByDefault}
                  onChange={(v) => handleDisplayChange("showHeatmapByDefault", v)}
                />
              </div>
              <div className="flex items-center justify-between py-2">
                <div>
                  <p className="text-sm font-medium text-gray-900">Show confidence scores</p>
                  <p className="text-xs text-gray-500">Display confidence percentages on predictions</p>
                </div>
                <Toggle
                  checked={displaySettings.showConfidenceScores}
                  onChange={(v) => handleDisplayChange("showConfidenceScores", v)}
                />
              </div>
              <div className="flex items-center justify-between py-2">
                <div>
                  <p className="text-sm font-medium text-gray-900">Compact mode</p>
                  <p className="text-xs text-gray-500">Reduce spacing and panel sizes</p>
                </div>
                <Toggle
                  checked={displaySettings.compactMode}
                  onChange={(v) => handleDisplayChange("compactMode", v)}
                />
              </div>
              <div className="flex items-center justify-between py-2">
                <div>
                  <p className="text-sm font-medium text-gray-900">Enable animations</p>
                  <p className="text-xs text-gray-500">Smooth transitions and loading effects</p>
                </div>
                <Toggle
                  checked={displaySettings.animationsEnabled}
                  onChange={(v) => handleDisplayChange("animationsEnabled", v)}
                />
              </div>
            </div>
          )}

          {activeTab === "api" && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-1">
                  Backend API Endpoint
                </label>
                <input
                  type="text"
                  value={apiSettings.endpoint}
                  onChange={(e) => handleApiChange("endpoint", e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-clinical-500 focus:border-transparent"
                  placeholder="http://localhost:8000"
                />
                <p className="mt-1 text-xs text-gray-500">
                  URL of the MedGemma backend service
                </p>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-1">
                  Request Timeout (ms)
                </label>
                <input
                  type="number"
                  value={apiSettings.timeout}
                  onChange={(e) => handleApiChange("timeout", parseInt(e.target.value) || 30000)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-clinical-500 focus:border-transparent"
                  placeholder="30000"
                  min={5000}
                  max={120000}
                  step={1000}
                />
                <p className="mt-1 text-xs text-gray-500">
                  Maximum time to wait for API responses (5000-120000ms)
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-gray-200 bg-gray-50">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            className="text-gray-500 hover:text-gray-700"
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
