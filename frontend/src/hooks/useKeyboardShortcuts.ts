"use client";

import { useEffect, useCallback, useRef } from "react";

export interface KeyboardShortcut {
  key: string;
  description: string;
  category: string;
  handler: () => void;
  modifiers?: {
    ctrl?: boolean;
    alt?: boolean;
    shift?: boolean;
    meta?: boolean;
  };
}

export interface UseKeyboardShortcutsOptions {
  enabled?: boolean;
  shortcuts: KeyboardShortcut[];
}

/**
 * Hook for managing global keyboard shortcuts.
 * Automatically disables shortcuts when focus is in input fields.
 */
export function useKeyboardShortcuts({
  enabled = true,
  shortcuts,
}: UseKeyboardShortcutsOptions) {
  const shortcutsRef = useRef(shortcuts);
  shortcutsRef.current = shortcuts;

  const handleKeyDown = useCallback(
    (event: KeyboardEvent) => {
      if (!enabled) return;

      // Ignore shortcuts when typing in input fields
      const target = event.target as HTMLElement;
      const tagName = target.tagName.toLowerCase();
      const isEditable =
        tagName === "input" ||
        tagName === "textarea" ||
        tagName === "select" ||
        target.isContentEditable;

      if (isEditable) return;

      // Find matching shortcut
      const matchingShortcut = shortcutsRef.current.find((shortcut) => {
        // Check key match (case-insensitive)
        const keyMatch =
          event.key.toLowerCase() === shortcut.key.toLowerCase() ||
          event.code.toLowerCase() === shortcut.key.toLowerCase();

        if (!keyMatch) return false;

        // Check modifiers
        const modifiers = shortcut.modifiers || {};
        const ctrlMatch = modifiers.ctrl ? event.ctrlKey || event.metaKey : !event.ctrlKey && !event.metaKey;
        const altMatch = modifiers.alt ? event.altKey : !event.altKey;
        const shiftMatch = modifiers.shift ? event.shiftKey : !event.shiftKey;

        return ctrlMatch && altMatch && shiftMatch;
      });

      if (matchingShortcut) {
        event.preventDefault();
        event.stopPropagation();
        matchingShortcut.handler();
      }
    },
    [enabled]
  );

  useEffect(() => {
    // Only add event listener in browser environment
    if (typeof window === "undefined") return;

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  return {
    shortcuts: shortcutsRef.current,
  };
}

/**
 * Get a formatted display string for a shortcut key
 */
export function formatShortcutKey(
  key: string,
  modifiers?: KeyboardShortcut["modifiers"]
): string {
  const parts: string[] = [];

  if (modifiers?.ctrl || modifiers?.meta) {
    const isMac = typeof navigator !== "undefined" && navigator.platform.includes("Mac");
    parts.push(isMac ? "Cmd" : "Ctrl");
  }
  if (modifiers?.alt) {
    parts.push("Alt");
  }
  if (modifiers?.shift) {
    parts.push("Shift");
  }

  // Format the key nicely
  let displayKey = key;
  switch (key.toLowerCase()) {
    case "arrowup":
      displayKey = "Up";
      break;
    case "arrowdown":
      displayKey = "Down";
      break;
    case "arrowleft":
      displayKey = "Left";
      break;
    case "arrowright":
      displayKey = "Right";
      break;
    case "escape":
      displayKey = "Esc";
      break;
    case "enter":
      displayKey = "Enter";
      break;
    case " ":
      displayKey = "Space";
      break;
    default:
      displayKey = key.toUpperCase();
  }

  parts.push(displayKey);
  return parts.join(" + ");
}

/**
 * Group shortcuts by category for display
 */
export function groupShortcutsByCategory(
  shortcuts: KeyboardShortcut[]
): Map<string, KeyboardShortcut[]> {
  const groups = new Map<string, KeyboardShortcut[]>();

  for (const shortcut of shortcuts) {
    const category = shortcut.category;
    if (!groups.has(category)) {
      groups.set(category, []);
    }
    groups.get(category)!.push(shortcut);
  }

  return groups;
}
