"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { Spinner } from "@/components/ui/Spinner";
import { cn } from "@/lib/utils";
import {
  FileText,
  Plus,
  Save,
  Trash2,
  Edit3,
  Clock,
  User,
  ChevronDown,
  ChevronUp,
  AlertCircle,
} from "lucide-react";

export interface CaseNote {
  id: string;
  slideId: string;
  content: string;
  author: string;
  createdAt: string;
  updatedAt?: string;
  category?: "clinical" | "pathology" | "treatment" | "general";
}

interface CaseNotesPanelProps {
  slideId: string | null;
  onNotesUpdate?: (notes: CaseNote[]) => void;
}

// LocalStorage key prefix
const NOTES_STORAGE_KEY = "atlas-case-notes";

// Helper to get notes from localStorage
function loadNotes(slideId: string): CaseNote[] {
  if (typeof window === "undefined") return [];
  try {
    const stored = localStorage.getItem(`${NOTES_STORAGE_KEY}-${slideId}`);
    return stored ? JSON.parse(stored) : [];
  } catch (err) {
    console.error("Failed to load notes from localStorage:", err);
    return [];
  }
}

// Helper to save notes to localStorage
function saveNotes(slideId: string, notes: CaseNote[]): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(`${NOTES_STORAGE_KEY}-${slideId}`, JSON.stringify(notes));
  } catch (err) {
    console.error("Failed to save notes to localStorage:", err);
  }
}

// Export function to get notes for PDF export
export function getCaseNotes(slideId: string): CaseNote[] {
  return loadNotes(slideId);
}

export function CaseNotesPanel({ slideId, onNotesUpdate }: CaseNotesPanelProps) {
  const [notes, setNotes] = useState<CaseNote[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(true);
  const [isAdding, setIsAdding] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [newNoteContent, setNewNoteContent] = useState("");
  const [newNoteCategory, setNewNoteCategory] = useState<CaseNote["category"]>("clinical");
  const [error, setError] = useState<string | null>(null);

  // Load notes when slide changes
  useEffect(() => {
    if (!slideId) {
      setNotes([]);
      return;
    }

    setIsLoading(true);
    // Load from localStorage
    const storedNotes = loadNotes(slideId);
    setNotes(storedNotes);
    setIsLoading(false);
  }, [slideId]);

  const handleAddNote = useCallback(() => {
    if (!slideId || !newNoteContent.trim()) return;

    const newNote: CaseNote = {
      id: `note_${Date.now()}`,
      slideId,
      content: newNoteContent.trim(),
      author: "Current User",
      createdAt: new Date().toISOString(),
      category: newNoteCategory,
    };

    const updatedNotes = [...notes, newNote];
    setNotes(updatedNotes);
    saveNotes(slideId, updatedNotes);
    onNotesUpdate?.(updatedNotes);

    setNewNoteContent("");
    setIsAdding(false);
  }, [slideId, newNoteContent, newNoteCategory, notes, onNotesUpdate]);

  const handleUpdateNote = useCallback(
    (noteId: string, content: string) => {
      if (!slideId) return;

      const updatedNotes = notes.map((n) =>
        n.id === noteId
          ? { ...n, content, updatedAt: new Date().toISOString() }
          : n
      );
      setNotes(updatedNotes);
      saveNotes(slideId, updatedNotes);
      onNotesUpdate?.(updatedNotes);
      setEditingId(null);
    },
    [slideId, notes, onNotesUpdate]
  );

  const handleDeleteNote = useCallback(
    (noteId: string) => {
      if (!slideId) return;

      const updatedNotes = notes.filter((n) => n.id !== noteId);
      setNotes(updatedNotes);
      saveNotes(slideId, updatedNotes);
      onNotesUpdate?.(updatedNotes);
    },
    [slideId, notes, onNotesUpdate]
  );

  const getCategoryColor = (category?: CaseNote["category"]) => {
    switch (category) {
      case "clinical":
        return "bg-blue-100 text-blue-700";
      case "pathology":
        return "bg-purple-100 text-purple-700";
      case "treatment":
        return "bg-green-100 text-green-700";
      default:
        return "bg-gray-100 text-gray-700";
    }
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  if (!slideId) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-4 w-4 text-gray-400" />
            Case Notes
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6 text-gray-500">
            <FileText className="h-8 w-8 mx-auto mb-2 text-gray-300" />
            <p className="text-sm">Select a slide to view notes</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-4 w-4 text-clinical-600" />
            Case Notes
            {notes.length > 0 && (
              <Badge variant="info" size="sm" className="font-mono">
                {notes.length}
              </Badge>
            )}
          </CardTitle>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsAdding(true)}
              disabled={isAdding}
              className="p-1.5"
              title="Add note"
            >
              <Plus className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
              className="p-1.5"
            >
              {isExpanded ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>
      </CardHeader>

      {isExpanded && (
        <CardContent className="pt-2 space-y-3">
          {/* Error Display */}
          {error && (
            <div className="p-2 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-red-500" />
              <span className="text-sm text-red-700">{error}</span>
            </div>
          )}

          {/* Loading State */}
          {isLoading && (
            <div className="flex items-center justify-center py-4">
              <Spinner size="sm" />
              <span className="text-sm text-gray-500 ml-2">Loading notes...</span>
            </div>
          )}

          {/* Add Note Form */}
          {isAdding && (
            <div className="p-3 bg-clinical-50 border border-clinical-200 rounded-lg animate-fade-in">
              <div className="flex items-center gap-2 mb-2">
                <select
                  value={newNoteCategory}
                  onChange={(e) =>
                    setNewNoteCategory(e.target.value as CaseNote["category"])
                  }
                  className="text-xs px-2 py-1 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-clinical-500"
                >
                  <option value="clinical">Clinical</option>
                  <option value="pathology">Pathology</option>
                  <option value="treatment">Treatment</option>
                  <option value="general">General</option>
                </select>
              </div>
              <textarea
                value={newNoteContent}
                onChange={(e) => setNewNoteContent(e.target.value)}
                placeholder="Enter clinical notes, observations, or comments..."
                className="w-full p-2 text-sm border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-clinical-500"
                rows={3}
                autoFocus
              />
              <div className="flex justify-end gap-2 mt-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setIsAdding(false);
                    setNewNoteContent("");
                  }}
                >
                  Cancel
                </Button>
                <Button
                  variant="primary"
                  size="sm"
                  onClick={handleAddNote}
                  disabled={!newNoteContent.trim()}
                >
                  <Save className="h-3 w-3 mr-1" />
                  Save Note
                </Button>
              </div>
            </div>
          )}

          {/* Notes List */}
          {!isLoading && notes.length === 0 && !isAdding && (
            <div className="text-center py-6 text-gray-500">
              <FileText className="h-8 w-8 mx-auto mb-2 text-gray-300" />
              <p className="text-sm font-medium text-gray-600">No notes yet</p>
              <p className="text-xs mt-1">
                Add clinical observations and comments
              </p>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setIsAdding(true)}
                className="mt-3"
              >
                <Plus className="h-3 w-3 mr-1" />
                Add First Note
              </Button>
            </div>
          )}

          {!isLoading && notes.length > 0 && (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {notes.map((note) => (
                <NoteItem
                  key={note.id}
                  note={note}
                  isEditing={editingId === note.id}
                  onEdit={() => setEditingId(note.id)}
                  onSave={(content) => handleUpdateNote(note.id, content)}
                  onDelete={() => handleDeleteNote(note.id)}
                  onCancelEdit={() => setEditingId(null)}
                  getCategoryColor={getCategoryColor}
                  formatDate={formatDate}
                />
              ))}
            </div>
          )}
        </CardContent>
      )}
    </Card>
  );
}

// Individual Note Item
interface NoteItemProps {
  note: CaseNote;
  isEditing: boolean;
  onEdit: () => void;
  onSave: (content: string) => void;
  onDelete: () => void;
  onCancelEdit: () => void;
  getCategoryColor: (category?: CaseNote["category"]) => string;
  formatDate: (dateStr: string) => string;
}

function NoteItem({
  note,
  isEditing,
  onEdit,
  onSave,
  onDelete,
  onCancelEdit,
  getCategoryColor,
  formatDate,
}: NoteItemProps) {
  const [editContent, setEditContent] = useState(note.content);

  useEffect(() => {
    setEditContent(note.content);
  }, [note.content]);

  if (isEditing) {
    return (
      <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
        <textarea
          value={editContent}
          onChange={(e) => setEditContent(e.target.value)}
          className="w-full p-2 text-sm border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-clinical-500"
          rows={3}
          autoFocus
        />
        <div className="flex justify-end gap-2 mt-2">
          <Button variant="ghost" size="sm" onClick={onCancelEdit}>
            Cancel
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={() => onSave(editContent)}
            disabled={!editContent.trim()}
          >
            <Save className="h-3 w-3 mr-1" />
            Update
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="p-3 bg-white dark:bg-slate-800 border border-gray-200 rounded-lg hover:border-gray-300 transition-colors group">
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-2 flex-wrap">
          {note.category && (
            <span
              className={cn(
                "text-2xs px-2 py-0.5 rounded-full font-medium",
                getCategoryColor(note.category)
              )}
            >
              {note.category}
            </span>
          )}
          <span className="text-2xs text-gray-500 flex items-center gap-1">
            <User className="h-3 w-3" />
            {note.author}
          </span>
        </div>
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={onEdit}
            className="p-1 text-gray-400 hover:text-clinical-600 transition-colors"
            title="Edit note"
          >
            <Edit3 className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={onDelete}
            className="p-1 text-gray-400 hover:text-red-600 transition-colors"
            title="Delete note"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
      <p className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
        {note.content}
      </p>
      <div className="flex items-center gap-2 mt-2 text-2xs text-gray-400">
        <Clock className="h-3 w-3" />
        <span>{formatDate(note.createdAt)}</span>
        {note.updatedAt && (
          <span className="italic">(edited {formatDate(note.updatedAt)})</span>
        )}
      </div>
    </div>
  );
}
