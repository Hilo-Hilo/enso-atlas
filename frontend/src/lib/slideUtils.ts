// Slide name utilities for Enso Atlas
// Handles cleaning TCGA slide filenames, deduplication, and filtering

/**
 * UUID pattern: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
 * These appear in TCGA filenames like:
 *   TCGA-04-1331-01A-01-BS1.27aaf831-a80b-4a55-a239-3a24caca9c28.svs
 */
const UUID_PATTERN = /\.[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/i;

/**
 * Test file names that should be hidden from the UI.
 */
const HIDDEN_SLIDES = new Set(["slide_000.svs", "slide_000"]);

/**
 * Clean a slide filename for display.
 * 
 * Input:  "TCGA-04-1331-01A-01-BS1.27aaf831-a80b-4a55-a239-3a24caca9c28.svs"
 * Output: "TCGA-04-1331-01A-01-BS1"
 * 
 * Input:  "TCGA-04-1331-01A-01-BS1.svs"
 * Output: "TCGA-04-1331-01A-01-BS1"
 * 
 * Strips UUID and .svs extension. Returns original if no pattern matches.
 */
export function cleanSlideName(filename: string): string {
  if (!filename) return filename;

  // Remove .svs extension
  let name = filename.replace(/\.svs$/i, "");

  // Remove trailing UUID (e.g. .27aaf831-a80b-4a55-a239-3a24caca9c28)
  name = name.replace(UUID_PATTERN, "");

  return name || filename;
}

/**
 * Extract a short case ID from a TCGA filename.
 * 
 * Input:  "TCGA-04-1331-01A-01-BS1.27aaf831-..."
 * Output: "TCGA-04-1331"
 * 
 * Falls back to cleanSlideName if pattern doesn't match.
 */
export function extractCaseId(filename: string): string {
  const match = filename.match(/^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})/i);
  return match ? match[1] : cleanSlideName(filename);
}

/**
 * Get the base name of a slide (without UUID) for deduplication.
 * 
 * "TCGA-04-1360-01A-01-TS1.faa3ee9d-ddba-4592-b45a-bdee35da1633.svs" → "TCGA-04-1360-01A-01-TS1.svs"
 * "TCGA-04-1360-01A-01-TS1.svs" → "TCGA-04-1360-01A-01-TS1.svs"
 */
function getBaseFilename(filename: string): string {
  return filename.replace(UUID_PATTERN, "");
}

/**
 * Check if a slide should be hidden from the UI.
 */
export function isHiddenSlide(slide: { id: string; filename: string }): boolean {
  return HIDDEN_SLIDES.has(slide.filename) || HIDDEN_SLIDES.has(slide.id);
}

/**
 * Deduplicate slides: when both UUID and non-UUID versions exist,
 * keep only the UUID version (canonical). Also filters out hidden test slides.
 * 
 * Example:
 *   Input:  ["TCGA-04-1360-01A-01-TS1.faa3ee9d-....svs", "TCGA-04-1360-01A-01-TS1.svs"]
 *   Output: ["TCGA-04-1360-01A-01-TS1.faa3ee9d-....svs"]
 */
export function deduplicateSlides<T extends { id: string; filename: string }>(
  slides: T[]
): T[] {
  // First filter out hidden slides
  const visible = slides.filter((s) => !isHiddenSlide(s));

  // Group by base filename (without UUID)
  const groups = new Map<string, T[]>();
  for (const slide of visible) {
    const base = getBaseFilename(slide.filename);
    const group = groups.get(base) || [];
    group.push(slide);
    groups.set(base, group);
  }

  // For each group, prefer the version WITH UUID (longer filename = has UUID)
  const result: T[] = [];
  groups.forEach((group) => {
    if (group.length === 1) {
      result.push(group[0]);
    } else {
      // Pick the one with UUID (longer id/filename typically has UUID)
      const withUuid = group.find((s) => UUID_PATTERN.test(s.filename) || UUID_PATTERN.test(s.id));
      result.push(withUuid || group[0]);
    }
  });

  return result;
}

/**
 * Clean a slide ID for display (IDs are filenames without .svs in this codebase).
 * Removes UUID suffix from slide IDs.
 */
export function cleanSlideId(slideId: string): string {
  return slideId.replace(UUID_PATTERN, "");
}
