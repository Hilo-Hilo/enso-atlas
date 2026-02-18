/**
 * Proxy for slide thumbnails - avoids CORS issues
 *
 * Graceful behavior: if backend thumbnail is unavailable (e.g., missing WSI),
 * return a lightweight SVG placeholder image instead of JSON to avoid broken
 * image icons in the UI.
 */
import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8003";

function fallbackSvg(slideId: string, size: number): string {
  const safeSlide = slideId.replace(/[<>&"']/g, "");
  const s = Math.max(64, Math.min(size || 256, 1024));
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${s}" height="${s}" viewBox="0 0 ${s} ${s}">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f1f5f9"/>
      <stop offset="100%" stop-color="#e2e8f0"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="${s}" height="${s}" fill="url(#g)"/>
  <rect x="1" y="1" width="${s - 2}" height="${s - 2}" fill="none" stroke="#cbd5e1" stroke-width="2"/>
  <text x="50%" y="46%" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="${Math.max(11, Math.floor(s * 0.08))}" fill="#334155">Embeddings</text>
  <text x="50%" y="58%" text-anchor="middle" font-family="Inter, system-ui, sans-serif" font-size="${Math.max(9, Math.floor(s * 0.055))}" fill="#64748b">WSI unavailable</text>
  <text x="50%" y="78%" text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="${Math.max(8, Math.floor(s * 0.04))}" fill="#94a3b8">${safeSlide.slice(0, 28)}</text>
</svg>`;
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ slideId: string }> }
) {
  const { slideId } = await params;
  const { searchParams } = new URL(request.url);
  const sizeRaw = searchParams.get("size") || "256";
  const projectId = searchParams.get("project_id");
  const size = Number.parseInt(sizeRaw, 10) || 256;

  try {
    const backendQs = new URLSearchParams({ size: String(size) });
    if (projectId) backendQs.set("project_id", projectId);
    const backendUrl = `${BACKEND_URL}/api/slides/${encodeURIComponent(slideId)}/thumbnail?${backendQs.toString()}`;

    const response = await fetch(backendUrl, {
      method: "GET",
    });

    if (!response.ok) {
      const svg = fallbackSvg(slideId, size);
      return new NextResponse(svg, {
        status: 200,
        headers: {
          "Content-Type": "image/svg+xml",
          "Cache-Control": "public, max-age=3600",
          "X-Thumbnail-Fallback": "proxy-generated",
          "X-WSI-Available": "false",
        },
      });
    }

    const imageBuffer = await response.arrayBuffer();
    const contentType = response.headers.get("Content-Type") || "image/jpeg";
    const wsiAvailable = response.headers.get("X-WSI-Available") || "true";

    return new NextResponse(imageBuffer, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=86400",
        "X-WSI-Available": wsiAvailable,
      },
    });
  } catch (error) {
    console.error(`[Thumbnail Proxy] Error fetching thumbnail for ${slideId}:`, error);
    const svg = fallbackSvg(slideId, size);
    return new NextResponse(svg, {
      status: 200,
      headers: {
        "Content-Type": "image/svg+xml",
        "Cache-Control": "public, max-age=3600",
        "X-Thumbnail-Fallback": "proxy-generated",
        "X-WSI-Available": "false",
      },
    });
  }
}
