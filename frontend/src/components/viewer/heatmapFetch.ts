export interface HeatmapFetchPayload {
  blob: Blob;
  headers: Headers;
}

export class HeatmapFetchError extends Error {
  status: number;
  body: string;

  constructor(status: number, body: string) {
    super(`Heatmap fetch failed with status ${status}`);
    this.name = "HeatmapFetchError";
    this.status = status;
    this.body = body;
  }
}

// Shared in-flight map so prewarm + viewer can reuse the same request.
const inFlightHeatmapFetches = new Map<string, Promise<HeatmapFetchPayload>>();

export async function fetchHeatmapWithDedupe(
  url: string,
  init: RequestInit = {}
): Promise<HeatmapFetchPayload> {
  const key = `GET:${url}`;

  const existing = inFlightHeatmapFetches.get(key);
  if (existing) {
    return existing;
  }

  const request = (async () => {
    const response = await fetch(url, {
      method: "GET",
      ...init,
    });

    if (!response.ok) {
      let body = "";
      try {
        body = await response.text();
      } catch {
        // best effort
      }
      throw new HeatmapFetchError(response.status, body);
    }

    const blob = await response.blob();
    return {
      blob,
      headers: response.headers,
    };
  })();

  inFlightHeatmapFetches.set(key, request);

  return request.finally(() => {
    if (inFlightHeatmapFetches.get(key) === request) {
      inFlightHeatmapFetches.delete(key);
    }
  });
}
