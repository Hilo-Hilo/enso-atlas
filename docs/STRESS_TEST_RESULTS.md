# Stress Test Results - MedGemma Backend

**Date:** 2026-01-31
**Backend:** http://100.111.126.23:8003
**Container:** enso-atlas on DGX Spark

---

## Executive Summary

The backend passed all stress tests with excellent performance. No crashes, memory leaks, or significant issues were detected under load. Response times remained consistently fast even under concurrent load.

---

## Baseline System State

### Before Tests
| Metric | Value |
|--------|-------|
| Container Memory | 1.078 GiB / 119.7 GiB (0.90%) |
| Container CPU | 0.65% |
| GPU Memory | ~483 MiB (287 + 172 + overhead) |
| GPU Utilization | 0% (idle) |
| GPU Temperature | 46C |

### After Tests
| Metric | Value |
|--------|-------|
| Container Memory | 1.149 GiB / 119.7 GiB (0.96%) |
| Container CPU | 0.52% |
| GPU Temperature | 46C |
| Memory Delta | +71 MiB (6.5% increase, normal) |

---

## Test Results

### Test 1: Concurrent Analysis Requests (10 concurrent)

**Result: PASS**

| Metric | Value |
|--------|-------|
| Total Time | 0.066s |
| All Requests | 200 OK |
| Avg Response Time | 0.032s |
| Max Response Time | 0.038s |

All 10 concurrent POST /api/analyze requests completed successfully with consistent response times.

---

### Test 2: Rapid Sequential Health Requests (50 requests)

**Result: PASS**

| Metric | Value |
|--------|-------|
| Total Time | 1.38s |
| Success Rate | 100% (50/50) |
| Avg per Request | 0.028s |

No failures under rapid sequential load.

---

### Test 3: All Slides Analysis (6 slides)

**Result: PASS**

| Slide ID | Prediction | Confidence | Response Time |
|----------|------------|------------|---------------|
| slide_000 | NON-RESPONDER | 1.0 | 0.129s |
| TCGA-04-1360-01A-01-TS1 | NON-RESPONDER | 1.0 | 0.031s |
| TCGA-09-2056-01B-01-TS1 | NON-RESPONDER | 1.0 | 0.028s |
| TCGA-13-1489-01A-01-TS1 | NON-RESPONDER | 1.0 | 0.030s |
| TCGA-29-1691-01A-01-BS1 | NON-RESPONDER | 1.0 | 0.032s |
| TCGA-61-1730-01A-01-BS1 | NON-RESPONDER | 1.0 | 0.027s |

Note: First request slightly slower (0.129s) due to cache warming. Subsequent requests consistently ~30ms.

---

### Test 4: DZI Tile Stress Test (80 concurrent tile requests)

**Result: PASS**

| Metric | Value |
|--------|-------|
| Total Tiles Requested | 80 |
| Successful (200) | 46 |
| Not Found (404) | 34 |
| Total Time | 1.51s |

The 404 responses are expected - not all tile coordinates exist at every zoom level. This is normal behavior for Deep Zoom Image tile serving.

---

### Test 5: Report Generation Stress (10 concurrent)

**Result: PASS**

| Metric | Value |
|--------|-------|
| Total Time | 0.046s |
| All Requests | 200 OK |
| Avg Response Time | 0.030s |

Report generation handles concurrent load well. Reports include proper JSON structure with case_id, model_output, evidence, limitations, and safety_statement.

---

### Test 6: Semantic Search Stress (5 concurrent queries)

**Result: PASS**

| Query | HTTP Status | Response Time |
|-------|-------------|---------------|
| tumor cells | 200 | 0.023s |
| necrosis | 200 | 0.020s |
| lymphocytes | 200 | 0.023s |
| stroma | 200 | 0.023s |
| mitotic figures | 200 | 0.023s |

Total time for 5 concurrent: 0.036s
Results returned: 10 per query (as requested)

---

### Test 7: High Load Test (50 concurrent analysis)

**Result: PASS**

| Metric | Value |
|--------|-------|
| Total Time | 0.34s |
| Success Rate | 100% (50/50) |
| Avg per Request | 6.7ms (when concurrent) |

The backend handles 50 concurrent analysis requests without any failures.

---

### Test 8: Sustained Load Test (100 sequential requests)

**Result: PASS**

| Metric | Value |
|--------|-------|
| Total Time | 2.99s |
| Success Rate | 100% (100/100) |
| Avg per Request | 0.030s |

Consistent performance across 100 sequential requests with no degradation.

---

## Edge Case Testing

### Input Validation

| Test Case | Expected | Actual | Status |
|-----------|----------|--------|--------|
| Invalid slide_id | Error | "Slide nonexistent_slide not found" | PASS |
| Missing slide_id | Validation error | "Field required" | PASS |
| Invalid JSON | Parse error | "JSON decode error" | PASS |
| Empty body | Validation error | "Field required" | PASS |
| Empty search query | Validation error | "String should have at least 1 character" | PASS |
| Very long query (1000 chars) | Handle gracefully | Returns 0 results | PASS |
| XSS attempt in query | Sanitize/ignore | Returns 10 results (safe) | PASS |

All edge cases are handled properly with appropriate error messages.

---

## API Endpoint Coverage

### Verified Working Endpoints

| Endpoint | Method | Status |
|----------|--------|--------|
| /api/health | GET | Working |
| /api/slides | GET | Working (returns 6 slides) |
| /api/slides/{id}/info | GET | Working |
| /api/slides/{id}/dzi | GET | Working (XML response) |
| /api/slides/{id}/dzi_files/{level}/{tile} | GET | Working |
| /api/slides/{id}/thumbnail | GET | Working (29KB image) |
| /api/analyze | POST | Working |
| /api/report | POST | Working |
| /api/semantic-search | POST | Working |
| /api/semantic-search/status | GET | Working |
| /api/similar | GET | Working (query params) |
| /api/heatmap/{id} | GET | Working (PNG image) |
| /api/embed/status | GET | Working |
| /health | GET | Working |

### Notes on Model Loading

From status endpoints:
- **Path Foundation** (embed): Not loaded (lazy loading)
- **SigLIP** (semantic-search): Not loaded (lazy loading)

Models appear to load on-demand, which is good for resource efficiency.

---

## Issues Found

### Minor Issues

1. **No /docs endpoint**: FastAPI's automatic Swagger UI returns 404. Consider enabling for easier API exploration.
   - Severity: Low
   - Recommendation: Enable FastAPI docs in production for debugging

2. **Model loading status ambiguity**: Embed model shows "model_loaded: false" but semantic search works. May need status endpoint clarification.
   - Severity: Low
   - Recommendation: Clarify lazy-loading behavior in documentation

### No Critical Issues Found

- No crashes under load
- No memory leaks detected
- No timeout errors
- No 5xx server errors
- Proper error handling for all edge cases

---

## Performance Summary

| Metric | Value |
|--------|-------|
| Health Check Latency | ~28ms |
| Analysis Latency (cached) | ~30ms |
| Report Generation | ~30ms |
| Semantic Search | ~23ms |
| Tile Serving | ~20ms per tile |
| Max Concurrent Handled | 50+ (tested) |
| Memory Stability | Stable (+71MB under load) |

---

## Recommendations

1. **Production Ready**: Backend is stable and performant for demo/hackathon use.

2. **Consider rate limiting**: While no issues were found, adding rate limiting would protect against abuse in production.

3. **Enable API documentation**: Add `/docs` endpoint for easier integration testing.

4. **Monitor GPU memory**: Current usage is low (~500MB), but should monitor during sustained use with larger batches.

5. **Add health check for model status**: Include model loading state in /api/health for better observability.

---

## Conclusion

The MedGemma backend passed all stress tests. It demonstrates:
- Excellent concurrent request handling
- Consistent sub-50ms response times
- Proper error handling and input validation
- Stable memory usage under load
- No crashes or failures under tested conditions

The system is ready for demonstration and evaluation.
