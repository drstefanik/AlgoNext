const DEFAULT_TIMEOUT_MS = 12000;

function getBackendBaseUrl() {
  return (
    process.env.BACKEND_URL ||
    process.env.NEXT_PUBLIC_BACKEND_URL ||
    ""
  ).replace(/\/$/, "");
}

function clampTimeout(value) {
  if (!Number.isFinite(value)) {
    return DEFAULT_TIMEOUT_MS;
  }
  return Math.min(Math.max(value, 10000), 15000);
}

function buildTargetUrl(baseUrl, pathParts, query) {
  const path = pathParts.length ? `/${pathParts.join("/")}` : "/";
  const targetUrl = new URL(`${baseUrl}${path}`);
  Object.entries(query).forEach(([key, value]) => {
    if (key === "path" || value === undefined) {
      return;
    }
    if (Array.isArray(value)) {
      value.forEach((item) => targetUrl.searchParams.append(key, String(item)));
    } else {
      targetUrl.searchParams.append(key, String(value));
    }
  });
  return targetUrl;
}

function buildForwardHeaders(headers) {
  const forward = {};
  Object.entries(headers || {}).forEach(([key, value]) => {
    if (!value) {
      return;
    }
    const lower = key.toLowerCase();
    if (lower === "host" || lower === "connection" || lower === "content-length") {
      return;
    }
    forward[key] = value;
  });
  return forward;
}

function extractRequestBody(req) {
  if (req.method === "GET" || req.method === "HEAD") {
    return undefined;
  }
  if (req.body === undefined || req.body === null) {
    return undefined;
  }
  if (typeof req.body === "string" || Buffer.isBuffer(req.body)) {
    return req.body;
  }
  return JSON.stringify(req.body);
}

export default async function handler(req, res) {
  const backendBaseUrl = getBackendBaseUrl();
  if (!backendBaseUrl) {
    res
      .status(500)
      .json({ ok: false, error: "Backend URL not configured" });
    return;
  }

  const timeoutMs = clampTimeout(
    Number(process.env.BACKEND_TIMEOUT_MS || DEFAULT_TIMEOUT_MS)
  );
  const pathParts = Array.isArray(req.query.path)
    ? req.query.path
    : req.query.path
      ? [req.query.path]
      : [];
  const targetUrl = buildTargetUrl(backendBaseUrl, pathParts, req.query);

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(targetUrl.toString(), {
      method: req.method,
      headers: buildForwardHeaders(req.headers),
      body: extractRequestBody(req),
      signal: controller.signal,
    });
    const payload = await response.arrayBuffer();
    res.status(response.status);
    response.headers.forEach((value, key) => {
      if (key.toLowerCase() === "transfer-encoding") {
        return;
      }
      res.setHeader(key, value);
    });
    res.send(Buffer.from(payload));
  } catch (error) {
    if (error?.name === "AbortError") {
      res
        .status(504)
        .json({ ok: false, error: "Backend unavailable, retry" });
    } else {
      res.status(502).json({ ok: false, error: "Proxy error" });
    }
  } finally {
    clearTimeout(timeout);
  }
}
