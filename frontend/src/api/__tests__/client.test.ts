import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getSessionToken, initAuthSession, resetAuthInitForTests } from "../client";

describe("initAuthSession StrictMode singleton guard", () => {
  const origLocation = window.location;

  beforeEach(() => {
    resetAuthInitForTests();
    vi.restoreAllMocks();
  });

  afterEach(() => {
    resetAuthInitForTests();
    window.history.replaceState(null, "", origLocation.pathname);
  });

  it("deduplicates concurrent StrictMode calls and strips ?token before await", async () => {
    window.history.replaceState(null, "", "/?token=one-time-tok&foo=1");

    let resolveFetch!: (resp: Response) => void;
    const fetchMock = vi.fn(
      (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Promise<Response>((resolve) => {
          resolveFetch = resolve;
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const p1 = initAuthSession();
    // ?token must be stripped synchronously before the first network await resolves
    expect(window.location.search).toBe("?foo=1");

    const p2 = initAuthSession();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(String(fetchMock.mock.calls[0]?.[0])).toContain(
      "/api/auth/bootstrap?token=one-time-tok",
    );

    resolveFetch(
      new Response(JSON.stringify({ authenticated: true, session_token: "sess-xyz" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );

    const [r1, r2] = await Promise.all([p1, p2]);
    expect(r1).toEqual({ authenticated: true, requireAuth: true });
    expect(r2).toEqual({ authenticated: true, requireAuth: true });
    expect(getSessionToken()).toBe("sess-xyz");
  });
});
