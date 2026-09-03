import { useEffect, useState } from "react";

/**
 * Load once, expose the three states a fetch actually has.
 *
 * TanStack Query would be the reach here, but nothing in this app caches across
 * routes or refetches on focus yet. Adding it now would be a dependency carrying no
 * weight; the moment mutations exist it earns its place.
 */

export type AsyncState<T> =
  | { status: "loading" }
  | { status: "error"; error: Error }
  | { status: "ready"; data: T };

export function useAsync<T>(load: () => Promise<T>, deps: unknown[] = []): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({ status: "loading" });

  useEffect(() => {
    let live = true;
    setState({ status: "loading" });

    load()
      .then((data) => live && setState({ status: "ready", data }))
      .catch((error: unknown) => {
        if (!live) return;
        setState({
          status: "error",
          error: error instanceof Error ? error : new Error(String(error)),
        });
      });

    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}
