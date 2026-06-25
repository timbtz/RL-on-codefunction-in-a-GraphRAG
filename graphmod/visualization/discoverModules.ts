// ============================================================
// discoverModules.ts — find & load modules by `*.module.ts` filename
// ============================================================
//
// Walks a directory tree for files ending in `.module.ts`, dynamically
// imports each, and returns every exported value that looks like a module
// (i.e. `defineModule(...)` output). Lets the CLI render the whole schema
// without hand-maintaining a list of imports.

import { readdirSync } from "node:fs";
import { resolve, join, dirname } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import type { AnyModule } from "../src/core/types.js";

const HERE = dirname(fileURLToPath(import.meta.url));

/** Directory holding the modules, relative to this file (`visualization/`). */
const DEFAULT_MODULES_DIR = resolve(HERE, "../src/modules");

export interface DiscoverOptions {
  /** Root directory to scan. Defaults to `src/modules`. */
  dir?: string;
  /** Directory names to skip anywhere in the tree. Defaults to `["old"]`. */
  ignoreDirs?: string[];
}

/** Runtime guard: does this exported value look like a `defineModule` result? */
function isModule(value: unknown): value is AnyModule {
  if (typeof value !== "object" || value === null) return false;
  const schema = (value as { schema?: unknown }).schema;
  if (typeof schema !== "object" || schema === null) return false;
  const s = schema as Record<string, unknown>;
  return (
    typeof s.name === "string" &&
    typeof s.root === "string" &&
    typeof s.nodes === "object" &&
    s.nodes !== null
  );
}

/** Recursively collect absolute paths of every `*.module.ts` file under `dir`. */
export function findModuleFiles(
  dir: string = DEFAULT_MODULES_DIR,
  ignoreDirs: string[] = ["old"],
): string[] {
  const ignore = new Set(ignoreDirs);
  const out: string[] = [];

  const walk = (current: string) => {
    for (const entry of readdirSync(current, { withFileTypes: true })) {
      if (entry.isDirectory()) {
        if (ignore.has(entry.name)) continue;
        walk(join(current, entry.name));
      } else if (entry.isFile() && entry.name.endsWith(".module.ts")) {
        out.push(join(current, entry.name));
      }
    }
  };

  walk(dir);
  return out.sort();
}

/**
 * Find, import, and collect every module exported from a `*.module.ts` file
 * under `dir`. Deduplicates by `schema.name` (a module re-exported or
 * referenced from several files is only returned once).
 */
export async function discoverModules(
  options: DiscoverOptions = {},
): Promise<AnyModule[]> {
  const { dir = DEFAULT_MODULES_DIR, ignoreDirs = ["old"] } = options;
  const files = findModuleFiles(dir, ignoreDirs);

  const byName = new Map<string, AnyModule>();
  for (const file of files) {
    let mod: Record<string, unknown>;
    try {
      mod = (await import(pathToFileURL(file).href)) as Record<string, unknown>;
    } catch (err) {
      // A broken/legacy file shouldn't sink the whole run — warn and skip.
      const reason = err instanceof Error ? err.message : String(err);
      console.warn(`⚠ Skipped ${file}: ${reason}`);
      continue;
    }
    for (const exported of Object.values(mod)) {
      if (isModule(exported) && !byName.has(exported.schema.name)) {
        byName.set(exported.schema.name, exported);
      }
    }
  }

  return [...byName.values()];
}
