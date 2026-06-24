// ============================================================
// visualize — turn module schemas into an interactive graph page
// ============================================================

import { writeFileSync } from "node:fs";
import type { AnyModule } from "../src/core/types.js";
import { buildGraphModel, type GraphModel } from "./buildModel.js";
import { renderHtml, type RenderOptions } from "./renderHtml.js";

export { buildGraphModel, renderHtml };
export type { GraphModel, RenderOptions };
export type {
  NodeInfo,
  EdgeInfo,
  ModuleInfo,
  PropInfo,
  EdgeKind,
} from "./buildModel.js";

/** Render the given modules (and anything they reference) to an HTML string. */
export function visualizeModules(
  modules: AnyModule[],
  options?: RenderOptions,
): string {
  return renderHtml(buildGraphModel(modules), options);
}

/** Render and write the HTML page to `filePath`. Returns the path written. */
export function visualizeToFile(
  modules: AnyModule[],
  filePath: string,
  options?: RenderOptions,
): string {
  writeFileSync(filePath, visualizeModules(modules, options), "utf8");
  return filePath;
}
