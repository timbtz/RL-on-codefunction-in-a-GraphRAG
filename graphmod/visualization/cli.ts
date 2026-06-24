// ============================================================
// visualization/cli.ts — CLI: render module schemas to an HTML page
// ============================================================
//
//   npm run visualize                         # all modules → graph-modules.html
//   npm run visualize -- out.html             # custom output path
//   npm run visualize -- --no-props           # start with properties hidden
//   npm run visualize -- --title "My Graph"   # custom heading
//
// Modules are discovered automatically by scanning `src/modules` for
// `*.module.ts` files, so new modules show up without editing this file.

import { resolve, isAbsolute, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { visualizeToFile } from "./index";
import { discoverModules } from "./discoverModules";

// Default output lives next to this script, inside visualization/.
const HERE = dirname(fileURLToPath(import.meta.url));

async function main() {
  const MODULES = await discoverModules();
  const args = process.argv.slice(2);
  let out = "graph-modules.html";
  let showProperties = true;
  let title = "Graph Module Map";

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--no-props") showProperties = false;
    else if (a === "--props") showProperties = true;
    else if (a === "--title") title = args[++i] ?? title;
    else if (!a.startsWith("-")) out = a;
  }

  // Bare default → write into visualization/. An explicit path (absolute or
  // relative) is resolved against the cwd as the user typed it.
  const path =
    out === "graph-modules.html"
      ? resolve(HERE, out)
      : isAbsolute(out)
        ? out
        : resolve(process.cwd(), out);
  visualizeToFile(MODULES, path, { showProperties, title });
  console.log(`✓ Wrote ${path}`);
  console.log(`  ${MODULES.length} module(s) discovered; open it in a browser.`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
