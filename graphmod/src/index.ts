// ============================================================
// index.ts — Public API
// ============================================================

export { defineModule } from "./core/module.js";
export type { Module } from "./core/module.js";

export { createLazyLoader } from "./core/lazy.js";
export type { LazyHydratedNode, InferModuleLazyHydrated, LazyLoader, LazyChain } from "./core/lazy.js";

export { createModuleInstance } from "./core/context.js";
export type { ModuleInstance } from "./core/context.js";

export { createDeleteContext } from "./core/context.js";
export { buildTraversalQuery } from "./core/cypher.js";
export { validateCardinality, assertValidInTx, SchemaValidationError } from "./core/validate.js";
export type { ValidationError } from "./core/validate.js";

export { isModuleRef, isInboundModuleRef } from "./core/types.js";
export type {
  PropType,
  Cardinality,
  NodeDef,
  RelationshipDef,
  ModuleRef,
  InboundModuleRef,
  RelOrModule,
  ModuleSchema,
  GraphSession,
  GraphTransaction,
  DeleteHandler,
  DeleteContext,
  AnyModule,
  HydratedNode,
  ResolveNode,
  ResolveProps,
  ResolveProp,
  CardinalityWrap,
  InferModuleHydrated,
} from "./core/types.js";

export * from "./modules/index.js";
