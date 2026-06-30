---
id: doc:search-index-design
title: Search Index Design
author: lasse
components: [search-index]
---

# Search Index Design

The search index serves full-text lookups over documents and tickets. This design
covers the rebuild tracked under PROJ-7, including the analyzer chain and the
incremental refresh strategy.

## Auth coupling

Index refresh runs as an authenticated background job, so it depends on the new
token scheme described in the [[auth-redesign]] spec. If the token cache misbehaves,
index refresh stalls.
