---
id: doc:auth-redesign
title: Auth Redesign Spec
author: tim
components: [auth-redesign]
---

# Auth Redesign Spec

This document specifies the migration of our authentication stack to OAuth 2.1.
The current session-cookie scheme is replaced by short-lived access tokens plus a
rotating refresh token. The work is tracked under PROJ-12.

## Token cache

Access tokens are validated against an in-memory token cache. The cache is the
single most performance-sensitive part of the redesign and the most likely source
of incidents if its eviction policy is wrong.

## Rollout

Rollout is staged behind a feature flag. See the search index design for how
token lookups interact with the [[search-index-design]] path.
