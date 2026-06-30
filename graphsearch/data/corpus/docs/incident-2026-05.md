---
id: doc:incident-2026-05
title: May 2026 Auth Incident Postmortem
author: mara
components: [auth-redesign]
---

# May 2026 Auth Incident Postmortem

On 2026-05-14 logins failed for roughly forty minutes. The root cause was an
incorrect eviction policy in the token cache introduced during the auth redesign.

## Resolution

Noah diagnosed the eviction bug and shipped the hotfix the same afternoon. The
follow-up hardening work is tracked under PROJ-19. Tim owns the longer-term fix to
the token cache eviction policy.
