---
id: doc:billing-overview
title: Billing Overview
author: mara
components: [billing]
---

# Billing Overview

The billing subsystem converts metered usage into invoices. This overview covers
the migration from the legacy monthly batch job to streaming usage events,
tracked under PROJ-42.

## Usage events

Each usage event carries a tenant id, a meter name, and a quantity. Events are
aggregated per billing period. The new flow is documented for new hires in the
[[onboarding]] guide.
