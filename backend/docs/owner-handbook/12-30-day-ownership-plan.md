# 12 - 30-Day Ownership Plan

## Goal
Move from "can run the project" to "can own production-safe changes" in 30 days.

## Structure
- Weeks 1-4: core ownership skills (20 focused workdays)
- Days 21-30: reinforcement, drills, and independent delivery
- Each week has measurable outcomes and a self-assessment

## Week 1 - Map and Run the System

### Weekly Goal
Understand architecture and run the full stack confidently.

### Daily Plan
| Day | Focus | Tasks | Deliverable |
|---|---|---|---|
| 1 | System overview | Read modules 01 + 02. Draw your own architecture sketch. | Personal architecture sketch |
| 2 | Backend boot | Run backend manually. Explore `/docs` and `/health`. | Working backend + endpoint notes |
| 3 | Frontend boot | Run frontend and validate login flow paths. | UI route map |
| 4 | Docker mode | Run docker-compose and compare with manual mode. | Runtime comparison notes |
| 5 | Flow tracing | Trace login flow end-to-end (UI -> API -> DB effects). | Written flow trace |

### Week 1 Checkpoint
- [ ] I can start stack in both manual and docker modes.
- [ ] I can explain where auth/session state lives.
- [ ] I can identify route file for any top-nav page.

## Week 2 - Backend and Data Ownership

### Weekly Goal
Be able to change backend behavior and schema safely.

### Daily Plan
| Day | Focus | Tasks | Deliverable |
|---|---|---|---|
| 6 | Route architecture | Read module 03 and inspect 3 route files deeply. | Route responsibility notes |
| 7 | Service deep dive | Trace one trade endpoint to service/model/external API. | Sequence diagram |
| 8 | Data model map | Read module 06 and classify models by domain. | Model-domain cheat sheet |
| 9 | Migration practice | Create a test migration on local branch (non-critical field) and apply/revert strategy. | Migration dry-run notes |
| 10 | Risk/security pass | Read module 07 and audit a risk-sensitive endpoint path. | Security review checklist |

### Week 2 Checkpoint
- [ ] I can add a backend endpoint using existing patterns.
- [ ] I understand migration flow and common pitfalls.
- [ ] I can explain token and credential security paths.

## Week 3 - Frontend + AI Integration Ownership

### Weekly Goal
Confidently connect UI changes to backend and AI services.

### Daily Plan
| Day | Focus | Tasks | Deliverable |
|---|---|---|---|
| 11 | Frontend architecture | Read module 04 and trace one page lifecycle. | Component-service map |
| 12 | API service layer | Trace a frontend service call to backend route and response render. | End-to-end call chain note |
| 13 | gRPC contract | Read module 05 and inspect proto + backend adapter. | RPC summary sheet |
| 14 | AI backend behavior | Compare llm-chain vs cli-agent flow for one endpoint. | Comparison table |
| 15 | Streaming behavior | Trace one SSE path and cancellation behavior. | SSE control-flow notes |

### Week 3 Checkpoint
- [ ] I can explain how provider routing is chosen.
- [ ] I can diagnose AI path failures by layer (route/gRPC/service/provider).
- [ ] I can add a frontend service method and wire it to UI.

## Week 4 - Safe Delivery and Operations

### Weekly Goal
Deliver changes with playbooks, tests, and operational confidence.

### Daily Plan
| Day | Focus | Tasks | Deliverable |
|---|---|---|---|
| 16 | Pattern reinforcement | Read module 08 and map 5 patterns to real files. | Pattern-to-file cheat sheet |
| 17 | Navigation speed | Use module 09 to solve 5 "where to change X" drills. | Completed drill sheet |
| 18 | Operations readiness | Read module 10 and run full pre-merge checklist once. | Ops checklist run log |
| 19 | Playbook simulation | Execute one playbook end-to-end on a small branch change. | Mini PR with notes |
| 20 | Ownership review | Self-review gaps and build a personal runbook summary. | Personal ownership runbook |

### Week 4 Checkpoint
- [ ] I can execute at least one playbook independently.
- [ ] I can run and interpret core checks.
- [ ] I can propose a safe rollout + rollback for a medium change.

## Days 21-30 - Reinforcement and Independent Execution

### Focus Blocks
| Day Range | Focus | Expected Output |
|---|---|---|
| 21-23 | Bug triage drills | 3 reproduced issues with root-cause writeups |
| 24-26 | Feature delivery drill | 1 small fullstack feature branch using playbooks |
| 27-28 | Hardening pass | Add/adjust tests for one risk-sensitive path |
| 29 | Incident simulation | Mock outage/runbook exercise |
| 30 | Final readiness | Present "how I own this repo" summary |

## Weekly Self-Assessment Template
Use this every Friday:
1. What flows can I trace without help?
2. Which module is still unclear?
3. Which change type do I still avoid (schema/auth/ai/worker)?
4. What single task next week closes that gap?

## Required Mini Deliverables by End of 30 Days
- [ ] One architecture diagram in your own words
- [ ] One route-to-service-to-model trace document
- [ ] One migration dry-run report
- [ ] One AI route debug report
- [ ] One small merged-quality change (or PR-ready branch)

## “Ready To Own Production Changes” Final Checklist
- [ ] I can implement a backend endpoint safely with tests.
- [ ] I can add/change frontend page + service wiring safely.
- [ ] I can perform schema change with migration discipline.
- [ ] I can diagnose auth/session problems quickly.
- [ ] I can diagnose AI/gRPC path failures by layer.
- [ ] I can use ops checklist before merge/deploy.
- [ ] I can document rollback plan for risky changes.
