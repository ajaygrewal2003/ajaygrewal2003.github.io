# Ajay Singh Grewal
### Machine Learning • Artificial Intelligence • Software Engineer

Welcome to my personal site. I am currently studying machine learning, AI, and building useful software. I will be sharing my research notes, learning progress, and projects as I continue developing in this field. This page will grow as I move through my AI learning roadmap.

## Quick Navigation

- Research
- Projects
- Useful Resources
- Contact

## Latest Updates

- Research: April 2026 — Engineering Agentic AI Systems: A Research Report on Building Proxi
- Project: April 2026 — Proxi — Full-Stack Agentic AI Framework

---

# 📚 Research

### Featured

<details open>
<summary><strong>April 2026 — Engineering Agentic AI Systems: A Research Report on Building Proxi</strong></summary>

## April 2026 — Engineering Agentic AI Systems: A Research Report on Building Proxi

### Introduction

This research report documents the design, architecture, and engineering decisions made during the development of **Proxi**, a full-stack agentic AI framework built as a Capstone project. Unlike a simple LLM wrapper or chatbot, Proxi is a complete *agent operating system* — a system where an AI model is not just generating text, but reasoning, planning, calling tools, delegating to sub-agents, managing memory across unlimited sessions, and coordinating across multiple user interfaces simultaneously.

The core research challenge was: *how do you build an AI agent that can act reliably and intelligently over long, open-ended tasks, without hallucinating, losing context, or failing silently?* This required deep engagement with the real engineering problems of production agentic systems, including context window management, multi-model abstraction, persistent memory architectures, tool safety gating, human-in-the-loop design, and reliable async concurrency.

What follows is a breakdown of the major AI and systems engineering problems encountered, and how they were solved.

---

### 1. The Agentic Loop: From Single Shots to Continuous Reasoning

The foundational shift in modern AI engineering is moving from *single-shot LLM calls* (one prompt, one output) to *agentic loops* — iterative cycles where an LLM reasons, acts, observes results, and decides what to do next.

Proxi implements this as a strict **REASON -> DECIDE -> ACT -> OBSERVE -> REFLECT -> LOOP** cycle inside a class called `AgentLoop`. Every turn, the agent:

1. Receives the current conversation history, available tools, and available sub-agents
2. Calls the LLM to produce a structured `ModelDecision` — not a free-form text reply, but a typed decision with four possible outcomes:
  - `RESPOND` — emit a natural language reply to the user
  - `TOOL_CALL` — execute one or more registered tools (filesystem, shell, web search, APIs)
  - `SUB_AGENT_CALL` — delegate a sub-task to a specialized sub-agent with its own separate loop
  - `REQUEST_USER_INPUT` — present the user a structured form (human-in-the-loop gate)
3. Executes the decision (the ACT phase), collects the result (OBSERVE), then feeds everything back into state and loops

The loop continues until the agent emits a `RESPOND` decision, exceeds the `max_turns` budget, or hits a timeout. Per-turn time budgets are configurable via environment variables (`PROXI_BUDGET_TURN_MS`, `PROXI_BUDGET_DECIDE_MS`, `PROXI_BUDGET_ACT_MS`), giving operators control over latency and cost.

A key engineering subtlety is parallel tool execution: when the model returns multiple `TOOL_CALL` entries in one turn, Proxi identifies which tools are safe to execute concurrently and runs them in parallel using asyncio, significantly reducing round-trip latency for complex multi-step tasks.

---

### 2. Multi-LLM Provider Abstraction

A production agent framework cannot be tied to a single LLM provider. Models differ in pricing, capability, privacy guarantees, and context window size. Proxi solves this with a `LLMClient` protocol — a uniform Python interface that all model backends implement. The same agent loop runs unchanged regardless of which provider is active underneath.

Three providers are supported:

- **Anthropic (Claude)** — including claude-3.5-sonnet, claude-3-opus, and the Claude 4.x family (claude-opus-4-6, claude-sonnet-4-6), all with 200,000 token context windows. Claude's tool-use API is adapted to Proxi's internal `ToolSpec` format.
- **OpenAI (GPT-4o, GPT-5 family)** — including gpt-4o, gpt-4o-mini, o1, o3, o4-mini, and the GPT-5 family. OpenAI's structured tool-calling format is used. The o-series models expose a `reasoning_effort` parameter (`minimal`, `low`, `medium`, `high`) that Proxi passes through to control how much compute the model spends on internal reasoning before responding.
- **vLLM** — for self-hosted open-source models. This enables fully offline, private deployments of Proxi using models like LLaMA, Mistral, or Qwen, served locally via the OpenAI-compatible vLLM server.

The model registry tracks context window sizes for all known models (ranging from GPT-3.5's 16K to GPT-5.4's 100 million tokens) and sets compaction thresholds relative to each model's actual limit.

---

### 3. Context Window Management and Compaction

One of the most practical and underappreciated problems in agentic AI is *context window exhaustion*. An agent working on a long task accumulates conversation history, tool call results, and intermediate reasoning across many turns. Eventually, this exceeds the model's input limit — and without a strategy, the agent simply crashes or silently loses memory.

Proxi implements a **head / middle / tail compaction algorithm** inside `ContextCompactor`:

1. **Threshold monitoring** — after every turn, the number of prompt tokens used is tracked. When usage crosses a configurable threshold (default: 85% of the context window), compaction triggers.
2. **Head protection** — the first N messages (default: 3) are always preserved because they contain the original task goal and user intent.
3. **Tool result pruning** — large tool outputs in the middle of history are replaced with truncated previews (default: 200 character previews), stripping bulk data that is no longer needed.
4. **Tail protection** — the most recent ~20,000 tokens of context are always preserved, ensuring the agent has full awareness of what just happened.
5. **Middle summarization** — the messages between the protected head and tail are sent to the LLM in a *cache-safe fork*: the same system prompt and head are used, meaning the provider's prompt cache is reused and only the summary request costs new tokens. The LLM produces a structured summary covering: main task, key constraints, progress completed, important references, and next steps.
6. **Reassembly** — the compacted history becomes: head + summary message + tail.
7. **Structural cleanup** — after compaction, two integrity passes run: orphaned tool_call/tool_result pairs are removed or injected with placeholder values, and role alternation (user/assistant/user/assistant...) is strictly enforced, since provider APIs reject consecutive same-role messages.

This design allows Proxi to run indefinitely-long sessions without hitting context limits, and the cache reuse property means compaction costs significantly less than a full re-prompt.

---

### 4. Persistent Memory Architecture

A one-session agent has no real intelligence advantage over a simple chatbot. True agentic capability requires memory that persists across sessions. Proxi implements a three-tier persistent memory architecture:

**Episodic Memory** stores summaries of past sessions in a SQLite database using FTS5 full-text search. Each episode contains an LLM-generated ~200 word summary, the raw full-text of the session (for searchability), and topic tags. The FTS5 virtual table (backed by insert/update/delete triggers maintaining a shadow index) allows the agent to recall relevant past conversations by topic. This solves the problem of an agent that "forgets" what it has already done for a user across sessions.

**Procedural Memory (Skills)** stores reusable multi-step workflows as Markdown files in an `agentskills.io`-compatible format with YAML frontmatter (version, created_by, use_count). When the agent notices it has solved a problem in a general way, it can call `save_skill` to write the workflow to disk. On future sessions, it can load and reuse this skill rather than re-deriving the solution. This is analogous to procedural memory in cognitive science — knowing *how* to do something without re-learning it.

**User Model** is a structured Markdown file (`USER.md`) capped at approximately 600 tokens (~2400 chars). It tracks user preferences, communication style, environment details, coding conventions, and relationships. The agent periodically receives a *memory nudge* (injected as a transient system message every N turns, configurable via `PROXI_MEMORY_NUDGE_INTERVAL`) that asks it to decide if anything is worth saving. Crucially, the nudge message is transient — it is injected only for the LLM call and then removed from persistent history so it does not pollute the conversation record or cost tokens on future turns. If the model's response to the nudge is detected as only acknowledging the nudge (via a heuristic matching short responses containing nudge-marker phrases), that response is also suppressed and the agent immediately re-decides — preventing the nudge from generating noise in the conversation.

---

### 5. Prompt Engineering and Stability

Good agentic systems require disciplined prompt construction. Proxi's `PromptBuilder` implements a **static -> incremental** structure:

The **system prefix** (containing the global system prompt, the agent's personality/soul file, deterministic tool definitions, and user model) is designed to stay byte-identical across turns whenever possible. A SHA-based cache key invalidates on file modification timestamps (soul, system prompt, user model, API key DB) and tool list changes. Stable system prompts enable provider-side *prompt caching*, where Anthropic and OpenAI charge significantly reduced rates for cached input tokens — a direct cost optimization.

The **chat history** (incremental part) is the conversation sequence and is passed through unmodified. Workspace context (plan.md, todos.md) is intentionally *not* injected automatically — the agent pulls this on demand via tool calls. This avoids inflating the system prefix with volatile content that would break the cache.

For deferred tools (tools not immediately loaded to reduce system prompt size), lightweight stubs (name + description only, no full parameter schemas) are rendered in the system prompt. This tells the LLM what tools exist without the token cost of full schemas, and it can call `search_tools` to load a full spec on demand.

---

### 6. Multi-Agent Orchestration

Many real tasks are too complex for a single agent. Proxi supports **multi-agent delegation**: the primary orchestrator can issue a `SUB_AGENT_CALL` decision, which spins up a specialized `SubAgent` instance with its own isolated execution context. Sub-agents run their own `AgentLoop` bounded by three independent budgets:

- `max_turns` — maximum reasoning iterations
- `max_tokens` — maximum LLM token spend
- `max_time` — maximum wall-clock execution time

Each sub-agent returns a structured `SubAgentResult` with a summary, artifacts (code, files, diffs, answers), a confidence score, a success flag, and optional follow-up suggestions. The parent orchestrator uses this result to decide what to do next.

This architecture enables patterns like: orchestrator decomposes a complex software task -> delegates code writing to a coding sub-agent -> delegates testing to a test-runner sub-agent -> assembles the results. Each delegated task is bounded and predictable.

---

### 7. Tool Systems, Integration Gating, and MCP

Tools are the mechanism by which agents take actions in the world. Proxi's `ToolRegistry` organizes tools into tiers:

- **Live tools** — always loaded into the LLM's tool list on every call (core filesystem, shell, memory, workspace tools)
- **Deferred tools** — loaded on demand when the agent calls `search_tools`, reducing system prompt bloat for large tool catalogs
- **Integration tools** — Gmail, Calendar, Spotify, weather, browser automation, and others, activated by feature flags stored in SQLite

Integration gating is implemented with a **dual-check pattern** for safety: tools for a disabled integration are neither registered in the tool list nor executable if somehow invoked while state is stale. This ensures that disabling an integration is a hard safety boundary, not a soft suggestion.

**MCP (Model Context Protocol)** extends Proxi's tools to external servers. The `MCPClient` communicates with MCP servers via JSON-RPC 2.0 over stdio, with a full circuit breaker implementation: consecutive timeouts increment a counter, and once the threshold is exceeded, the circuit opens and fast-fails all requests for a configurable cooldown period. This prevents a stalled external tool server from blocking the entire agent. Per-request semaphores cap in-flight MCP requests (default: 16) and separate timeout budgets are applied to tool calls (120s) vs. metadata requests (30s).

---

### 8. Gateway, Event Routing, and Multi-Channel Architecture

Proxi's runtime is centered on a **FastAPI/Uvicorn gateway daemon** that starts once and persists in the background. A `LaneManager` maintains isolated `AgentLane` queues — one per active agent/session combination. This design means:

- Multiple users can talk to independent agent instances simultaneously
- The TUI, React frontend, Discord relay, and cron scheduler all connect to the same gateway via HTTP/SSE
- Cron jobs and heartbeats are first-class event sources routed through the same `EventRouter` as user messages
- The gateway manages MCP server availability, session state persistence, and live tool-list refresh when integrations are toggled

State is persisted asynchronously: conversation history is written to `history.jsonl` via a dedicated background `_HistoryWriter` thread that processes writes off the asyncio event loop, preventing disk I/O from blocking agent turns.

---

### Summary of Research Findings

Building Proxi produced several concrete findings relevant to the broader AI engineering field:

- **Agentic loops require explicit budgets** — without turn, token, and time limits, agents can enter pathological loops. All three budgets must be enforced independently.
- **Prompt cache stability is an economic lever** — designing system prompts to be byte-stable across turns reduces API costs substantially. Volatile injections (workspace context, plan files) should be tool-pulled, not auto-injected.
- **Context compaction must be structural, not truncation** — naive truncation destroys task context. Head/tail protection with LLM summarization of the middle preserves reasoning continuity.
- **Memory nudges should be transient** — injecting memory prompts into persistent history pollutes future context and wastes tokens. Transient injection + suppression if the model only acknowledges the nudge is the correct pattern.
- **Dual-gate integration safety** — feature flags at registration time and execution time are both necessary. Relying on only one gate creates race conditions.
- **MCP circuit breakers are essential** — external tool servers fail. Without circuit breakers, a single stalled server can block an entire agent session indefinitely.

</details>

### Archive

<details>
<summary><strong>December 1, 2025 — Research Section Setup</strong></summary>

## December 1, 2025 
Initial setup of this section. Research notes and experiments will begin being added here soon.

Topics will include:

- Machine learning fundamentals  
- Deep learning experiments  
- Transformer and LLM research  
- Applied AI notes  
- Model evaluations and findings  

More research will be added soon.

</details>

---

# 🛠 Projects

### Featured

<details open>
<summary><strong>April 2026 — Proxi — Full-Stack Agentic AI Framework</strong></summary>

## April 2026

**Proxi — Full-Stack Agentic AI Framework**

### Introduction

Proxi is an agentic AI assistant framework built as a Capstone project, designed to make computers more accessible for users who face barriers with traditional interfaces, and to give power users a way to delegate complex multi-step work to an AI that can actually act — not just respond. It is not a wrapper around a chatbot API. It is a complete agent operating system: it reasons, plans, calls tools, manages persistent memory across sessions, delegates to sub-agents, and serves multiple interfaces simultaneously from a single gateway daemon.

The project was built entirely from scratch in Python 3.12 (backend) and TypeScript/React (frontends), with no off-the-shelf agent framework used. Every component — the agentic loop, context compaction, memory system, tool registry, MCP client, gateway routing, and multi-LLM abstraction — was designed and implemented as part of this project.

---

### What Proxi Can Do

- Execute natural language tasks that span filesystem operations, shell commands, web research, and external service APIs
- Connect to Gmail, Google Calendar, Spotify, weather services, and browser automation — all togglable at runtime without restarting
- Remember users across sessions using three-tier persistent memory: past session summaries (episodic), reusable workflows (skills), and user preferences (user model)
- Run indefinitely-long sessions without hitting LLM context limits via automated context compaction
- Handle complex multi-step tasks by delegating sub-tasks to specialized sub-agents with isolated budgets
- Present structured forms to users when human input is required mid-task (human-in-the-loop)
- Connect to any external tool server via MCP (Model Context Protocol) with circuit-breaker protection
- Be accessed via a terminal TUI, a React web frontend, or Discord — all connected to the same persistent gateway

---

### Technical Architecture

**Core Agent Loop (`proxi/core/loop.py`)**  
The `AgentLoop` class implements the REASON → DECIDE → ACT → OBSERVE → REFLECT → LOOP cycle. Each turn, an LLM produces a structured `ModelDecision` typed as one of: `RESPOND`, `TOOL_CALL`, `SUB_AGENT_CALL`, or `REQUEST_USER_INPUT`. Multiple tool calls in a single turn are executed in parallel where safe, reducing latency. Configurable per-turn, per-decide, and per-act time budgets (via environment variables) control resource usage.

**Multi-LLM Backend (`proxi/llm/`)**  
A uniform `LLMClient` protocol abstracts over Anthropic (Claude 3.x and 4.x, 200K context), OpenAI (GPT-4o, GPT-5, o-series up to 100M context on GPT-5.4), and vLLM (self-hosted open-source models). A model registry maps every known model to its context window size, driving compaction threshold calculations automatically.

**Context Compaction (`proxi/core/compactor.py`)**  
When token usage crosses 85% of the context window, a head/middle/tail algorithm triggers: head (first 3 messages) and tail (~20K token equivalent) are protected, large tool results in the middle are truncated to 200-char previews, and the middle is LLM-summarized using a cache-safe fork. Orphaned tool call/result pairs and role alternation violations are repaired structurally after compaction.

**Three-Tier Memory (`proxi/memory/`)**  
Episodic memory: SQLite FTS5 with insert/update/delete triggers maintaining a shadow search index over session summaries and full-text content. Skills memory: agentskills.io-compatible Markdown files with YAML frontmatter, written by the agent when it identifies a reusable workflow. User model: a structured USER.md capped at ~600 tokens, updated via transient memory nudges injected every N turns and suppressed from persistent history after processing.

**Tool Registry and Integration Gating (`proxi/tools/`, `proxi/integrations/`)**  
Tools are tiered as live (always loaded), deferred (loaded on-demand via `search_tools`), and integration (feature-flagged). Disabled integrations are blocked at both registration and execution time. The MCP client (`proxi/mcp/client.py`) uses JSON-RPC 2.0 over stdio with configurable concurrency (semaphore, default 16 in-flight), per-request timeouts (30s metadata / 120s tool calls), circuit breaker (opens after threshold consecutive timeouts, cools down for configurable seconds), and retry logic with backoff.

**Prompt Builder (`proxi/core/prompt_builder.py`)**  
Static system prefix (soul file, global system prompt, tool definitions, user model) is cached via SHA hash keyed on file modification timestamps and tool list shape, enabling provider-side prompt cache reuse. Deferred tool stubs are rendered in the system prompt without full schemas. Chat history is passed through unmodified.

**Gateway and Multi-Channel (`proxi/gateway/`)**  
FastAPI/Uvicorn daemon managed by a PID file and health endpoint. `LaneManager` routes events to per-session `AgentLane` queues. Event sources: HTTP (TUI, frontend), SSE streaming, Discord relay, cron scheduler, heartbeat. History is written asynchronously via a background `_HistoryWriter` thread to keep disk I/O off the asyncio event loop.

**Multi-Agent Orchestration (`proxi/agents/`)**  
Sub-agents implement the `SubAgent` protocol with isolated `max_turns`, `max_tokens`, and `max_time` budgets. Results are returned as structured `SubAgentResult` with summary, artifacts, confidence score, success flag, and follow-up suggestions.

---

### Stack

| Layer | Technology |
|---|---|
| Backend runtime | Python 3.12, asyncio, FastAPI, Uvicorn |
| Data validation | Pydantic v2 |
| Persistence | SQLite (FTS5, feature flags, API key store) |
| LLM providers | Anthropic API, OpenAI API, vLLM |
| External tools | MCP (JSON-RPC 2.0 over stdio) |
| Terminal TUI | TypeScript, Bun, Ink (React for terminals) |
| Web frontend | React, Node.js |
| Discord integration | Node.js relay server |
| Logging / tracing | structlog, custom perf tracing |

---

**Repository:** https://github.com/gkpodder/Proxi

</details>

### Archive

<details>
<summary><strong>December 1, 2025 — Meal Maker</strong></summary>

## December 1, 2025  
First project entry added.

**Meal Maker**  
AI powered cooking app that generates recipes from user provided ingredients, estimates nutrition, and supports chat based recipe creation.

**Website:** https://www.mealmaker.ca  
More AI based projects will be added here as I build them. Some that are built or in progress cannot yet be posted here due to confidential reasons. Will try posting whatever I can. 

</details>

---

# 📖 Useful Resources
These are some great resources I use when doing my research and work. I have included them below along with their authors.

- **Mathematics for Machine Learning**  
  *Marc Peter Deisenroth, A Aldo Faisal, Cheng Soon Ong*  
  Foundations in linear algebra, probability, and optimization.

- **Hands On Machine Learning with Scikit Learn, Keras, and TensorFlow**  
  *Aurélien Géron*  
  Practical ML fundamentals and deep learning workflows.

- **Machine Learning: A Probabilistic Perspective**  
  *Kevin P. Murphy*  
  Core ML theory including Bayesian methods and maximum likelihood.

- **Deep Learning**  
  *Ian Goodfellow, Yoshua Bengio, Aaron Courville*  
  Neural networks, optimization, CNNs, RNNs, and training stability.

- **Dive Into Deep Learning (D2L)**  
  *Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola*  
  Hands on deep learning with notebooks and implementations.

- **Natural Language Processing with Transformers**  
  *Lewis Tunstall, Leandro von Werra, Thomas Wolf*  
  Attention, transformers, and modern NLP systems.

- **Reinforcement Learning: An Introduction**  
  *Richard S. Sutton, Andrew G. Barto*  
  MDPs, policy gradients, value functions, and Q learning.

- **Designing Machine Learning Systems**  
  *Chip Huyen*  
  Real world ML engineering, pipelines, monitoring, and deployment.

---

# 📫 Contact
**LinkedIn:** https://www.linkedin.com/in/ajay-grewal/
