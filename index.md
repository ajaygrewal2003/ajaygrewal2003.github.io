---
layout: default
---

# Ajay Singh Grewal
### Machine Learning • Artificial Intelligence • Software Engineer

Welcome to my personal site. I am currently studying machine learning, AI, and building useful software. I will be sharing my research notes, learning progress, and projects as I continue developing in this field. This page will grow as I move through my AI learning roadmap.

## Quick Navigation

### Research
- [April 2026 — Engineering Agentic AI Systems](#research-april-2026)
- [March 2026 — Real-Time Voice AI Systems](#research-march-2026)
- [December 1, 2025](#research-december-2025)

### Projects
- [April 2026 — Proxi](#project-april-2026)
- [March 2026 — AI Phone Agent Platform](#project-march-2026)
- [December 1, 2025 — Meal Maker](#project-december-2025)

### Other
- [Useful Resources](#-useful-resources)
- [Contact](#-contact)

## Latest Updates

- Research: April 2026 — Engineering Agentic AI Systems: A Research Report on Building Proxi
- Project: April 2026 — Proxi — Full-Stack Agentic AI Framework

---

# 📚 Research

<a id="research-april-2026"></a>
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

---

<a id="research-march-2026"></a>
## March 2026 — Real-Time Voice AI Systems: A Research Report on Building a Production AI Phone Agent Platform

### Introduction

This research report documents the engineering and AI systems work carried out during the design and build of a full-stack, production-grade AI phone agent platform built for businesses. The platform allows any business to deploy AI-powered phone agents — capable of handling inbound and outbound calls, booking appointments, running automated dialing campaigns, retrieving business knowledge, and integrating with external services — all in real time over standard telephone infrastructure.

What made this project technically demanding was not the use of any single AI model, but the orchestration of several AI and signal processing systems simultaneously under the strict latency requirements of a live phone call. When a person calls a business number, they expect a response within one to two seconds. Every architectural decision — from audio encoding format to embedding model selection to how prompts are constructed — was made under that constraint.

The research focuses on eight core technical areas requiring significant study and original implementation work: real-time bidirectional audio streaming, low-latency speech synthesis with interrupt handling, acoustic voicemail detection, retrieval-augmented generation for voice, visual flow graphs as AI behavior programs, MCP tool integration during live calls, multi-tenant agent configuration and outbound campaign systems, and post-call intelligence.

---

### 1. Real-Time Bidirectional Audio Streaming

The foundation of the system is a three-way WebSocket pipeline: Twilio (the telephony carrier) ↔ the backend ↔ OpenAI Realtime API.

When a call arrives, Twilio's webhook fires a POST request to the backend, which returns a TwiML document instructing Twilio to open a Media Stream WebSocket. Simultaneously, the backend opens a second WebSocket to the OpenAI Realtime API. From this point, two concurrent async tasks run in parallel — one forwarding audio from Twilio into OpenAI, and another forwarding AI-generated text responses from OpenAI out to the synthesis pipeline.

The audio encoding format is G.711 μ-law at 8kHz — the standard encoding for PSTN telephone calls. Because Twilio transmits raw μ-law and the OpenAI Realtime API accepts it directly, no transcoding is needed on the inbound path. A custom μ-law decoder was implemented in Python to produce signed 16-bit PCM samples for the signal analysis pipeline.

OpenAI's Realtime API operates in text modality: it receives μ-law audio, performs internal speech-to-text and language understanding, and returns text responses. Routing audio input through the Realtime API while routing audio output through ElevenLabs was a deliberate architectural decision — it gave higher-quality, more controllable speech synthesis than the Realtime API's native audio output, at the cost of slightly more pipeline complexity.

A per-call session store tracks all active sessions in memory. Each session record holds the Twilio and OpenAI WebSocket handles, token usage metrics, synthesis tracking state for interrupt support, voicemail state machine state, call direction, contact metadata, and a full ordered transcript event log.

---

### 2. Speech Synthesis with Interrupt Handling

ElevenLabs Flash v2.5 is used for text-to-speech output. The Flash model produces audio in milliseconds, making it suitable for telephony latency requirements. The backend requests G.711 μ-law output directly from ElevenLabs, eliminating any runtime audio transcoding dependency.

Audio is streamed back to Twilio in fixed-size frames matching the expected packetization rate for telephony — approximately 20ms of audio per frame.

A core engineering problem in voice AI is interruption: if the user begins speaking while the AI is still talking, the AI should stop immediately and listen. This was solved with a per-synthesis request identifier. Every synthesis session is tagged with a unique ID at creation time. Before each frame is sent to Twilio, the code checks whether the active synthesis ID still matches — if the user interrupts, the active ID is cleared, and the next frame check causes the synthesis loop to exit immediately. The ElevenLabs HTTP response stream is also explicitly closed on interrupt, cutting off any further incoming audio at the network level.

A text preprocessing step converts structured data in AI responses — currency values, numerals, abbreviations — into natural spoken language before synthesis, improving the quality and naturalness of telephone speech significantly. Sentence buffering accumulates partial tokens and dispatches them to synthesis only at sentence boundaries, preventing choppy output from mid-sentence completions.

---

### 3. Acoustic Voicemail Detection via Signal Processing

A persistent engineering challenge in outbound calling is voicemail detection: when an AI agent dials a contact and the call connects, it must determine within seconds whether it is speaking with a human or has reached a voicemail system, and act accordingly. Twilio's built-in Answering Machine Detection was bypassed for latency reasons, requiring a fully custom detection system.

The custom implementation uses a dual-signal strategy combining acoustic analysis with natural language pattern recognition.

**Acoustic beep detection** uses the Goertzel algorithm — a computationally efficient discrete Fourier transform variant that calculates power at a single target frequency without computing a full FFT. Voicemail beep tones fall in a known frequency band. For each incoming telephony audio frame, the system decodes μ-law bytes to PCM samples, applies Goertzel at several target frequencies and several noise-guard frequencies, and computes a tonal dominance ratio. Frames where target-frequency energy significantly exceeds background noise are flagged as probable beeps.

**Transcript pattern matching** analyzes each speech-to-text fragment against sets of voicemail-indicating phrases ("leave a message", "not available", "after the beep", "mailbox") and human-indicating phrases ("hello", "yes", "speaking", "who is this"). Each match adjusts a continuous confidence score using a weighted update function. Long monologue-style transcripts without questions also incrementally raise voicemail confidence.

A state machine governs behavior across distinct phases: initial state, human candidate, voicemail candidate, waiting for beep, and message phase. A background watchdog task monitors phase transitions and triggers a timeout fallback if the beep expectation window expires without a confirmed beep, preventing the call from stalling indefinitely.

---

### 4. Retrieval-Augmented Generation for Voice

Businesses need their phone agents to answer questions about their specific operations: product inventory, appointment availability, pricing, service policies. This requires a knowledge retrieval system capable of responding to a natural language query during a live call under strict latency constraints.

The RAG system is built on PostgreSQL with the pgvector extension, which stores and queries high-dimensional float vectors inside the relational database. Two separate embedding stores handle different data types.

Document embeddings capture chunks of text extracted from uploaded PDFs, Word documents, and text files. Table embeddings capture rows from uploaded CSV and Excel files, including multi-sheet workbooks. Each spreadsheet row is converted to a natural-language text representation before embedding, which allows the agent to semantically search tabular data — price lists, directories, inventory — using natural language questions.

All embeddings are produced by sentence-transformers/all-MiniLM-L6-v2, a 384-dimensional model well-suited to semantic similarity tasks over short passages.

Query-time retrieval uses cosine similarity for the semantic component, combined with a BM25-style lexical scoring function that gives weight to exact term matches. Final results are ranked by a weighted combination of both scores. This hybrid approach handles paraphrase queries well (where semantic similarity excels) and precise term lookups well (where lexical scoring excels). The entire retrieval path is implemented with the async database client to avoid blocking the audio pipeline.

---

### 5. Visual Flow Graphs as AI Behavior Programs

Rather than requiring businesses to write system prompts from scratch, the platform includes a visual drag-and-drop flow builder where users design call workflows as node-edge graphs. The backend compiles these graphs into structured natural language instructions that become the AI's system prompt at call time.

Nodes in the graph represent logical operations: entry points for inbound and outbound calls, custom instruction blocks, tool invocations, data source references, conditional branches, call transfer directives, error handlers, variable assignments, event logging steps, and call-end actions. Edges connect nodes and can carry optional conditions based on time of day, detected caller intent, or caller properties.

The flow executor traverses the node-edge graph recursively at the start of each call, collecting instruction strings from each node and assembling them into the AI's prompt. Conditional nodes branch the traversal based on call context. A depth limit prevents infinite traversal in malformed graphs.

A more advanced execution mode treats the flow as a flexible action checklist rather than a rigid script. This mode generates guidance instructing the AI to complete all required actions in any order that serves the caller naturally — allowing the conversation to flow freely while ensuring every configured step is executed. A progress tracker records which actions have been completed and which remain pending, enabling detection of incompletely executed workflows.

---

### 6. MCP Tool Integration in Live Voice Sessions

During a call, the AI may need to take real-world actions: check Google Calendar for availability, add a record to a Google Sheet, or send a confirmation email via Gmail. These capabilities are registered as function call tools inside the OpenAI Realtime API session using the Model Context Protocol (MCP) pattern.

A central tool dispatcher handles Google Calendar (OAuth 2.0, availability queries, event creation with timezone support), Google Sheets (row reads and writes), and Gmail (send with attachment support, governed by a configurable attachment policy per agent).

A key engineering challenge is that external API calls are synchronous operations within an asynchronous audio stream. When the AI calls a calendar availability function, the audio pipeline cannot simply discard incoming user speech during that window. An execution-in-progress flag signals the audio forwarding loop to buffer incoming speech frames during tool execution rather than drop them, so the user's words are not lost.

All OAuth credentials are stored encrypted at the field level in the database and decrypted only at the moment of execution, keeping credentials out of process memory except during the brief execution window. Tool calls are wrapped in a per-call timeout, and failures return structured errors that allow the AI to gracefully inform the caller rather than hanging the call.

---

### 7. Multi-Tenant Configuration and Outbound Campaign Systems

The platform is designed as a multi-tenant SaaS system. Each user account can configure multiple independent AI agents, each with its own dedicated phone number, voice selection, separate inbound and outbound system prompts, business hours schedule, and its own assigned knowledge sources and tool integrations.

The outbound campaign system introduces a distinct set of AI coordination challenges. Campaign workers run as background async tasks, iterating over contact lists and initiating outbound calls. Each contact tracks attempt count, retry delay, maximum attempts, and outcome. Contact records can carry custom fields that are interpolated into campaign scripts at the time of each call using a variable substitution system.

A callback recognition system handles the scenario where a contact from a previous outbound campaign calls back inbound. The system looks up recent campaign contacts by normalized phone number within a configurable time window, restores the prior campaign context, and gives the inbound AI agent awareness of the previous outbound interaction — enabling natural conversational continuity without the caller needing to re-explain why they are calling.

---

### 8. Post-Call Intelligence and Cost Attribution

After every call ends, a post-processing pipeline generates structured intelligence on the conversation. The complete transcript — assembled from time-ordered events captured throughout the session — is submitted to GPT-4o-mini requesting a two-sentence summary, a sentiment classification (positive, neutral, or negative), and detection of any user-configured success criteria that were met during the call.

Success criteria are defined per agent and represent business goals such as "appointment booked" or "pricing question answered". The model's output is validated against the database before being stored, preventing hallucinated criteria from being logged.

All cost components are metered and attributed at the individual call level. OpenAI Realtime API usage is tracked across text input, audio input, and cached token categories. ElevenLabs usage is tracked by character count. Post-call summarization tokens are tracked separately. Twilio voice duration is fetched from the Twilio REST API after the call. This per-component attribution makes it possible to identify cost drivers by call type and make informed optimization decisions.

---

### Summary of Research Findings

Building this platform produced several concrete findings relevant to production voice AI systems:

- **Three-way WebSocket concurrency is the fundamental primitive** — voice AI is not a request-response system. The entire architecture must be designed around concurrent, non-blocking message loops from the first design decision.
- **μ-law passthrough eliminates transcoding** — routing G.711 telephony audio natively through the pipeline without PCM conversion removes a latency-adding step. This is made possible by synthesis providers that support native telephony output formats.
- **Interrupt handling requires a synthesis request identifier, not a boolean flag** — a shared boolean "is speaking" creates race conditions between concurrent async tasks. A unique request ID checked per frame is the correct low-level primitive for safe cancellation.
- **Goertzel is the right algorithm for real-time tonal detection** — a full FFT per audio frame is wasteful at telephony rates. The Goertzel algorithm computes power at a single target frequency in O(n) without a transform, making it practical for per-frame inline signal analysis.
- **Hybrid semantic and lexical RAG outperforms either alone** — cosine similarity alone misses precise term lookups; BM25 alone misses paraphrase queries. A weighted hybrid scorer handles the full range of business knowledge retrieval more reliably than either approach in isolation.
- **Visual flow graphs decouple business logic from prompt engineering** — giving businesses a visual way to configure call behavior makes the system accessible to non-technical operators while leaving the AI complexity unchanged underneath.
- **Per-call cost attribution at the component level is necessary for SaaS economics** — aggregate billing is insufficient; per-call, per-component metering is required to understand cost drivers and make informed optimization decisions.

---

<a id="research-december-2025"></a>

## December 1, 2025 
Initial setup of this section. Research notes and experiments will begin being added here soon.

Topics will include:

- Machine learning fundamentals  
- Deep learning experiments  
- Transformer and LLM research  
- Applied AI notes  
- Model evaluations and findings  

More research will be added soon.

---

# 🛠 Projects

<a id="project-april-2026"></a>
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

---

<a id="project-march-2026"></a>
## March 2026 — AI Phone Agent Platform (Confidential)

### Introduction

This project is a production-grade AI phone agent platform built for businesses that want to automate inbound and outbound telephone interactions using conversational AI. An agent deployed on the platform can answer inbound calls, make outbound calls, hold natural multi-turn conversations, retrieve business knowledge, book appointments, send follow-up emails, run automated dialing campaigns against contact lists, transfer calls to human agents, and detect voicemail — all without human involvement.

The system is built entirely from scratch, integrating Twilio's telephony infrastructure, OpenAI's Realtime API for live speech understanding, ElevenLabs for high-quality voice synthesis, and PostgreSQL with pgvector for semantic knowledge retrieval. A Next.js frontend provides the full operator dashboard for configuring agents, monitoring calls, managing contacts and campaigns, uploading knowledge bases, connecting external integrations, and reviewing per-call analytics.

The platform is a complete SaaS product — not a demo or prototype — with subscription billing, multi-tenant user accounts, per-agent phone number assignment, and a visual drag-and-drop flow builder for configuring call behavior without requiring code.

---

### What the Platform Can Do

- Handle inbound phone calls with a fully conversational AI — answering questions, taking messages, and transferring to staff
- Run outbound calling campaigns against contact lists with retry logic, drip sequencing, and voicemail drop
- Answer caller questions by searching uploaded business documents and spreadsheets in real time using RAG
- Check and book appointment slots on Google Calendar with timezone-aware availability queries
- Send confirmation emails via Gmail, including file attachments governed by a configurable policy
- Read and write data in Google Sheets for real-time integration with business records
- Detect voicemail acoustically using the Goertzel algorithm combined with NLP transcript analysis
- Transfer calls mid-conversation to a human agent with context continuity
- Support multiple AI voice profiles across English, Hindi, Spanish, French, German, and other languages
- Generate AI summaries, sentiment labels, and success criteria detection for every completed call
- Provide per-call cost attribution across all AI and telephony provider components
- Deliver an analytics dashboard covering call volume, sentiment trends, and campaign performance

---

### Technical Architecture

**Telephony Pipeline**
Twilio webhooks receive inbound and outbound call events. The backend returns TwiML instructing Twilio to open a Media Stream WebSocket. A three-way async pipeline runs in parallel bridging Twilio, the backend, and OpenAI's Realtime API. Audio travels as G.711 μ-law at 8kHz in both directions. OpenAI processes incoming speech and returns text responses and tool call events; ElevenLabs converts those responses to μ-law audio streamed back to Twilio in fixed telephony-rate frames.

**AI Speech Understanding (OpenAI Realtime API)**
The Realtime API operates in text modality — receiving raw telephony audio and emitting natural language responses and tool call instructions. Tool executions are dispatched to external APIs and results injected back into the session. Token usage across input, output, and cached categories is tracked per session for cost attribution.

**AI Voice Synthesis (ElevenLabs Flash v2.5)**
Text responses are sentence-buffered before being dispatched to ElevenLabs, which returns native μ-law audio with no transcoding step. A per-synthesis request identifier enables interrupt cancellation at the frame level — when the user interrupts, the identifier is cleared and the active synthesis loop exits within its next frame cycle.

**Knowledge Retrieval / RAG**
Uploaded documents and tabular data files are embedded using sentence-transformers/all-MiniLM-L6-v2 and stored in PostgreSQL via pgvector. Queries at call time use a hybrid ranking of cosine similarity (semantic relevance) and BM25-style lexical scoring (term precision). Retrieval is fully async to avoid blocking the audio pipeline.

**Voicemail Detection**
Acoustic detection uses the Goertzel algorithm to analyze tonal power at voicemail beep frequencies in each audio frame. NLP pattern matching analyzes speech transcripts for voicemail and human-response cues. A continuous confidence score updates probabilistically on each signal event. A state machine drives behavior through distinct phases, with a background watchdog enforcing a timeout fallback.

**Visual Flow Engine**
A drag-and-drop node-edge flow builder in the frontend produces JSON workflow graphs. The backend traverses these graphs at call time to assemble the AI's system prompt. A flexible execution mode treats the flow as an action checklist, allowing natural conversation while tracking completion of every configured step.

**External Tool Integration (MCP)**
Google Calendar, Google Sheets, and Gmail are registered as OpenAI function call tools. OAuth credentials are stored field-level encrypted and decrypted only at execution time. An in-progress flag buffers incoming user audio during external API calls to prevent dropped speech. Tool calls carry timeouts with graceful error handling.

**Outbound Campaign System**
Background async workers manage contact lists, retry queues with configurable delay and attempt limits, and per-contact outcome tracking. Campaign scripts support variable interpolation from contact records. A callback recognition system restores prior campaign context when a previously-called contact calls back inbound.

**Post-Call Intelligence**
After each call, GPT-4o-mini analyzes the ordered conversation transcript and returns a structured summary, sentiment classification, and matched success criteria. Results are validated before storage. All cost components across all provider APIs are computed and logged per call.

**Frontend (Next.js 14, TypeScript)**
Full SaaS operator dashboard: agent configuration, visual flow builder, document and data source management, OAuth integration setup, contact and campaign management, call log review, analytics, phone number management, and billing.

---

### Stack

| Component | Technology |
|---|---|
| Backend runtime | Python 3.12, FastAPI, Uvicorn, asyncio |
| Telephony | Twilio Voice, Twilio Media Streams WebSocket |
| AI speech understanding | OpenAI Realtime API |
| AI voice synthesis | ElevenLabs Flash v2.5 |
| Post-call intelligence | OpenAI GPT-4o-mini |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector database | PostgreSQL + pgvector |
| Relational database | PostgreSQL (async ORM) |
| Google integrations | Google Calendar API, Gmail API, Google Sheets API (OAuth 2.0) |
| Frontend | Next.js 14, React, TypeScript, Tailwind CSS |

---

<a id="project-december-2025"></a>

## December 1, 2025  
First project entry added.

**Meal Maker**  
AI powered cooking app that generates recipes from user provided ingredients, estimates nutrition, and supports chat based recipe creation.

**Website:** https://www.mealmaker.ca  
More AI based projects will be added here as I build them. Some that are built or in progress cannot yet be posted here due to confidential reasons. Will try posting whatever I can.

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
