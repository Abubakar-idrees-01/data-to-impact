# 1. What MCP is, and why it exists

## Intuition
Before MCP, every AI app needed its own plugin for every tool: a ChatGPT plugin, a Cursor extension, a Claude
integration, each written differently. With *M* apps and *N* tools, that's M × N integrations.

The **Model Context Protocol** (introduced by Anthropic in November 2024, now an open standard used across the
industry) is a common plug: a tool is wrapped once as an **MCP server**, and every **MCP client** can use it.
M × N becomes M + N. People call it "USB-C for AI".

## The three roles
| role | what it is | here |
|---|---|---|
| **Host** | The app the user talks to | Claude Code in VS Code, Claude Desktop, Cursor |
| **Client** | The host's connector, one per server | Inside Claude Code; also `mcp.Client` in our tests |
| **Server** | A program that offers capabilities | `studyrag-mcp` (this project) |

The host starts the server (for stdio: as a subprocess), the client and server agree on a protocol version, and
the model can then use what the server offers.

## What a server can offer
| primitive | who decides to use it | example here |
|---|---|---|
| **Tools**: functions with a JSON input schema | the **model** calls them when useful | `search_notes`, `check_answer` |
| **Resources**: readable data identified by a URI | the **app or user** attaches them | `notes://documents`, `notes://documents/Lec 2` |
| **Prompts**: reusable message templates | the **user** picks them (often as slash commands) | `quiz_me`, `revise_document` |

The split matters: tools are *model-controlled*, so their descriptions are written for an LLM to read and choose
from. Resources and prompts are *application/user-controlled*.

## Transports
- **stdio**: the host launches the server and exchanges newline-delimited JSON-RPC over stdin/stdout. Local,
  simple, no network. This is how Claude Code runs `studyrag-mcp`.
- **Streamable HTTP**: the server listens on a URL; used for remote or shared servers. `studyrag-mcp --http`.

## What this project does with it
Your study material (project 1's index of slides, PDFs and notes) and your evaluation results (project 2) become
tools that any assistant can use. The assistant is the language model, and the server is its librarian: it finds
and returns the right passages, and fact-checks answers against them.

## Interview questions
- *MCP vs. function calling?* Function calling is how one model API lets a model request a function call. MCP
  is how tools are *packaged and discovered* across apps: the host turns MCP tools into function definitions for
  whatever model it uses.
- *Why not just a REST API?* MCP standardizes discovery (`tools/list` with schemas and descriptions written for
  models), version negotiation, and the difference between a tool failing and the protocol failing.
- *Who controls each primitive?* Tools: the model. Resources: the application. Prompts: the user.
