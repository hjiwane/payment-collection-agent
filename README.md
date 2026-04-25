# Payment Collection AI Agent

A production-style conversational payment collection agent built with a strict 6-layer architecture, deterministic policy gates, LangGraph state management, regex-first extraction, API tool calling, safe response templates, and structured evaluation.

The agent follows this required interface:

```python
class Agent:
    def next(self, user_input: str) -> dict:
        return {"message": "..."}