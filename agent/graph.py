from langgraph.graph import StateGraph, START, END
from .state import Data
from .agent import healthcare_agent
from .nodes import send_sms_node, end_call_node
from .checkpoint import checkpointer


def route_after_agent(state: Data) -> str:
    """Route to SMS summary and end-call finalization if the call has concluded."""
    if state.get("call_ended"):
        return "send_sms"
    return END


graph = StateGraph(Data)

graph.add_node("agent", healthcare_agent)
graph.add_node("send_sms", send_sms_node)
graph.add_node("end_call", end_call_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "send_sms": "send_sms",
        END: END,
    },
)
graph.add_edge("send_sms", "end_call")
graph.add_edge("end_call", END)

workflow = graph.compile(checkpointer=checkpointer)
