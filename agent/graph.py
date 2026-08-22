from langgraph.graph import StateGraph,START,END
from .state import Data
from .agent import healthcare_agent
from .nodes import send_sms_node, end_call_node
from .checkpoint import checkpointer

graph=StateGraph(Data)

graph.add_node("agent",healthcare_agent)
graph.add_node("sms", send_sms_node)
graph.add_node("end_call", end_call_node)

graph.add_edge(START,"agent")
graph.add_edge("agent","sms")   
graph.add_edge("sms","end_call")
graph.add_edge("end_call",END)

workflow=graph.compile(checkpointer=checkpointer)
