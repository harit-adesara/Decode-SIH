import os
import logging
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

logger = logging.getLogger("bharatswasthya.checkpoint")

checkpointer = MemorySaver()