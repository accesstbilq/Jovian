from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import SystemMessage
from langchain.agents.middleware import before_model
from langchain_core.messages import HumanMessage

from dataclasses import dataclass
from ..helpers.system_prompt import SYSTEM_PROMPT

@dataclass
class AgentContext:
    rag_context: str      # Actual knowledge content
    has_rag_data: bool


def create_rag_context_middleware():
    """
    Middleware that injects RAG context into the system message.
    This runs BEFORE the model is called, so context is available.
    """
    
    @before_model
    def inject_rag_context(request, runtime: ToolRuntime[AgentContext]) -> str:
        """Inject RAG context into system message dynamically"""
        
       # ✅ CORRECT: Access context from request.runtime.context
        rag_context = runtime.context.rag_context
        has_rag_data = runtime.context.has_rag_data
        
        print(f"[MIDDLEWARE] Injecting RAG context ({len(rag_context)} chars, has_data={has_rag_data})")
        
        # Build dynamic context instruction
        if has_rag_data:
            rag_instruction = f"""
            ## RAG CONTEXT - USE THIS INFORMATION:

            {rag_context}

            ---

            Instructions for using the above context:
            - Answer the client's question using ONLY the provided RAG context above
            - Be specific and cite details from the knowledge base
            - Never speculate beyond what's in the context
            - If the question isn't covered, say so professionally
            """
        else:
            rag_instruction = """
            ## NO KNOWLEDGE BASE AVAILABLE

            You do not have specific information for this query in our knowledge base.
            - Do NOT make up information or guess
            - Politely inform the client: "We don't have specific information on this in our system"
            - Suggest: "I'd recommend contacting our team directly for accurate details"
            - Keep the tone professional and helpful
            """
        
        # Modify the system message to include RAG context
        # original_system_msg = request.system_message
        
        # if original_system_msg:
        #     # Append RAG context to existing system message
        #     new_content = str(original_system_msg.content) + "\n" + rag_instruction
        #     new_system_msg = SystemMessage(content=new_content)
        # else:
        #     # Create new system message with RAG context
        #     new_system_msg = SystemMessage(content=rag_instruction)
        
        # Update request with modified system message
        return {
            "messages": [SystemMessage(content=rag_instruction)],
        }
        
    
    return inject_rag_context


def create_message_agent(model, checkpointer):
    """
    Create a message generation agent that grounds responses in RAG context.
    
    When RAG context is available: Answer ONLY from provided knowledge
    When RAG context is empty: Decline to answer and request more info
    
    Args:
        model: Language model (ChatOpenAI, etc.)
        checkpointer: Optional checkpointer for persistence
    
    Returns:
        Compiled agent with context-aware grounding
    """

    # Create middleware that injects RAG context
    rag_middleware = create_rag_context_middleware()
    
    # Create agent with context schema
    agent = create_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        context_schema=AgentContext,
        middleware=[rag_middleware],
        checkpointer=checkpointer,
    )
    
    return agent