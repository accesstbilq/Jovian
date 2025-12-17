import numpy as np
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import interrupt, Command, RetryPolicy
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_openai import OpenAIEmbeddings
# from langchain_core.utils.math import cosine_similarity
# from langchain_community.vectorstores import Chroma  # or your vectorstore
# from langchain.tools.retriever import create_retriever_tool

from typing import Optional, Annotated, Literal, Any, Dict
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from pathlib import Path
import operator
from dataclasses import dataclass


BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHROMA_DB_PATH = BASE_DIR / "jovian/chroma_db"
COLLECTION_NAME = 'project_portfolio'

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

class RouteQuery(BaseModel):
    """Route user query for agency knowledge & execution."""

    intent: Literal[
        "knowledge_search",
        "bug_or_issue",
        "how_it_works",
        "comparison",
        "task_request",
        "general_chat"
    ]

    urgency: Literal["low", "medium", "high"]
    topic: str
    summary: str

@dataclass
class AgentContext:
    rag_context: str      # Actual knowledge content
    has_rag_data: bool


def get_vectorstore() -> Chroma:
    """
    Re-open the existing Chroma collection using the same settings
    used when creating the embeddings.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DB_PATH),
    )
    return vectorstore


def create_supervisor_agent(llm, message_agent, checkpointer):
    """
    Create a supervisor agent that routes tasks to specialist agents.
    
    Args:
        llm: Language model to use for routing decisions
        message_agent: Agent for message generation
        checkpointer: Checkpoint saver for graph persistence
    
    Returns:
        Compiled supervisor graph
    """

    # ✅ FIXED State - Clean conversation tracking
    class SupervisorState(TypedDict):
        messages: Annotated[list, operator.add]  # Full history
        current_conversation: list              # Current turn only
        intent: str
        classification: Dict[str, Any]
        task_description: str
        rag_data: Optional[list]
        user_message: str
        response_generated: bool                # ✅ CRITICAL: Prevent duplicates
        final_response: str
        goto: str



    ROUTE_MAP = {
        "technical_capability": [
            "What tech stack do you use?",
            "Do you know React?",
            "How do you handle scaling?",
            "Is your code secure?",
            "Do you use AWS or Azure?"
        ],
        "engagement_hiring": [
            "Can I hire a developer?",
            "What is your hourly rate?",
            "Do you have a dedicated team?",
            "How much does it cost?",
            "Are your developers available now?"
        ],
        "process_communication": [
            "How do we communicate?",
            "Do you sign an NDA?",
            "What is your project management process?",
            "Can I get a quote?",
            "Do you use Slack?"
        ],
        "business_trust": [
            "Have you worked with startups?",
            "Show me your case studies.",
            "Why should I trust you?",
            "What are your reviews like?",
            "Have you built something like this before?"
        ],
        "general_chat": [
            "Hi",
            "Hello",
            "Who are you?",
            "Good morning"
        ]
    }

    # Pre-compute embeddings for your route map (Do this ONCE when app starts)
    route_vectors = {
        key: embeddings.embed_documents(sentences) 
        for key, sentences in ROUTE_MAP.items()
    }


    # def fast_semantic_router(state):
    #     query = state["question"]
    #     query_vector = embeddings.embed_query(query)
        
    #     # Compare input vector to all route vectors
    #     scores = {}
    #     for category, vectors in route_vectors.items():
    #         # Get max similarity for this category
    #         similarity = cosine_similarity([query_vector], vectors)[0]
    #         scores[category] = np.max(similarity)
        
    #     # Get the category with the highest score
    #     best_category = max(scores, key=scores.get)
        
    #     # Threshold check: If similarity is too low, default to general search or chat
    #     if scores[best_category] < 0.7: 
    #         return "general_chat" # Or a "fallback_search"
            
    #     return best_category




    def read_message(state: SupervisorState) -> SupervisorState:
        """✅ FIXED: Extract ONLY latest user message"""
        # Get ONLY the last user message
        input_messages = state.get("messages", [])
        if input_messages:
            last_msg = input_messages[-1]
            user_message = last_msg.content if hasattr(last_msg, 'content') else str(last_msg)
        else:
            user_message = "Hello"
        
        return {
            "user_message": user_message,
            "current_conversation": [HumanMessage(content=user_message)],  # Clean slate
            "messages": input_messages,
            "response_generated": False  # Reset for new turn
        }

    def intent_classifier(state: SupervisorState) -> Command[Literal["rag_executor", "general_message"]]:
        """Classify user intent using structured output"""

        print("!!! ENTER INTENT CLASSIFIER NODD !!!")


        user_message = state.get("user_message", "")

        classification_prompt = """
        You are an intent classifier for a digital agency AI assistant.

        Classify the user's message into exactly ONE intent:

        - knowledge_search: asking about projects, technologies, CMS, services, stacks, agency work
        - how_it_works: conceptual or explanatory questions
        - comparison: comparing technologies, frameworks, CMS, tools
        - task_request: asking the AI to generate, build, analyze, or perform a task
        - general_chat: greetings, thanks, casual or unclear messages

        Return ONLY the intent name.
        No explanation.
        No extra text.

        """

        # Use with_structured_output for structured output
        structured_llm = llm.with_structured_output(RouteQuery)
        
        classification = structured_llm.invoke([SystemMessage(content=classification_prompt), HumanMessage(content=user_message)])

        # ✅ CRITICAL FIX: Convert Pydantic object to dict
        classification_dict = classification.model_dump()  # Pydantic v2
        # Fallback for Pydantic v1:
        # classification_dict = classification.dict() 
        
        intent = classification_dict.get("intent", "general_chat")

        if intent in ["knowledge_search", "comparison", "how_it_works", "task_request"]:
            goto = "rag_executor"
        else:
            goto = "general_message"

        print("### INTENT FOUND #####", intent)

        return {
            "intent": intent,
            "classification": classification_dict,
            "goto": goto
        }

    def rag_executor(state: SupervisorState) -> SupervisorState:
        """Execute RAG retrieval based on intent"""
        # Build search query from classification
        classification = state.get('classification', {})
        query = f"{classification.get('intent', '')} {classification.get('topic', '')}"

        print("### ENTER RAG EXECUTOR NODD ###", query)

        try:
            vectorstore = get_vectorstore()
            
            base_retriever = vectorstore.as_retriever(
                search_type="mmr", 
                search_kwargs={
                    "k": 6,          # How many unique docs to return per query
                    "fetch_k": 20,   # Pool of docs to select from
                }
            )

            retrieval_llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

            # MultiQueryRetriever
            advanced_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=retrieval_llm
            )

            # Invoke returns List[Document]
            results = advanced_retriever.invoke(query)

            print(f"[RAG] Retrieved {len(results)} documents")

            return {"rag_data": results}           
        
        except Exception as e:
            print(f"[RAG Error] {str(e)}")
            return {"rag_data": []}

    def general_message(state: SupervisorState) -> SupervisorState:
        """
        Generate response using message agent with RAG context grounding.
        
        Flow:
        1. Extract RAG data from state
        2. Build context string from RAG documents
        3. Create AgentContext with grounding info
        4. Invoke agent with context
        5. Extract and return response
        """
        messages = state.get("messages", [])
        search_results = state.get("rag_data", [])


        print(f"[MESSAGE GENERATOR] Processing {len(messages)} messages with {len(search_results)} RAG docs")
        
        try:
            # Build RAG context from documents
            rag_context = ""
            has_rag_data = len(search_results) > 0
            
            if has_rag_data:
                # Combine document content with metadata for richer context
                context_parts = []
                for doc in search_results[:5]:  # Use top 5 docs
                    category = doc.metadata.get("category", "General")
                    content = doc.page_content
                    context_parts.append(f"[{category}] {content}")
                
                rag_context = "\n\n".join(context_parts)
                print(f"[MESSAGE GENERATOR] RAG Context (length: {len(rag_context)}):")
                print(rag_context[:200] + "...\n")
            else:
                print("[MESSAGE GENERATOR] ⚠️ NO RAG DATA - Agent will not hallucinate")
                rag_context = ""

            # Invoke agent with context - THIS IS WHERE GROUNDING HAPPENS
            result = message_agent.invoke(
                {"messages": messages},
                context=AgentContext(
                    rag_context=rag_context,
                    has_rag_data=has_rag_data
                )
            )
            
            # Extract the response
            if hasattr(result, "messages") and result.messages:
                # Agent returns state with messages
                last_message = result.messages[-1]
                agent_response = last_message.content if hasattr(last_message, "content") else str(last_message)
            elif hasattr(result, "content"):
                agent_response = result.content
            elif isinstance(result, dict):
                agent_response = result.get("messages", [])
                if agent_response and hasattr(agent_response[-1], "content"):
                    agent_response = agent_response[-1].content
            else:
                agent_response = str(result)
            
            print(f"[MESSAGE GENERATOR] Generated response: {agent_response}")
            
            return {
                **state,  # Preserve everything
                "messages": state["messages"] + [AIMessage(content=agent_response)],  # Append to history
                "current_conversation": [AIMessage(content=agent_response)],  # Current turn
                "final_response": agent_response,
                "response_generated": True,  # ✅ PREVENTS DUPLICATES!
                "task_description": "Response complete"
            }
            
        except Exception as e:
            print(f"[MESSAGE GENERATOR ERROR] {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "messages": [AIMessage(content=f"Error processing request: {str(e)}")]
            }
    
    def router_node(state: SupervisorState) -> SupervisorState:
        """✅ FIXED: Router returns dict, not str"""
        goto = state.get("goto", "general_message")
        print(f"[ROUTER] Processing state → {goto}")
        
        # ✅ LangGraph nodes ALWAYS return dict
        return {
            "task_description": f"Routed to {goto}",
            "routing_decision": goto
        }
    
    # Build the graph using StateGraph
    workflow = StateGraph(SupervisorState)

    workflow.add_node("read_message", read_message)
    workflow.add_node("intent_classifier", intent_classifier)
    workflow.add_node("rag_executor", rag_executor)
    workflow.add_node("general_message", general_message)
    workflow.add_node("router", router_node)

    # ========================================
    # ✅ PERFECT LINEAR FLOW - NO DUPLICATES!
    # ========================================
    workflow.add_edge(START, "read_message")
    workflow.add_edge("read_message", "intent_classifier")
    workflow.add_edge("intent_classifier", "router")
    
    # Router decides path
    workflow.add_conditional_edges(
        "router",
        lambda state: state.get("routing_decision", "general_message"),
        {
            "rag_executor": "rag_executor",
            "general_message": "general_message"
        }
    )
    
    # All paths converge ONCE to final response
    workflow.add_edge("rag_executor", "general_message")
    workflow.add_edge("general_message", END)

    # Compile with checkpointer
    supervisor_graph = workflow.compile(checkpointer=checkpointer)
    return supervisor_graph