import os
import json
from flask import Flask, request, Response, jsonify, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv

# ==========================================================
# Load API Keys
# ==========================================================
load_dotenv()

# ==========================================================
# Flask App Setup
# ==========================================================
app = Flask(__name__)
CORS(app, origins=["*"])

# ==========================================================
# Import LangChain modules - FIXED for langgraph 0.0.20
# ==========================================================
try:
    # For langgraph 0.0.20, use tool_executor instead
    from langgraph.prebuilt.tool_executor import ToolExecutor
    from langgraph.prebuilt import ToolInvocation
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_openai import ChatOpenAI
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from langchain_core.tools import tool
    print("✅ LangChain modules loaded successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Try alternative for older langgraph
    try:
        from langgraph.prebuilt import ToolExecutor
        from langgraph.prebuilt import ToolInvocation
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_openai import ChatOpenAI
        from langchain_community.tools.tavily_search import TavilySearchResults
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        print("✅ LangChain modules loaded with alternative paths")
    except ImportError as e2:
        print(f"❌ All import attempts failed: {e2}")

# ==========================================================
# Initialize LLM and Tools
# ==========================================================
llm = None
search_tool = None
agent = None

# Initialize LLM
try:
    llm = ChatOpenAI(
        model="openrouter/free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.2,
        streaming=True,
        max_tokens=800
    )
    print("✅ LLM initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize LLM: {e}")

# Initialize Tavily search
try:
    search_tool = TavilySearchResults(
        max_results=3,
        topic="general"
    )
    print("✅ Tavily search tool initialized")
except Exception as e:
    print(f"❌ Failed to initialize Tavily: {e}")

# ==========================================================
# System Prompt
# ==========================================================
SYSTEM_PROMPT = """
You are DeepChat, an advanced AI assistant with conversational memory and web search capabilities.

CRITICAL INSTRUCTION FOR TOOL USAGE:
You must AUTOMATICALLY use the web search tool for any user query that depends on real-world facts, real-time data, or conditions that can change over time. Do NOT rely on your internal knowledge for these topics.

Always run a search if the query involves:
1. Sports: Match results, tournament brackets, cups won, player stats.
2. Current Events & News: Breaking news, stocks, releases, technology trends.
3. Temporal Facts: Anything related to specific years, upcoming dates, or recent changes.

You remember previous turns in this conversation. Use context when the user refers to something previously discussed.
"""

# ==========================================================
# Create Agent with Memory - USING REACT AGENT FROM CORRECT PATH
# ==========================================================
try:
    if llm and search_tool:
        # Use the correct import for react agent in langgraph 0.0.20
        from langgraph.prebuilt import create_react_agent
        
        memory = MemorySaver()
        agent = create_react_agent(
            model=llm,
            tools=[search_tool],
            checkpointer=memory
        )
        print("✅ Agent initialized with conversational memory")
    else:
        print(f"⚠️ Agent not initialized - LLM: {llm is not None}, Search: {search_tool is not None}")
except ImportError as e:
    print(f"❌ Import error for create_react_agent: {e}")
    # Try fallback using tool executor
    try:
        from langgraph.prebuilt.tool_executor import ToolExecutor
        from langgraph.graph import StateGraph, END
        from langgraph.graph.message import add_messages
        from typing import TypedDict, Annotated, List
        from langchain_core.messages import AnyMessage
        
        # Create a simple agent with tool executor
        class AgentState(TypedDict):
            messages: Annotated[List[AnyMessage], add_messages]
        
        # Simple agent implementation
        def call_model(state):
            messages = state["messages"]
            response = llm.invoke(messages)
            return {"messages": [response]}
        
        def call_tool(state):
            messages = state["messages"]
            last_message = messages[-1]
            # Simple tool handling
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    if tool_call["name"] == "tavily_search":
                        result = search_tool.invoke(tool_call["args"])
                        return {"messages": [AIMessage(content=str(result))]}
            return {"messages": []}
        
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", call_tool)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            lambda state: "tools" if state["messages"][-1].tool_calls else END,
            {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")
        memory = MemorySaver()
        agent = workflow.compile(checkpointer=memory)
        print("✅ Fallback agent initialized")
    except Exception as e2:
        print(f"❌ Fallback agent initialization failed: {e2}")

# ==========================================================
# Routes
# ==========================================================
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if agent else "degraded",
        "message": "AI Assistant is running" if agent else "Agent not initialized",
        "llm_ready": llm is not None,
        "search_ready": search_tool is not None
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        "name": "DeepChat AI Backend",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/api/health": "Health check",
            "/api/chat/stream": "Streaming chat endpoint"
        }
    })

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Stream chat responses using Server-Sent Events (SSE)"""
    if not agent:
        return jsonify({"error": "Agent not initialized. Check server logs."}), 503
    
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' field"}), 400
        
        user_input = data['message'].strip()
        if not user_input:
            return jsonify({"error": "Empty message"}), 400
        
        thread_id = data.get('thread_id', 'default_thread')
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"📩 [Thread: {thread_id}] Received: {user_input[:50]}...")
        
        def generate():
            """Generator for streaming responses"""
            try:
                full_response = ""
                
                # For simple agent without streaming, get response
                if hasattr(agent, 'stream'):
                    # Use streaming if available
                    for chunk in agent.stream(
                        {
                            "messages": [
                                SystemMessage(content=SYSTEM_PROMPT),
                                HumanMessage(content=user_input)
                            ]
                        },
                        config=config,
                        stream_mode="values"
                    ):
                        if "messages" in chunk:
                            last_message = chunk["messages"][-1]
                            if isinstance(last_message, AIMessage):
                                content = last_message.content
                                if isinstance(content, str) and content:
                                    if len(content) > len(full_response):
                                        new_text = content[len(full_response):]
                                        if new_text:
                                            chunk_data = json.dumps({'content': new_text})
                                            yield f"data: {chunk_data}\n\n"
                                            full_response = content
                else:
                    # Fallback for non-streaming agent
                    response = agent.invoke(
                        {
                            "messages": [
                                SystemMessage(content=SYSTEM_PROMPT),
                                HumanMessage(content=user_input)
                            ]
                        },
                        config=config
                    )
                    if response and "messages" in response:
                        last_message = response["messages"][-1]
                        if isinstance(last_message, AIMessage):
                            content = last_message.content
                            if content:
                                chunk_data = json.dumps({'content': content})
                                yield f"data: {chunk_data}\n\n"
                                full_response = content
                
                yield f"data: {json.dumps({'done': True})}\n\n"
                print(f"✅ [Thread: {thread_id}] Streaming complete")
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error during stream: {error_msg}")
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive'
            }
        )
        
    except Exception as e:
        print(f"❌ Request Error: {e}")
        return jsonify({"error": str(e)}), 500

# ==========================================================
# Error Handlers
# ==========================================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ==========================================================
# Main Entry Point
# ==========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "="*60)
    print("🤖 DeepChat AI Backend")
    print("="*60)
    print(f"📡 Server running on port {port}")
    print(f"🌐 Health check: https://deepchat-backend-tyss.onrender.com/api/health")
    print("="*60 + "\n")
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )
