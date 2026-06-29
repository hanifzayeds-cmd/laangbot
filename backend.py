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
    # Correct import for this version
    from langgraph.prebuilt.chat_agent_executor import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_openai import ChatOpenAI
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    print("✅ LangChain modules loaded successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Fallback: Use the old import style
    try:
        from langgraph.prebuilt import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_openai import ChatOpenAI
        from langchain_community.tools.tavily_search import TavilySearchResults
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        print("✅ LangChain modules loaded with fallback import")
    except ImportError as e2:
        print(f"❌ All import attempts failed: {e2}")
        # Exit or handle error appropriately

# ==========================================================
# Initialize LLM and Tools
# ==========================================================
llm = None
search_tool = None
agent = None

# Initialize LLM - FIXED: Removed 'proxies' causing parameters
try:
    # Only use supported parameters
    llm = ChatOpenAI(
        model="openrouter/free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.2,
        max_tokens=800
    )
    print("✅ LLM initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize LLM: {e}")
    # Try with even fewer parameters
    try:
        llm = ChatOpenAI(
            model="openrouter/free",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        print("✅ LLM initialized with minimal parameters")
    except Exception as e2:
        print(f"❌ Minimal LLM initialization also failed: {e2}")

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
# Create Agent with Memory
# ==========================================================
try:
    if llm and search_tool:
        memory = MemorySaver()
        agent = create_react_agent(
            model=llm,
            tools=[search_tool],
            checkpointer=memory
        )
        print("✅ Agent initialized with conversational memory")
    else:
        print(f"⚠️ Agent not initialized - LLM: {llm is not None}, Search: {search_tool is not None}")
except Exception as e:
    print(f"❌ Failed to initialize agent: {e}")

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
                # Use invoke for simplicity and compatibility
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
                            # Simulate streaming by sending the whole response
                            # For a better streaming experience, consider using a streaming model
                            chunk_data = json.dumps({'content': content})
                            yield f"data: {chunk_data}\n\n"
                
                yield f"data: {json.dumps({'done': True})}\n\n"
                print(f"✅ [Thread: {thread_id}] Request complete")
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error during request: {error_msg}")
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
