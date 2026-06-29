import os
import json
from flask import Flask, request, Response, jsonify, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv

# Try importing with fallbacks
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    print("⚠️ langchain_openai not found, installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "langchain-openai==0.0.8"])
    from langchain_openai import ChatOpenAI

try:
    from langchain_community.tools.tavily_search import TavilySearchResults
except ImportError:
    print("⚠️ langchain_community not found, installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "langchain-community==0.0.10"])
    from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ==========================================================
# Load API Keys
# ==========================================================
load_dotenv()

required_keys = ["OPENROUTER_API_KEY", "TAVILY_API_KEY"]
missing_keys = [key for key in required_keys if not os.getenv(key)]
if missing_keys:
    print(f"❌ Missing required API keys: {', '.join(missing_keys)}")
    print("Please add them to your .env file")

# ==========================================================
# Flask App Setup
# ==========================================================
app = Flask(__name__)
CORS(app, origins=["*"])

# ==========================================================
# LLM Configuration
# ==========================================================
llm = None
search_tool = None
agent = None

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

# ==========================================================
# Web Search Tool
# ==========================================================
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
        memory = InMemorySaver()
        agent = create_react_agent(
            model=llm,
            tools=[search_tool],
            checkpointer=memory
        )
        print("✅ Automated Agent with Conversational Memory initialized")
    else:
        print("⚠️ Agent not initialized - missing LLM or search tool")
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
        "message": "AI Assistant is running" if agent else "Agent not initialized"
    })

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Stream chat responses using Server-Sent Events (SSE) with Memory"""
    if not agent:
        return jsonify({"error": "Agent not initialized"}), 503
    
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' field"}), 400
        
        user_input = data['message'].strip()
        if not user_input:
            return jsonify({"error": "Empty message"}), 400
        
        thread_id = data.get('thread_id', 'default_thread')
        config = {"configurable": {"thread_id": thread_id}}
        
        print(f"📩 [Thread: {thread_id}] Received: {user_input}")
        
        def generate():
            """Generator for streaming responses"""
            try:
                full_response = ""
                
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
                                new_text = content[len(full_response):]
                                if new_text:
                                    chunk_data = json.dumps({'content': new_text})
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
# Main Entry Point
# ==========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "="*50)
    print("🤖 DeepChat AI Backend")
    print("="*50)
    print(f"📡 Server running on port {port}")
    print("="*50 + "\n")
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
