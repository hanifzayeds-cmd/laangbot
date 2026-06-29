import os
import json
import time
import logging
from flask import Flask, request, Response, jsonify, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv

# ==========================================================
# Load API Keys
# ==========================================================
load_dotenv()

# ==========================================================
# Logging Setup
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==========================================================
# Flask App Setup
# ==========================================================
app = Flask(__name__)
CORS(app, origins=["*"])

# ==========================================================
# Import LangChain modules
# ==========================================================
try:
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_openai import ChatOpenAI
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    logger.info("✅ LangChain modules loaded successfully")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    try:
        from langgraph.prebuilt.chat_agent_executor import create_react_agent
        from langgraph.checkpoint.memory import MemorySaver
        from langchain_openai import ChatOpenAI
        from langchain_community.tools.tavily_search import TavilySearchResults
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        logger.info("✅ LangChain modules loaded with fallback import")
    except ImportError as e2:
        logger.error(f"❌ All import attempts failed: {e2}")

# ==========================================================
# Initialize LLM and Tools - WITH MAXIMUM TOKEN CAPACITY
# ==========================================================
llm = None
search_tool = None
agent = None

# Initialize LLM with maximum token capacity for long responses
try:
    llm = ChatOpenAI(
        model="openrouter/free",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.2,
        max_tokens=4096,
        timeout=120,
        max_retries=3,
        model_kwargs={
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        }
    )
    logger.info("✅ LLM initialized with 4096 token capacity")
except Exception as e:
    logger.error(f"❌ Failed to initialize LLM: {e}")
    try:
        llm = ChatOpenAI(
            model="openrouter/free",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            max_tokens=4096,
            timeout=120
        )
        logger.info("✅ LLM initialized with minimal parameters and 4096 tokens")
    except Exception as e2:
        logger.error(f"❌ Minimal LLM initialization also failed: {e2}")

# Initialize Tavily search with improved settings
try:
    search_tool = TavilySearchResults(
        max_results=5,
        topic="general",
        include_answer=True,
        include_raw_content=False
    )
    logger.info("✅ Tavily search tool initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize Tavily: {e}")

# ==========================================================
# System Prompt - Enhanced for Professional & Long Responses
# ==========================================================
SYSTEM_PROMPT = """You are DeepChat, an advanced AI assistant with conversational memory and web search capabilities.
🚨 CRITICAL INSTRUCTION FOR TOOL USAGE:
You MUST AUTOMATICALLY use the web search tool for ANY user query that depends on real-world facts, real-time data, or conditions that can change over time. Do NOT rely on your internal knowledge for these topics. Always run a search if the query involves:
1. Sports: Match results, tournament brackets, cups won, player stats, team performance
2. Current Events & News: Breaking news, stocks, releases, technology trends, political events
3. Temporal Facts: Anything related to specific years, upcoming dates, or recent changes
4. Real-time Data: Weather, prices, statistics, rankings
5. Historical Events: Historical dates, events, or figures
6. Scientific Facts: Recent discoveries, research findings
7. Geographic Information: Country statistics, population data
8. Any factual question where accuracy matters

📝 RESPONSE FORMATTING INSTRUCTIONS - BE PROFESSIONAL AND COMPREHENSIVE:
1. Structure:
   - Use # for main title (if comprehensive response)
   - Use ## for major sections
   - Use ### for subsections
   - Keep logical flow from introduction to conclusion
2. Formatting:
   - Use **bold** for emphasis on key terms and important concepts
   - Use *italic* for less emphasis or examples
   - Use bullet points (-) for lists of items
   - Use numbered lists (1., 2., 3.) for sequential or prioritized information
   - Use tables for comparative data (| Column 1 | Column 2 |)
   - Use code blocks (```) for code, formulas, or structured data
   - Use blockquotes (> ) for important notes, warnings, or key takeaways
3. Content Guidelines:
   - Keep paragraphs clear, concise, and well-structured
   - For simple questions: Provide clear, direct answers with 1-2 paragraphs
   - For complex questions: Provide comprehensive, detailed responses with multiple sections
   - Always aim to be thorough and educational
   - Include examples where helpful
   - Provide summaries or recaps for longer responses
   - Use emoji sparingly for visual appeal (💡, 📊, ⚡, 🎯, ✅, 🔍)
4. Professional Tone:
   - Be helpful, informative, and accurate
   - Acknowledge limitations when appropriate
   - Cite sources when possible
   - Maintain a professional and friendly tone
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
        logger.info("✅ Agent initialized with conversational memory")
    else:
        logger.warning(f"⚠️ Agent not initialized - LLM: {llm is not None}, Search: {search_tool is not None}")
except Exception as e:
    logger.error(f"❌ Failed to initialize agent: {e}")

# ==========================================================
# Routes
# ==========================================================
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed status"""
    status = {
        "status": "healthy" if agent else "degraded",
        "message": "AI Assistant is running" if agent else "Agent not initialized",
        "llm_ready": llm is not None,
        "search_ready": search_tool is not None,
        "max_tokens": 4096,
        "timeout": 120,
        "version": "2.0.0",
        "features": {
            "streaming": True,
            "word_by_word": True,
            "web_search": True,
            "conversational_memory": True
        }
    }
    return jsonify(status)

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        "name": "DeepChat AI Backend",
        "version": "2.0.0",
        "status": "running",
        "description": "Advanced AI assistant with web search and conversational memory",
        "features": {
            "max_tokens": 4096,
            "web_search": True,
            "conversational_memory": True,
            "streaming": True,
            "word_by_word_delivery": True
        },
        "endpoints": {
            "/": "API information",
            "/api/health": "Health check with detailed status",
            "/api/chat/stream": "Streaming chat endpoint with word-by-word delivery",
            "/api/stats": "System statistics",
            "/api/threads/<thread_id>": "Manage conversation threads"
        }
    })

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Stream chat responses with word-by-word delivery and huge token capacity"""
    if not agent:
        return jsonify({
            "error": "Agent not initialized. Please check server logs.",
            "status": "degraded"
        }), 503
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request body"}), 400
        if 'message' not in data:
            return jsonify({"error": "Missing 'message' field"}), 400
        user_input = data['message'].strip()
        if not user_input:
            return jsonify({"error": "Empty message"}), 400
        
        thread_id = data.get('thread_id', 'default_thread')
        config = {"configurable": {"thread_id": thread_id}}
        logger.info(f"📩 [Thread: {thread_id}] Received: {user_input[:50]}...")
        
        def generate():
            """Generator for streaming responses with word-by-word delivery"""
            try:
                full_response = ""
                start_time = time.time()
                response = agent.invoke(
                    {
                        "messages": [
                            SystemMessage(content=SYSTEM_PROMPT),
                            HumanMessage(content=user_input)
                        ]
                    },
                    config=config
                )
                elapsed_time = time.time() - start_time
                logger.info(f"⏱️ [Thread: {thread_id}] Agent response time: {elapsed_time:.2f}s")
                
                if response and "messages" in response:
                    last_message = response["messages"][-1]
                    if isinstance(last_message, AIMessage):
                        content = last_message.content
                        if content:
                            full_response = content
                            words = content.split(' ')
                            total_words = len(words)
                            logger.info(f"📝 [Thread: {thread_id}] Streaming {total_words} words")
                            
                            if total_words > 0:
                                yield f"data: {json.dumps({'content': words[0] + ' '})}\n\n"
                            
                            for i in range(1, total_words):
                                word = words[i]
                                chunk = word + ' ' if i < total_words - 1 else word
                                yield f"data: {json.dumps({'content': chunk})}\n\n"
                                
                                if total_words > 500:
                                    time.sleep(0.015)
                                elif total_words > 300:
                                    time.sleep(0.02)
                                elif total_words > 200:
                                    time.sleep(0.025)
                                elif total_words > 100:
                                    time.sleep(0.03)
                                else:
                                    time.sleep(0.04)
                                
                                if i % 100 == 0:
                                    time.sleep(0.001)
                            
                            word_count = len(full_response.split(' '))
                            char_count = len(full_response)
                            completion_data = {
                                'done': True,
                                'word_count': word_count,
                                'char_count': char_count,
                                'elapsed_time': elapsed_time
                            }
                            yield f"data: {json.dumps(completion_data)}\n\n"
                            logger.info(f"✅ [Thread: {thread_id}] Complete - {char_count} chars, {word_count} words in {elapsed_time:.2f}s")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ Error during request: {error_msg}")
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
        
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive',
                'Content-Type': 'text/event-stream'
            }
        )
    except Exception as e:
        logger.error(f"❌ Request Error: {e}")
        return jsonify({"error": str(e)}), 500

# ==========================================================
# Additional Utility Routes
# ==========================================================
@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics and performance metrics"""
    return jsonify({
        "agent_ready": agent is not None,
        "llm_ready": llm is not None,
        "search_ready": search_tool is not None,
        "max_tokens": 4096,
        "timeout": 120,
        "memory_type": "MemorySaver",
        "search_tool": "TavilySearchResults",
        "model": "openrouter/free",
        "temperature": 0.2,
        "version": "2.0.0"
    })

@app.route('/api/threads/<thread_id>', methods=['DELETE'])
def delete_thread(thread_id):
    """Delete a conversation thread"""
    return jsonify({
        "message": f"Thread {thread_id} marked for deletion",
        "note": "MemorySaver doesn't support deletion, thread will be cleared on restart"
    })

@app.route('/api/clear', methods=['POST'])
def clear_memory():
    """Clear all conversation memory"""
    global agent
    try:
        if agent:
            memory = MemorySaver()
            agent = create_react_agent(
                model=llm,
                tools=[search_tool],
                checkpointer=memory
            )
            return jsonify({
                "status": "success",
                "message": "All conversation memory cleared"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Agent not initialized"
            }), 503
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to clear memory: {str(e)}"
        }), 500

# ==========================================================
# Error Handlers
# ==========================================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not found",
        "message": "The requested endpoint does not exist"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again later."
    }), 500

@app.errorhandler(503)
def service_unavailable(error):
    return jsonify({
        "error": "Service unavailable",
        "message": "The AI service is currently unavailable. Please try again later."
    }), 503

# ==========================================================
# Request/Response Logging
# ==========================================================
@app.before_request
def log_request_info():
    """Log incoming requests for debugging"""
    logger.debug(f"📨 {request.method} {request.path}")
    if request.method == 'POST' and request.path == '/api/chat/stream':
        try:
            data = request.get_json()
            if data and 'message' in data:
                msg_preview = data['message'][:50] + '...' if len(data['message']) > 50 else data['message']
                logger.debug(f"📝 Message preview: {msg_preview}")
        except Exception:
            pass

@app.after_request
def log_response_info(response):
    """Log response status for debugging"""
    logger.debug(f"📤 Response: {response.status_code}")
    return response

# ==========================================================
# Main Entry Point
# ==========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 70)
    print("🤖 DeepChat AI Backend v2.0")
    print("=" * 70)
    print(f"📡 Server running on port {port}")
    print("🔢 Max tokens: 4096 (for long, comprehensive responses)")
    print("⏱️ Timeout: 120 seconds")
    print(f"🔍 Web Search: {'Enabled' if search_tool else 'Disabled'}")
    print(f"🧠 Memory: {'Enabled' if agent else 'Disabled'}")
    print("📊 Model: openrouter/free")
    print(f"🌐 Health check: http://localhost:{port}/api/health")
    print("=" * 70)
    print("💡 Ready to handle requests...\n")
    
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )
