import streamlit as st
import yaml
import os
from datetime import datetime
import json
from typing import Dict, List, Optional
import pandas as pd

# API clients
try:
    import openai
    from anthropic import Anthropic
    import google.generativeai as genai
except ImportError as e:
    st.error(f"Missing required library: {e}")

class AgentOrchestrator:
    """Main orchestrator for the GUDID agentic AI system"""
    
    def __init__(self, config_path: str = "agents.yaml"):
        self.config = self.load_config(config_path)
        self.agents = {}
        self.conversation_history = []
        self.initialize_agents()
    
    def load_config(self, path: str) -> Dict:
        """Load agent configurations from YAML"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            st.error(f"Failed to load config: {e}")
            return {}
    
    def initialize_agents(self):
        """Initialize all agents based on configuration"""
        for agent_name, agent_config in self.config.get('agents', {}).items():
            self.agents[agent_name] = Agent(agent_name, agent_config)
    
    def route_query(self, user_query: str) -> str:
        """Route user query to appropriate agent"""
        # Simple keyword-based routing (can be enhanced with ML)
        query_lower = user_query.lower()
        
        routing_keywords = {
            'nlp_analyzer': ['分析', '文字', 'analyze', 'text', 'nlp', '實體'],
            'anomaly_detector': ['異常', 'anomaly', '偵測', 'detect', '檢測'],
            'duplicate_checker': ['重複', 'duplicate', '相似', 'similar'],
            'label_matcher': ['標籤', 'label', '比對', 'match', 'ocr'],
            'data_standardizer': ['標準化', 'standardize', '正規化', 'normalize'],
            'adverse_event_linker': ['不良事件', 'adverse', '連結', 'link'],
            'recall_manager': ['回收', 'recall', '追蹤', 'track'],
            'eifu_manager': ['說明書', 'eifu', 'instructions'],
            'customs_verifier': ['海關', 'customs', '查驗', 'verify'],
            'international_connector': ['國際', 'international', '同步', 'sync']
        }
        
        for agent_name, keywords in routing_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return agent_name
        
        return 'nlp_analyzer'  # Default agent
    
    def process_query(self, user_query: str, selected_agent: Optional[str] = None) -> Dict:
        """Process user query through appropriate agent"""
        agent_name = selected_agent if selected_agent else self.route_query(user_query)
        
        if agent_name not in self.agents:
            return {"error": f"Agent {agent_name} not found"}
        
        agent = self.agents[agent_name]
        response = agent.execute(user_query)
        
        # Log conversation
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'agent': agent_name,
            'query': user_query,
            'response': response
        })
        
        return {
            'agent': agent_name,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }

class Agent:
    """Individual agent for specific GUDID use case"""
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.llm_provider = config.get('llm_provider', 'openai')
        self.model = config.get('model', 'gpt-4o-mini')
        self.system_prompt = config.get('system_prompt', '')
        self.capabilities = config.get('capabilities', [])
        
    def execute(self, query: str) -> str:
        """Execute agent logic based on query"""
        try:
            if self.llm_provider == 'openai':
                return self._execute_openai(query)
            elif self.llm_provider == 'anthropic':
                return self._execute_anthropic(query)
            elif self.llm_provider == 'gemini':
                return self._execute_gemini(query)
            else:
                return f"Unsupported LLM provider: {self.llm_provider}"
        except Exception as e:
            return f"Error executing agent {self.name}: {str(e)}"
    
    def _execute_openai(self, query: str) -> str:
        """Execute using OpenAI API"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return "OpenAI API key not configured"
        
        client = openai.OpenAI(api_key=api_key)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    def _execute_anthropic(self, query: str) -> str:
        """Execute using Anthropic API"""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            return "Anthropic API key not configured"
        
        client = Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        
        return response.content[0].text
    
    def _execute_gemini(self, query: str) -> str:
        """Execute using Google Gemini API"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            return "Gemini API key not configured"
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model)
        
        prompt = f"{self.system_prompt}\n\nUser Query: {query}"
        response = model.generate_content(prompt)
        
        return response.text

def main():
    st.set_page_config(
        page_title="GUDID Agentic AI System",
        page_icon="🏥",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            padding: 1rem 0;
        }
        .agent-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .metric-card {
            background-color: #e8f4f8;
            padding: 1.5rem;
            border-radius: 0.5rem;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🏥 GUDID 智能代理系統</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize orchestrator
    if 'orchestrator' not in st.session_state:
        st.session_state.orchestrator = AgentOrchestrator()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    orchestrator = st.session_state.orchestrator
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ 系統設定")
        
        # API Keys Configuration
        with st.expander("🔑 API 金鑰設定", expanded=False):
            openai_key = st.text_input("OpenAI API Key", type="password", 
                                       value=os.getenv('OPENAI_API_KEY', ''))
            anthropic_key = st.text_input("Anthropic API Key", type="password",
                                         value=os.getenv('ANTHROPIC_API_KEY', ''))
            gemini_key = st.text_input("Gemini API Key", type="password",
                                       value=os.getenv('GEMINI_API_KEY', ''))
            
            if st.button("💾 儲存金鑰"):
                os.environ['OPENAI_API_KEY'] = openai_key
                os.environ['ANTHROPIC_API_KEY'] = anthropic_key
                os.environ['GEMINI_API_KEY'] = gemini_key
                st.success("金鑰已儲存！")
        
        st.markdown("---")
        
        # Agent Selection
        st.subheader("🤖 選擇代理")
        agent_options = {
            'auto': '🎯 自動路由',
            'nlp_analyzer': '📝 NLP 分析',
            'anomaly_detector': '🔍 異常檢測',
            'duplicate_checker': '👥 重複檢查',
            'label_matcher': '🏷️ 標籤比對',
            'data_standardizer': '📊 資料標準化',
            'adverse_event_linker': '⚠️ 不良事件連結',
            'recall_manager': '📢 回收管理',
            'eifu_manager': '📖 電子說明書',
            'customs_verifier': '🛃 海關查驗',
            'international_connector': '🌍 國際連結'
        }
        
        selected_agent = st.selectbox(
            "選擇代理",
            options=list(agent_options.keys()),
            format_func=lambda x: agent_options[x]
        )
        
        st.markdown("---")
        
        # System Status
        st.subheader("📊 系統狀態")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("活躍代理", len(orchestrator.agents))
        with col2:
            st.metric("處理請求", len(st.session_state.messages))
        
        # Clear conversation
        if st.button("🗑️ 清除對話", use_container_width=True):
            st.session_state.messages = []
            orchestrator.conversation_history = []
            st.rerun()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["💬 對話介面", "🤖 代理資訊", "📊 分析儀表板", "📚 使用說明"])
    
    with tab1:
        # Chat interface
        st.subheader("對話介面")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "agent" in message:
                    st.caption(f"🤖 處理代理: {message['agent']}")
        
        # Chat input
        if prompt := st.chat_input("請輸入您的查詢..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Process query
            with st.chat_message("assistant"):
                with st.spinner("處理中..."):
                    agent_to_use = None if selected_agent == 'auto' else selected_agent
                    result = orchestrator.process_query(prompt, agent_to_use)
                    
                    response = result.get('response', 'No response generated')
                    agent_used = result.get('agent', 'unknown')
                    
                    st.markdown(response)
                    st.caption(f"🤖 處理代理: {agent_used}")
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "agent": agent_used
                    })
    
    with tab2:
        # Agent information
        st.subheader("代理資訊")
        
        for agent_name, agent in orchestrator.agents.items():
            with st.expander(f"🤖 {agent_name}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**描述:**", agent.config.get('description', 'No description'))
                    st.write("**LLM 提供者:**", agent.llm_provider)
                    st.write("**模型:**", agent.model)
                
                with col2:
                    st.write("**功能:**")
                    for capability in agent.capabilities:
                        st.write(f"- {capability}")
                
                with st.container():
                    st.write("**系統提示:**")
                    st.code(agent.system_prompt, language="text")
    
    with tab3:
        # Analytics dashboard
        st.subheader("分析儀表板")
        
        if orchestrator.conversation_history:
            # Agent usage statistics
            agent_usage = {}
            for entry in orchestrator.conversation_history:
                agent = entry['agent']
                agent_usage[agent] = agent_usage.get(agent, 0) + 1
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("總請求數", len(orchestrator.conversation_history))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("最常用代理", max(agent_usage, key=agent_usage.get))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("活躍代理數", len(agent_usage))
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Agent usage chart
            st.subheader("代理使用統計")
            df_usage = pd.DataFrame(list(agent_usage.items()), 
                                   columns=['Agent', 'Usage Count'])
            st.bar_chart(df_usage.set_index('Agent'))
            
            # Recent activity
            st.subheader("最近活動")
            recent_activities = orchestrator.conversation_history[-10:]
            for activity in reversed(recent_activities):
                with st.container():
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.write(f"**{activity['agent']}**")
                        st.caption(activity['timestamp'])
                    with col2:
                        st.write(f"Query: {activity['query'][:100]}...")
        else:
            st.info("尚無分析資料。請開始使用系統以查看統計資訊。")
    
    with tab4:
        # Documentation
        st.subheader("使用說明")
        
        st.markdown("""
        ### 📖 GUDID 智能代理系統使用指南
        
        #### 🎯 系統概述
        本系統是基於全球唯一器材識別資料庫(GUDID)需求的概念驗證(POC)系統，
        整合了多個AI代理來處理醫療器材管理的各種任務。
        
        #### 🤖 可用代理
        
        1. **NLP 分析代理** - 自然語言處理與實體識別
        2. **異常檢測代理** - 資料異常偵測與標記
        3. **重複檢查代理** - 智能重複資料檢測
        4. **標籤比對代理** - 標籤資訊自動比對
        5. **資料標準化代理** - 資料標準化與正規化
        6. **不良事件連結代理** - 不良事件智能連結
        7. **回收管理代理** - 產品回收追蹤管理
        8. **電子說明書代理** - eIFU管理
        9. **海關查驗代理** - 海關進出口查驗
        10. **國際連結代理** - 國際資料庫連結
        
        #### 💡 使用方式
        
        1. **配置API金鑰**: 在側邊欄的設定區域輸入您的API金鑰
        2. **選擇代理**: 選擇特定代理或使用自動路由
        3. **輸入查詢**: 在對話介面輸入您的問題或需求
        4. **查看結果**: 系統會自動處理並返回結果
        
        #### 📊 功能特色
        
        - ✅ 多LLM支援 (OpenAI, Anthropic, Gemini)
        - ✅ 智能路由系統
        - ✅ 對話歷史記錄
        - ✅ 即時分析儀表板
        - ✅ 可擴展的代理架構
        
        #### ⚠️ 注意事項
        
        - 這是概念驗證系統，不適用於生產環境
        - 請勿輸入真實的敏感醫療資料
        - API金鑰在會話結束後不會保存
        
        #### 🔗 相關資源
        
        - [GUDID官方網站](https://example.com)
        - [技術文檔](https://example.com/docs)
        - [GitHub Repository](https://github.com/example/gudid)
        """)
        
        st.markdown("---")
        st.info("💡 提示: 使用側邊欄切換不同的代理，或讓系統自動為您選擇最合適的代理。")

if __name__ == "__main__":
    main()
