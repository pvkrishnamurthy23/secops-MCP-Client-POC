import os
import openai
import sys
import asyncio
import json
from typing import List, Dict, Any, Optional
import streamlit as st
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StreamableHttpConnection
import boto3
from botocore.exceptions import ClientError

# Add the project root to sys.path to import Utils.logger
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Utils.logger import logger

# Helper to mask sensitive values in logs
_DEF_MASK = "***"

def _mask_secret(value: Optional[str], show_last: int = 4) -> str:
    if not value:
        return _DEF_MASK
    try:
        tail = value[-show_last:]
        return f"{_DEF_MASK}{tail}"
    except Exception:
        return _DEF_MASK


def get_secret():

    secret_name = "ipl/secops/db/openai"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        logger.info("Fetching secrets from AWS Secrets Manager: %s (region=%s)", secret_name, region_name)
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
        secret = get_secret_value_response['SecretString']
        logger.debug("Secrets retrieved successfully from AWS Secrets Manager")
        return json.loads(secret)
    except ClientError as e:
        logger.exception("Unable to retrieve secret from AWS Secrets Manager")
        raise RuntimeError(f"Unable to retrieve secret: {e}")

# Load secrets once at startup
try:
    logger.info("Initializing application. Loading secrets from AWS Secrets Manager...")
    secrets = get_secret()
    safe_keys = ", ".join(sorted([k for k in secrets.keys() if not any(s in k.lower() for s in ["key", "token", "secret", "password"])]))
    logger.info("Secrets loaded. Available non-sensitive keys: %s", safe_keys)

    openai.api_key = secrets.get('OPENAI_API_KEY')
    if openai.api_key:
        logger.debug("OpenAI API key configured via secrets ")
    else:
        logger.error("OPENAI_API_KEY is missing in AWS secret; the app will not function correctly.")
except Exception:
    logger.exception("Startup failed while loading secrets")
    raise


# ---------- LangChain Q&A Agent ----------
def create_llm(config: Dict[str, Any]) -> ChatOpenAI:
    model = config.get("openai_model", "gpt-4o-mini")
    temperature = config.get("llm_temperature", 0.0)
    max_tokens = config.get("llm_max_tokens", None)
    timeout = config.get("llm_timeout", None)
    logger.info(
        "Creating LLM client: model=%s, temperature=%s, max_tokens=%s, timeout=%s",
        model, temperature, max_tokens, timeout,
    )
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        api_key=secrets.get("OPENAI_API_KEY"),
    )


def create_system_prompt() -> str:
    """Create the system prompt with knowledge sources"""
    return (
        """You are a specialized cybersecurity assistant.\n\n"
        "You have access to the following intelligent tools:\n"
        "1. **SecurityReportSummaryTool**: Accepts user input in natural language (e.g., 'Give me summary of my environment? or what kind of issues are there in my environment') and It provides a security summary to the user.\n"
        "2. **FindingsInterpreterTool**: Accepts user input in natural language (e.g., 'What kind of critical issues are there in iam? or what kind of high severity misconfigurations are there in s3?') and converts it into structured query results about findings or misconfigurations detected in the environment.\n"
        "3. **RemediationSuggesterTool**: Given a specific finding or misconfiguration or description (e.g., 'Public S3 bucket'), it returns actionable remediation guidance or custom bash script or terraform script if the user asks for it.\n"
        "4. **PlatformWebSearchTool**: Performs a RAG search to answer general calibo platform-related questions (e.g., 'How do I integrate Azure AD in the platform?').\n\n"
        "INSTRUCTIONS:\n"
        "1. Start by analyzing the user's question carefully.\n"
        "2. If the input is related to *asking about security issues summary, or findings in general*, use **SecurityReportSummaryTool** to provide summary.\n"
        "3. If the input is related to *asking about security issues, misconfigurations, or findings in the environment*, use **FindingsInterpreterTool** to extract the structured query.\n"
        "4. If the input is about *remediation steps for a specific misconfiguration or finding*, use **RemediationSuggesterTool**.\n"
        "5. If the input is a *general platform-related or integration question*, use **PlatformWebSearchTool**.\n"
        "6. If the input is ambiguous or missing important context (e.g., 'How do I fix this?' without saying what “this” is), ask a clear follow-up question to get required details.\n"
        "7. NEVER assume missing information. Always clarify before proceeding.\n"
        "8. Only respond with answers when confident. Otherwise, request more input.\n\n"
        "TOOL USAGE STRATEGY:\n"
        "- **First**, analyze intent: Is this about summary, findings, remediation, or platform help?\n"
        "- **Second**, use the appropriate tool to process the request.\n"
        "- **Third**, if the response from the tool is not enough, ask the user for more details.\n\n"
        "RULES:\n"
        "- Be clear, precise, and helpful.\n"
        "- Never guess. Always clarify vague inputs.\n"
        "- Always use tools in the reasoning loop before giving a final answer.\n"
        "- Cite sources or reasoning where possible.\n\n"
        "Guardrails:\n"
        "- if the question is not related to calibo platform or misconfigurations or cyber security. give Final response as "No Information, Ask questions on Findings, Remediations and Calibo Platfprm only)"\n"
        "- Dont give any improper responses.\n"
        "TOOLS YOU CAN USE:\n"
        "{{tools}}\n\n"
        "Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).\n\n"
        "Valid 'action' values: 'Final Answer' or {{tool_names}}\n\n"
        ""In action_input, "query" is a compulsary sub field if action key is tool name.\n"
        ""If the action is Final Answer, format the response recieved from the tool for best presentation, provide in table format if required. dont shorten the content. provide all information which came from the tool. the response should be a valid string and not a dictionary\n"
        "Provide only ONE action per $JSON_BLOB, as shown:\n"
        {{{{  
  "action": "$TOOL_NAME",  
  "action_input": {{  
    "query": "$INPUT",  
    
  }}  
}}}}
        "strictly Follow format:\n\n"
        "Question: input question to answer\n"
        "Thought: consider previous and subsequent steps\n"
        "Action:\n```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action: 
```
{{{{
  "action": "Final Answer",
  "action_input": "response"
}}}}
```
Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate and ask for clarification if something is not clear. Format is Action:```$JSON_BLOB```then Observation
"""
    )


class DomainQAAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            raise ValueError("Configuration is required")
        self.config = config
        self.mcp_server_url = config.get("mcp_server_url", "http://0.0.0.0:8000/mcp")
        self.llm = create_llm(config)
        self.chat_history: List[BaseMessage] = []
        self.mcp_client = None
        self.tools = []
        self.agent_executor = None
        logger.info(
            "DomainQAAgent initialized (model=%s, MCP server=%s)",
            self.config.get("openai_model"), self.mcp_server_url,
        )

    async def _initialize_mcp_client(self):
        if self.mcp_client is None:
            logger.debug("Initializing MCP client connection to %s", self.mcp_server_url)
            connection = StreamableHttpConnection(url=self.mcp_server_url, transport="streamable_http")
            self.mcp_client = MultiServerMCPClient(connections={"qa_agent_server": connection})
            self.tools = await self.mcp_client.get_tools()
            logger.info("Loaded %d tools: %s", len(self.tools), ", ".join([t.name for t in self.tools]))
            self.agent_executor = self._create_agent()

    def _create_agent(self) -> AgentExecutor:
        system_message = create_system_prompt()
        logger.debug("Creating structured chat agent with %d tools", len(self.tools))
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                (
                    "human",
                    "{input}\n\n{agent_scratchpad}(reminder to respond in a JSON blob no matter what)...",
                ),
            ]
        ).partial(
            tools="\n".join([f"- {tool.name}" for tool in self.tools]),
            tool_names=", ".join([tool.name for tool in self.tools])
        )

        agent = create_structured_chat_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )

    async def achat(self, user_input: str) -> str:
        await self._initialize_mcp_client()
        logger.info("User input: %s", user_input)
        agent_input = {
            "input": user_input,
            #"chat_history": self.chat_history[-5:],
        }
        try:
            response = await self.agent_executor.ainvoke(agent_input)
            steps = response.get("intermediate_steps", []) if isinstance(response, dict) else []
            if steps:
                for idx, step in enumerate(steps, start=1):
                    try:
                        action, observation = step
                        tool_name = getattr(action, "tool", str(action))
                        tool_input = getattr(action, "tool_input", {})
                        logger.info("Agent Step %s: action=%s input=%s", idx, tool_name, tool_input)
                        logger.debug("Agent Step %s: observation=%s", idx, observation)
                    except Exception:
                        logger.debug("Agent Step %s logging failed; raw step: %s", idx, step)
            answer = response.get("output", "I couldn't process your request.") if isinstance(response, dict) else str(response)
            self.chat_history.extend([
                HumanMessage(content=user_input), AIMessage(content=answer if isinstance(answer, str) else str(answer))
            ])
            logger.info("Agent response length: %s", len(answer) if isinstance(answer, str) else -1)
            return answer
        except Exception:
            logger.exception("Agent execution failed")
            return "Sorry, I encountered an error while processing your request."

    async def close(self):
        if self.mcp_client:
            await self.mcp_client.close()
            logger.debug("MCP client connection closed")

# ---------- Streamlit Chat UI ----------
st.set_page_config(page_title="SecOps Co-Pilot", page_icon="🛡️")
st.title("Calibo Accelerate SecOPS Assistant")

if "agent" not in st.session_state:
    config = {
        "openai_model": "gpt-4o-mini",
        "mcp_server_url": "http://0.0.0.0:8000/mcp"
    }
    st.session_state.agent = DomainQAAgent(config=config)
    st.session_state.history = []
    logger.info("Streamlit session initialized with default config: %s", config)

# 🔁 Display full message history
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)


# Prebuilt question buttons
st.markdown("### Quick Questions")
col1, col2, col3 = st.columns(3)

# Define a list of prebuilt questions
prebuilt_questions = [
    "Give me summary of the Security misconfigurations in my environment?",
    "Provide count of all the high severity misconfiguration for each service?",
    "List all the high severity misconfigurations in EC2",
    "Write script to make my s3 bucket private?",    
    "Tell me about all the security features of calibo platform?",
    "How to integrate Sonarqube in Calibo platform"
]

# Store selected question
selected_question = None

# Layout buttons dynamically
for i, question in enumerate(prebuilt_questions):
    if i % 3 == 0:
        col = col1
    elif i % 3 == 1:
        col = col2
    else:
        col = col3
    if col.button(question):
        selected_question = question
        logger.info("Quick question selected: %s", question)

# 📝 Accept new input (from button or chat input)
user_input = st.chat_input("Ask your question about findings, remediation, or platform...") or selected_question
# 📝 Accept new input
#user_input = st.chat_input("Ask your question about findings, remediation, or platform...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        logger.debug("Invoking agent for user input")
        response = asyncio.run(st.session_state.agent.achat(user_input))
        logger.debug("Agent response received")
        # Try to parse response as JSON if it's a string
        if isinstance(response, str):
            try:
                parsed = json.loads(response)
                st.json(parsed)
            except json.JSONDecodeError:
                st.markdown(response)
        elif isinstance(response, dict) or isinstance(response, list):
            st.json(response)
        else:
            st.markdown(str(response))
        logger.info("Response rendered to UI")
        

    # 💾 Save to history
    st.session_state.history.append(HumanMessage(content=user_input))
    st.session_state.history.append(AIMessage(content=str(response)))
    #st.session_state.agent.chat_history = st.session_state.history[-1:]


