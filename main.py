from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
import os

# 设置tavily搜索工具的API Key
os.environ["TAVILY_API_KEY"] = "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"

# 基础模型
llm = ChatOpenAI(
    model="qwen2.5-1.5b-instruct",
    base_url="http://127.0.0.1:1234/v1",
    api_key=SecretStr("lm-studio"),
)

# llm = ChatOpenAI(
#     model="deepseek-r1-distill-qwen-1.5b",
#     base_url="http://127.0.0.1:1234/v1",
#     api_key=SecretStr("lm-studio"),
# )

# 定义工具
@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

# 设置搜索工具
search = TavilySearchResults(max_results=3)

tools = [add, multiply, search]

# 绑定工具
llm_with_tools = llm.bind_tools(tools)


# 使用AI
query = "1024乘1023等于多少？ 10023901234加上123456789等于多少？"
# query = "搜索一下 北京今天天气怎么样？"
# 创建消息列表
messages = [HumanMessage(query)]
# AI响应
ai_msg = llm_with_tools.invoke(messages)
# 打印调用的工具
print(ai_msg.tool_calls)
# 加入到消息列表
messages.append(ai_msg)

# 调用工具
for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "multiply": multiply, "tavily_search_results_json": search}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)

print(messages)

print(llm_with_tools.invoke(messages))