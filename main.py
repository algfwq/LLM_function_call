from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# 基础模型
llm = ChatOpenAI(
    model="qwen2.5-1.5b-instruct",
    base_url="http://127.0.0.1:1234/v1",
    api_key=SecretStr("lm-studio"),
)

# 定义工具
@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]

# 绑定工具
llm_with_tools = llm.bind_tools(tools)


# 使用AI
query = "1024乘1023等于多少？ 10023901234加上123456789等于多少？"
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
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    messages.append(tool_msg)

print(messages)

print(llm_with_tools.invoke(messages))