from typing_extensions import Annotated,TypedDict
from langgraph.graph.message import AnyMessage,add_messages
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import ToolNode
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
from langchain_ollama.chat_models import ChatOllama

# Langserve のインポートを追加
from langserve import add_routes
from fastapi import FastAPI # FastAPI アプリケーションを作成するために必要

class State(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]

def food_node(state):
    """
    MessagePlaceholdernにより、stateから過去のmessagesを取り出してLLMに送り、stateのMessageに追記
    """
    llm = OllamaLLM(model="gemma3:latest",temperature=0)
    promt = ChatPromptTemplate.from_messages([
        (
            "system",
            "あなたはおすすめの食事に関するAIエージェントです。ユーザーの入力に対して適切な回答を返してください。語尾はにゃんにしてください。"
        ),
        MessagesPlaceholder(variable_name="messages")
    ])
    agent = promt | llm | StrOutputParser()
    result = agent.invoke(state)
    return {"messages" : result}


def manga_node(state):
    """
    ChatModel.bind_tool()を利用することによって Tool を紐づけることができる。
    Tool 呼び出しが必要なときは呼び出す Tool やその引数の情報を返し、呼び出しが不要なときは普通に回答を生成する。
    """
    llm = ChatOllama(model="llama3.2",temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたは漫画に関する情報を回答するAIエージェントです。Toolを利用してユーザーの入力に対して適切な回答を返してください。"
            ),
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    agent = prompt | llm.bind_tools(tools) | StrOutputParser()
    result = agent.invoke(state)
    return {"messages": result}

def food_or_mange_router(state):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "あなたは次のノードを決定するAIエージェントです。ユーザーの入力から適切な次のノードを選択してください。"
                "次のノードは 'food', 'manga', 'end' のいずれかです。"
                "回答はこれらの単語のいずれかのみにしてください。他の文字は含めないでください。"
                "例えば、食事に関する内容であれば 'food' とだけ回答してください。"
            ),
            MessagesPlaceholder(variable_name="messages")
        ]
    )
    llm = OllamaLLM(model="gemma3:latest",temperature=0)
    router_chain = (
        prompt | llm | StrOutputParser()
    )
    result = router_chain.invoke(state)
    predicted_next_node = result.strip().lower()

    return predicted_next_node

# __name__=="__main__" のブロックから外に出して、Langserveがインポートできるようにする
wrapper = DuckDuckGoSearchAPIWrapper(region="jp-jp", max_results=3)
search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="text")
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [search,wikipedia]
tool_node = ToolNode(tools)

graph = StateGraph(State)

graph.add_node("food", food_node)
graph.add_node("manga" , manga_node)
graph.add_node("tools", tool_node)

graph.add_conditional_edges(
    START,
    food_or_mange_router,
    {
        "food":"food",
        "manga":"manga",
        "end":END
    }
)

graph.add_edge("food",END)
graph.add_conditional_edges(
"manga",
tools_condition,
{
    "tools": "tools",
    "__end__": END
}
)
graph.add_edge("tools", "manga")

app_graph = graph.compile() # 変数名を `app` から `app_graph` に変更し、Langserve の `app` と区別する

# FastAPI アプリケーションのインスタンスを作成
app = FastAPI(
    title="My LangGraph App",
    version="1.0",
    description="A LangGraph application for food and manga recommendations.",
)

# Langserve のルーティングを追加
# `/graph` というパスでアプリケーションを公開します
add_routes(app, app_graph, path="/graph")

# このファイルが直接実行された場合に Uvicorn でサーバーを起動
# このブロックは、Langserve がアプリケーションをインポートする際には実行されません。
# `uvicorn app:app` コマンドで実行するために必要です。
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)