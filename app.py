## Load Libraries
import os
import certifi
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_react_agent,AgentExecutor

## Load Environment Variables
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

OPEN_AI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


## Tavily Search Tool
search_tool = TavilySearchResults(max_results=2)
result=search_tool.invoke("How to make a cup of tea")

## Custome tool for getting weather data
@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )

## Create a ChatOpenAI instance
llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPEN_AI_API_KEY)

result=llm.predict("Write a short poem about the beauty of nature.")

prompt=hub.pull("hwchase17/react")

tools=[search_tool,get_weather_data]

agent=create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor=AgentExecutor(agent=agent, tools=tools, verbose=True)

response=agent_executor.invoke({"input":("Find the captial of India and then get the current weather in that city.")})

print(response["output"])
