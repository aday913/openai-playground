import csv
import logging
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
import openai


def get_functions_for_client():
    all_tools = []
    for file in os.listdir("."):
        if not file.endswith(".json"):
            continue
        logging.info(f"Reading file: {file} for function config")
        with open(file) as f:
            data = json.load(f)
            logging.debug(
                f"When reading file {file}, found the following data:\n\n{data}\n\n"
            )
            all_tools.append(data)
    logging.info(f"Found {len(all_tools)} tools")
    return all_tools


def get_concert_info(user_name: str):
    all_concerts = []
    with open("concert_data.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["UserName"].lower() == user_name:
                row.pop("UserName")
                all_concerts.append([f"{key}: {value}" for key, value in row.items()])
    logging.info(f"Found {len(all_concerts)} concerts for {user_name}")
    logging.debug(f"Concerts: {all_concerts}")
    return all_concerts


def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def csv_data_fetcher_conversation(client: OpenAI, tools: list):
    # Create a conversation with the user asking for concert information
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant who fetches concert information for users.",
        },
        {"role": "user", "content": "Can you fetch the concert info for Jack?"},
    ]
    
    # Call the openai api to get a response trying to call the tool
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )
    logging.info(f"\nResponse: {response}\n")

    # Extract the tool call from the response
    tool_call = response.choices[0].message.tool_calls[0]
    logging.info(f"Tool call: {tool_call}\n")
    arguments = json.loads(tool_call.function.arguments)

    # Extract the user name from the arguments
    user_name = arguments.get("user_name")
    if not user_name:
        logging.error("User name not found in arguments")
        return
   
    # Get the concert information for the user
    concerts = get_concert_info(user_name)

    # Create a message to send back to the api
    result_message = {
        "role": "tool",
        "content": json.dumps({
            "user_name": user_name,
            "concert_info": concerts
        }),
        "tool_call_id": response.choices[0].message.tool_calls[0].id
    }

    # Append the response and the result message to the messages list
    messages.append(response.choices[0].message)
    messages.append(result_message)
    completion_payload = {
        "model": "gpt-4o",
        "messages": messages,
        "tools": tools
    }

    # Call the openai api to get a response with the concert information
    response = openai.chat.completions.create(
        model=completion_payload["model"],
        messages=completion_payload["messages"],
    )

    logging.info(f"\nComplete Response: {response}\n")
    if response.choices[0].message.content:
        logging.info(response.choices[0].message.content)
    return response


def main(api_key: str):
    client = get_client(api_key)
    logging.info("Client: %s", client)

    tools = get_functions_for_client()

    csv_data_fetcher_conversation(client, tools)

    # get_concert_info("Jack")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logging.info("Loading .env file")
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found in .env file")

    main(api_key)
