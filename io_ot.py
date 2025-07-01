# gemini_qa_bot.py

import os
import logging
from dotenv import load_dotenv
from telethon import TelegramClient, events
from pinecone import Pinecone
import google.generativeai as genai

# --- 1. SETUP AND INITIALIZATION ---

# Load environment variables from .env file
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Client Initialization ---
# Load credentials from .env file
API_ID = os.getenv("telegram_id")
API_HASH = os.getenv("telegram_hash")
BOT_TOKEN = os.getenv("iolo_token")
PINECONE_API_KEY = os.getenv("pinecone_api")
GEMINI_API_KEY = os.getenv("gemma_gemini_api")

# Initialize all clients
client = TelegramClient('bot_session', API_ID, API_HASH).start(bot_token=BOT_TOKEN)
pc = Pinecone(api_key=PINECONE_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# --- Pinecone and Gemini Model Setup ---
# This section is taken directly from your que_em.py logic
INDEX_NAME = "biology-grade-11"
dense_index = pc.Index(INDEX_NAME)
model = genai.GenerativeModel('gemini-1.5-flash') # Using a valid model name

logging.info(f"Connected to Pinecone index: '{INDEX_NAME}'")
logging.info("Gemini Model 'gemini-1.5-flash' initialized.")


# --- 2. QUERY AND ANSWER LOGIC (FROM que_em.py) ---
# This logic is preserved and adapted to return data instead of printing it.

def generate_content(query, contents):
    """
    Generates content based on the query and context using the Gemini model.
    This function's internal logic is unchanged.
    """
    system_prompt = f"""
    You are a specialized AI assistant. Your sole purpose is to answer the user's query based exclusively on the provided search results. You must adhere to the following instructions without deviation.

    **Instructions:**

    1.  **Analyze the User's Query:** The user wants to know: "{query}"

    2.  **Review the Provided Context:** You are given the following search results, each containing an ID, SCORE, PAGE_NUMBER, TEXT_HEADER, and TEXT_CONTENT.

        ```context
        {contents}
        ```

    3.  **Synthesize the Answer:**
        * Formulate a descriptive answer to the user's query using *only* the information found in the `TEXT_CONTENT` of the provided results.
        * The answer must be more than five sentences.
        * Do not invent, infer, or use any information outside of the provided context.
        * If you couldn't find a suitable answer to the {query}, just reply with "Im sorry, there isnt a suitale answer to your question in the book."
    4.  **Cite Your Sources:**
        * After the answer, list the sources you used.
        * For each source, you must include its `ID`, `SCORE`, and `PAGE_NUMBER`.
        * Format each citation exactly as: `Source: \n ID: [ID], SCORE: [SCORE],HEADER:[TEXT_HEADER] PAGE_NUMBER: [PAGE_NUMBER]`
    

    **Output Mandate:**

    * Your entire output must consist of two parts ONLY: the synthesized answer first, followed by the list of source citations.
    * DO NOT add any introductory phrases, greetings, apologies, or concluding remarks.
    * DO NOT use any formatting other than what is specified.
    """
    response = model.generate_content(system_prompt)
    return response.text # Changed from print() to return

async def process_query(query: str):
    """
    Performs the multi-namespace Pinecone search and gets the final answer.
    This function's internal logic is unchanged.
    """
    namespaces = ["key_words", "general_text", "tables"]
    all_contexts = []
    
    for ns in namespaces:
        # The search/query logic from your script is preserved
        results = dense_index.search(
            namespace=ns,
            query={
                "top_k": 5,
                "inputs": {
                    'text': query
                }
            }
        )
        formatted_results = "\n".join(
            f"ID: {hit.get('_id', 'N/A')} | SCORE: {round(hit.get('_score', 0), 2)} | PAGE_NUMBER: {hit.get('fields', {}).get('page_number', 'N/A')}\n"
            f"TEXT_HEADER: {hit.get('fields', {}).get('topic', 'N/A')}\n"
            f"TEXT_CONTENT: {hit.get('fields', {}).get('chunk_text', 'N/A')}\n\n"
            for hit in results.get('result', {}).get('hits', [])
        )
        all_contexts.append(formatted_results)
    
    full_context = "\n".join(all_contexts)
    
    # Pass the query and formatted context to the generation function
    final_answer = generate_content(query, full_context)
    return final_answer # Changed from print() to return


# --- 3. TELEGRAM BOT HANDLERS ---

@client.on(events.NewMessage(pattern='/start'))
async def start(event):
    """Handles the /start command."""
    welcome_message = (
        "Hello! I am a Q&A bot for the Grade 11 Biology curriculum.\n\n"
        "Please ask me a question, and I will find the answer for you from the textbook."
    )
    await event.respond(welcome_message)
    logging.info(f"Started new session for chat_id: {event.chat_id}")


@client.on(events.NewMessage)
async def message_handler(event):
    """Handles all non-command text messages."""
    # Ignore any message that starts with '/', which is a command
    if event.text.startswith('/'):
        return

    user_query = event.text
    chat_id = event.chat_id
    logging.info(f"Received query from chat_id {chat_id}: '{user_query}'")

    # Inform the user that the bot is processing the request
    async with client.action(chat_id, 'typing'):
        try:
            # Call the integrated query processing logic
            response_text = await process_query(user_query)
            await event.respond(response_text)
            logging.info(f"Successfully sent response to chat_id {chat_id}")
        except Exception as e:
            logging.error(f"An error occurred while processing query for chat_id {chat_id}: {e}")
            await event.respond("I'm sorry, an unexpected error occurred while processing your request. Please try again later.")


# --- 4. MAIN EXECUTION BLOCK ---

async def main():
    """Main function to run the bot."""
    logging.info("Bot is starting up...")
    await client.run_until_disconnected()
    logging.info("Bot has stopped.")

if __name__ == '__main__':
    client.loop.run_until_complete(main())