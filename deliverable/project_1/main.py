import streamlit as st
import anthropic
import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

def search_serpapi(query: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Search using SerpAPI for the given query and return the results.

    :param query: The search query string.
    :param api_key: Your SerpAPI key.
    :return: A list of search results.
    :raises Exception: For any errors during the request.
    """
    try:
        search = GoogleSearch({
            "q": query,
            "location": "Austin, Texas, United States",
            "api_key": api_key
        })
        results = search.get_dict()
        return results.get("organic_results", [])
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

st.title("Claude Chatbot")

# Sidebar checkbox for internet search
with st.sidebar:
    use_serpapi = st.checkbox("Enable Internet Search (SerpAPI)", value=False)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Placeholder for search results display
    search_results_container = st.empty()

    # Get response from Claude
    try:
        # If SerpAPI is enabled, perform search and add results to context
        search_results_data = None
        if use_serpapi:
            serpapi_key = os.getenv("SERPAPI_API_KEY")
            if serpapi_key:
                try:
                    search_results = search_serpapi(prompt, serpapi_key)
                    if search_results:
                        search_context = "\n\nInternet Search Results:\n"
                        search_results_data = []

                        for result in search_results[:5]:  # Limit to top 5 results
                            # Extract all relevant fields with proper error handling
                            title = result.get('title', 'No title available')
                            link = result.get('link', '#')
                            snippet = result.get('snippet', '')
                            displayed_link = result.get('displayed_link', result.get('display_link', ''))
                            position = result.get('position', 0)

                            # Store formatted results for display
                            search_results_data.append({
                                'title': title,
                                'link': link,
                                'snippet': snippet,
                                'displayed_link': displayed_link,
                                'position': position
                            })

                            # Build context for Claude
                            search_context += f"- [{position}] {title}\n"
                            search_context += f"  URL: {link}\n"
                            if snippet:
                                search_context += f"  Summary: {snippet}\n"
                            search_context += "\n"

                        # Add search context to the last user message
                        st.session_state.messages[-1]["content"] += search_context

                        # Display search results in the UI
                        with search_results_container.container():
                            st.info(f"🔍 Found {len(search_results_data)} search results")
                            with st.expander("View Search Results", expanded=True):
                                for idx, result in enumerate(search_results_data, 1):
                                    st.markdown(f"**{idx}. [{result['title']}]({result['link']})**")
                                    if result['displayed_link']:
                                        st.caption(f"🌐 {result['displayed_link']}")
                                    if result['snippet']:
                                        st.write(result['snippet'])
                                    st.divider()
                except Exception as search_error:
                    st.warning(f"Search error: {str(search_error)}")
            else:
                st.warning("SERPAPI_API_KEY not found in environment variables")

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=st.session_state.messages,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5
            }]
        )
        # Extract text blocks from response (filter out tool use blocks)
        response = ""
        for block in message.content:
            if block.type == "text":
                response += block.text
    except Exception as e:
        response = f"Error: {str(e)}"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
