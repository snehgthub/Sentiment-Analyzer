from openai import OpenAI, OpenAIError
import streamlit as st

# Set streamlit page configuration
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬")

# Get the OpenAI API Key from the user
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Set the page title and caption
st.title("🌏💬Multi-Language Sentiment Analyzer")
st.caption(
    "Analyze and Understand Emotions in Your Language: Positive 😁, Negative 😞, Neutral 😐"
)

# Define OpenAI's LLM to use in session state variable
if "openai_model" not in st.session_state:
    st.session_state.model = "gpt-3.5-turbo"


# System prompt for the LLM
SYSTEM_PROMPT = f"""
You are a sentiment analysing agent.

You will be provided with text by user. You have to analyze the text and output the sentiment from the text and also a 1-2 sentence explanation on why that's the sentiment.

If the text is not in English, first translate it to English. If input text is more than 100 words, do not translate. The explanation(reason) should also some parts of text which is useful for sentiment analysis.

Possible sentiment values are 'Positive', 'Negative', 'Neutral'. Preserve the case.
Do not answer any other question of user irrelevant to sentiment analysis.


The output format for english text is shown delimited below by triple backticks:
```
Sentiment: <sentiment>

Reason: <Reason for the sentiment>
```

The output format for non-english text is shown below delimited by triple backticks:
Note: If the input text is greater than 100 words omit the 'Translation' section below.
```
Sentiment: <sentiment>

Translation: <Translation in English>

Reason: <Reason for the sentiment>
```

Output should not contain triple backticks.

Only use this format if sentiment analysis can be performed on text. Do not use it for irrelavent user queries.
"""


# Function to handle the sentiment analysis
def get_sentiment(input_text: str) -> str:
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}"},
        {"role": "user", "content": f"{input_text}"},
    ]

    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model=st.session_state.model,
            messages=messages,
            temperature=0,
            max_tokens=200,
            n=1,
        )
        return response.choices[0].message.content

    except OpenAIError as e:
        return f"Error: {str(e)}"


# Starting point of the program
def main():
    with st.form("sentiment-chat"):
        input_text = st.text_area(
            "Enter text: ",
            placeholder="e.g., 'I love this product!' or 'This is frustrating'",
            height=150,
        )
        submitted = st.form_submit_button("Submit")

        if not openai_api_key.startswith("sk-"):
            st.warning("Please enter a valid OpenAI API Key!", icon="⚠️")

        if openai_api_key.startswith("sk-") and submitted:
            with st.spinner("Analysing..."):
                sentiment_result = get_sentiment(input_text)
                st.info(sentiment_result)


if __name__ == "__main__":
    main()
