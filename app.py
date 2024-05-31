import openai
from openai import OpenAI
import streamlit as st

# Set streamlit page configuration
st.set_page_config(page_title="Sentiment Insights", page_icon="💬")

# Get the OpenAI API Key from the user
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Set the page title and caption
st.title("🌏💬 Multi-Language Sentiment Analyzer")
st.caption(
    "Analyze and Understand Emotions in Your Language: Positive 😁, Negative 😞, Neutral 😐"
)

# Define OpenAI's LLM to use in session state variable
if "openai_model" not in st.session_state:
    st.session_state.model = "gpt-3.5-turbo"

# Setting the delimiter for input text:
delimiter = "####"

# System prompt for the LLM
SYSTEM_PROMPT = f"""
You are a sentiment analysing agent.

You will be provided with text by user. That text will be delimited by {delimiter} characters. You have to analyze the text and output the sentiment from the text and also a 1-2 sentence explanation on why that's the sentiment.

Do not analyze text that include any harmful, racist, sexist, toxic, dangerous, or illegal content. Tell them that you detected either of these: harmful, racist, sexist, toxic, dangerous, or illegal content and also instruct them to use the app responsibly.

Also, your response should not contain any harmful, racist, sexist, toxic, dangerous, or illegal content.

If the text is not in English, first translate it to English. If input text is more than 100 words, do not translate. The explanation(reason) should also some parts of text which is useful for sentiment analysis.

Possible sentiment values are 'Positive', 'Negative', 'Neutral'. Preserve the case.
Do not answer any other question of user irrelevant to sentiment analysis.

The emojis for different sentiments are given below:
1. Positive: 😁
2. Negative: 😞
3. Neutral: 😐

The output format for english text is shown delimited below by triple backticks:
```
Sentiment: <sentiment> <emoji>

Reason: <Reason for the sentiment>
```

The output format for non-english text is shown below delimited by triple backticks:
Note: If the input text is greater than 100 words omit the 'Translation' section below.
```
Sentiment: <sentiment> <emoji>

Translation: <Translation in English>

Reason: <Reason for the sentiment>
```

Output should not contain triple backticks.

Only use this format if sentiment analysis can be performed on text. Do not use it for irrelavent user queries.
"""


# Check input text for harmful content
def moderate_input(client: OpenAI, input_text: str) -> bool:
    response = client.moderations.create(input=input_text)
    if response.results[0].flagged:
        for category, value in response.results[0].categories:
            if value:
                return True, category
    else:
        return False, "Nothing"


# Function to handle the sentiment analysis
def get_sentiment(input_text: str) -> str:
    try:
        client = OpenAI(api_key=openai_api_key)
        harmful_content, category = moderate_input(client=client, input_text=input_text)

        if not harmful_content:
            messages = [
                {"role": "system", "content": f"{SYSTEM_PROMPT}"},
                {"role": "user", "content": f"{delimiter}{input_text}{delimiter}"},
            ]
            response = client.chat.completions.create(
                model=st.session_state.model,
                messages=messages,
                temperature=0,
                max_tokens=200,
                n=1,
            )
            st.info(response.choices[0].message.content)

        else:
            response = client.chat.completions.create(
                model=st.session_state.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"Harmful text of category: {category} is detected in user's input. Tell the user the same in 1-2 sentences. Tell them to use the product responsibly",
                    }
                ],
                temperature=0.8,
                max_tokens=100,
                n=1,
            )
            st.info(response.choices[0].message.content)

    except openai.BadRequestError as e:
        st.warning(e.body["message"], icon="⚠️")

    except openai.AuthenticationError as e:
        st.warning(e.body["message"], icon="⚠️")

    except openai.PermissionDeniedError as e:
        st.warning(e.body["message"], icon="⚠️")

    except openai.NotFoundError as e:
        st.warning(e.body["message"], icon="⚠️")

    except openai.UnprocessableEntityError as e:
        st.warning(e.body["message"], icon="⚠️")

    except openai.RateLimitError as e:
        st.warning(e.body["message"], icon="⚠️")

    except openai.InternalServerError as e:
        st.warning(e.body["message"], icon="⚠️")

    except openai.APIConnectionError as e:
        st.warning(e.body["message"], icon="⚠️")


# Starting point of the program
def main():
    with st.form("sentiment-chat"):
        input_text = st.text_area(
            "Enter text for analysis: ",
            placeholder="e.g., 'I love this product!' or 'This is frustrating'",
            height=150,
        )
        submitted = st.form_submit_button("Submit")

        if not openai_api_key.startswith("sk-"):
            st.warning("Please enter a valid OpenAI API Key!", icon="⚠️")

        if openai_api_key.startswith("sk-") and submitted:
            with st.spinner("Analysing..."):
                get_sentiment(input_text)


if __name__ == "__main__":
    main()
