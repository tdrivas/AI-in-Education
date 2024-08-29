import openai


class ChatGPTClient:
    """
    A client for interacting with OpenAI's ChatGPT API.

    Attributes:
        api_key (str): The API key for authenticating requests to OpenAI.
        model (str): The model to use for generating responses (e.g., 'gpt-3.5-turbo').
        messages (list): The conversation history stored as a list of message dictionaries.

    Methods:
        append_message(role, content):
            Appends a message to the conversation history.
        get_response(max_tokens=150, temperature=0.7):
            Sends the current conversation history to the API and retrieves a response.
    """

    def __init__(self, api_key, model="gpt-3.5-turbo"):
        """
        Initializes the ChatGPTClient with the given API key and model.

        Args:
            api_key (str): The API key for OpenAI.
            model (str, optional): The model to use for generating responses. Default is 'gpt-3.5-turbo'.

        """
        openai.api_key = api_key
        self.model = model
        self.messages = []

    def append_message(self, role, content):
        """
        Appends a message to the conversation history.

        Args:
            role (str): The role of the message sender. Must be one of 'system', 'user', or 'assistant'.
            content (str): The content of the message.

        Raises:
            ValueError: If the role is not 'system', 'user', or 'assistant'.

        """
        if role not in {"system", "user", "assistant"}:
            raise ValueError("Role must be one of 'system', 'user', or 'assistant'.")
        self.messages.append({"role": role, "content": content})

    def get_response(self, max_tokens=150, temperature=0.7):
        """
        Sends the current conversation history to the OpenAI API and retrieves a response.

        Args:
            max_tokens (int, optional): The maximum number of tokens to generate in the response. Default is 150.
            temperature (float, optional): Controls the randomness of the response. Default is 0.7.

        Returns:
            str: The content of the response from the assistant.

        Raises:
            openai.error.OpenAIError: If there is an issue with the API request.

        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            reply = response.choices[0].message["content"].strip()
            self.append_message(
                "assistant", reply
            )  # Add assistant's response to history
            return reply
        except openai.error.OpenAIError as e:
            print(f"An error occurred: {e}")
            return None


if __name__ == "__main__":
    #################################
    #   user's parameters to change #
    #################################
    api_key = "your-api-key"

    # use a loop for extended dialog
    message1 = "Can you help me explain Waves in Physics?"
    message2 = "Thanks! Give me now 5 simple exersices for beginners related to Harmonic Wave at a Fixed Position"

    ### end of user's parameters ###

    chat_client = ChatGPTClient(api_key)

    # just dummy data for our example
    chat_client.append_message("system", "You are a helpful assistant.")
    chat_client.append_message("user", message1)

    response = chat_client.get_response()
    if response:
        print(f"Response from ChatGPT:{response}")
    else:
        print("Failed to get a response from ChatGPT.")

    # another example
    chat_client.append_message(
        "user",
        message2,
    )
    response = chat_client.get_response()
    if response:
        print(f"Response from ChatGPT:{response}")
    else:
        print("Failed to get a response from ChatGPT.")
