import openai

# Set your OpenAI API key
api_key = "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"
openai.api_key = api_key

# Set my org
openai.organization = "org-ADaggha6RcG4uPgRrlhTwxHd"

# Set the initial user message
user_message = """You are the spymaster in a popular word guessing game called Codenames.
There are four classes of words: blue words, red words, pale words, and black words.
As the spymaster, you want to provide clues to your teammate in the form of a single word followed by a space and then a number.
The word should be as similar in semantic meaning as possible to the other blue words while being as different as possible from the red words, pale words, and black words. It is especially important to choose a hint that is different from the black word.
The number after the space indicates the number of blue words that are related to your hint.
Do not choose a number that is greater than the number of blue words that are strongly related to your hint.
The selected words are shown below:
Blue words: beijing, pumpkin, england, tick, pistol, star, ambulance, space, london
Red words: post, snow, cover, round, helicopter, strike, court, foot
Pale words: greece, check, sock, princess, state, fence, berlin
Black word: Spike
Please respond with your hint in the format hint word followed by a space followed by the number of related blue words."""

# Create the conversation
conversation = [
    {"role": "user", "content": user_message}
]


response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",  # You can adjust the engine based on your preferences
    messages=conversation
)

# Extract and print the assistant's reply
assistant_reply = response.choices[0].message["content"]
print("Assistant:", assistant_reply)