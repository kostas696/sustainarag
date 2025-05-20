import os
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.3,
    max_new_tokens=256
)

response = llm.invoke("What are the goals of sustainable development?")
print(response)
# Check if the response is not empty
assert response, "No response from the model."
# Check if the response contains relevant content
assert "sustainable development" in response.lower(), "Response does not contain relevant content."
# Check if the response is a string
assert isinstance(response, str), "Response is not a string."
# Check if the response does not contain any error messages
assert "error" not in response.lower(), "Response contains an error message."
# Check if the response is in a human-readable format
assert any(char.isalpha() for char in response), "Response is not in a human-readable format."