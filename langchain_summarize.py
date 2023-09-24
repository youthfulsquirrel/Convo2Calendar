from langchain import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer

llm = LLM(
    name="t5-base",
    model=AutoModelForCausalLM.from_pretrained("t5-base"),
    tokenizer=AutoTokenizer.from_pretrained("t5-base"),
)

from langchain import load_summarize_chain

chain = load_summarize_chain(llm, chain_type="map_reduce")
docs = ["In this meeting, the participants discuss the effects of AI on note-taking in meetings, specifically focusing on the transcriber tool developed by Aaron Kaplan. The tool allows users to transcribe meetings and receive meeting minutes with key points, action items, a summary, and sentiment analysis. The tool utilizes OpenAI's whisper and API, but can also be used with local LLMs or a local installation of whisper. Aaron Kaplan requests assistance in developing the tool further, including adding more unit tests, improving the user interface, and providing an example of using the tool with a local installation of whisper and an LLM. The tool is Docker enabled."]
summaries = chain.run(docs)
