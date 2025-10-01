import os
from dotenv import load_dotenv
import assemblyai as aai
import voyageai
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole

load_dotenv()

api_key = os.getenv("ASSEMBLYAI_API_KEY")
aai.settings.api_key = api_key

audio_file = "https://assembly.ai/wildfires.mp3"

# audio_file = "資訊網後台 影音模組 2025年4月21日.mp3"

config = aai.TranscriptionConfig(speech_model=aai.SpeechModel.universal, speaker_labels=True)

transcript = aai.Transcriber(config=config).transcribe(audio_file)

speaker_transcripts = []
for utterance in transcript.utterances:
    speaker_transcripts.append({
        "speaker": f"Speaker {utterance.speaker}",
        "text": utterance.text
    })

speaker_sents = [f"{item['speaker']}: {item['text']}" for item in speaker_transcripts]

# ===== Embed and Store =====
# create voyage client
vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

# Generate embeddings
embeds = vo.embed(texts=speaker_sents,
                  model='voyage-context-3',
                  input_type='document'
                  ).embeddings

# docs to store in a MongoDB collection
docs = []
for (transcript, embed) in zip(speaker_sents, embeds):
    docs.append({
        "text": transcript,
        "embedding": embed
    })

# Connect to store in a MongoDB collection
connection_string = os.getenv("MONGODB_CONNECTION_STRING")
client = MongoClient(connection_string)
collection = client["rag_db"]["test"]

# Insert documents into the collection
result = collection.insert_many(docs)

# ===== Query Collection =====
# create the search index
search_index_model = SearchIndexModel(
    definition={"field": [{"type": "vector", "numDimensions": 1536,
                           "path": "embeddings", "similarity": "cosine"}]},
    name="vector_index",
    type="vectorSearch"
)

collection.create_search_index(model=search_index_model)

# Generate query embedding
query_embed = vo.embed(["Which companies were found by Andrew Ng?"],
                       model='voyage-context-3',
                       input_type='query'
                       ).embeddings[0]

# Perform vector search
results = collection.aggregate({"SvectorSearch": {"index": "vector_index",
                                                  "queryVector": query_embed,
                                                  "path": "embedding",
                                                  "limit": 5,
                                                  "exact": True}},
                               {"$project": {"_id": 0, "text": 1}}
                               )
context = list(results)

# ===== Generate Response =====
# Define context
merged_context = "\n\n---\n\n".join([i['text'] for i in results])
# Set up LLM
llm_name = "gpt-oss"
llm = Ollama(model=llm_name)

# Construct query prompt
prompt = (f"Context information is below.\n"
          f" ----------------------- \n"
          f"{merged_context}\n"
          f" ----------------------- \n"
          f"Given the context information above, think step by step "
          f"to answer the query in a crisp manner.\n"
          f"Query: {query}\n"
          f"Answer: ")

# Generate response
user_msg = ChatMessage(role=MessageRole.User, content=prompt)
response = llm.stream_complete(user_msg.content)
