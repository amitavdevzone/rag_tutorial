# RAG using Llama 3.1 & ChromaDB

This is a RAG system which creates a RAG system where we can scrape any web URL. Once done, we can ask questions based on the information that was scrapred.

This application uses

1. Llama 3.1 model (we are expecting the ollama service is running locally)
2. Chroma DB to store the embeddings

# Instructions on using

To start the Flask APIs, we just need to run `python app.py`

There are 3 APIs that this application exposes

## Scrape a URL

Pass it a valid web URL and it will try to crawl the information.

```curl
curl  -X POST \
  'http://localhost:8080/scrape' \
  --header 'Accept: */*' \
  --header 'User-Agent: Thunder Client (https://www.thunderclient.com)' \
  --header 'Content-Type: application/json' \
  --data-raw '{
  "url": "https://www.amitavroy.com/about"
}'
```

## Ask a question

This URL will allow you to ask a question to the bot based on the information crawled.

```curl
curl  -X POST \
  'http://localhost:8080/ask_bot' \
  --header 'Accept: */*' \
  --header 'User-Agent: Thunder Client (https://www.thunderclient.com)' \
  --header 'Content-Type: application/json' \
  --data-raw '{
  "question": "Who is Amitav Roy?"
}'
```

## Training based on the intent CSV

The CSV is inside the data folder. To generate the embeddings, run `python intent.py`

## Get intent

This URL assumes the basic intent training is done.

```curl
curl  -X POST \
  'http://localhost:8080/get_intent' \
  --header 'Accept: */*' \
  --header 'User-Agent: Thunder Client (https://www.thunderclient.com)' \
  --header 'Content-Type: application/json' \
  --data-raw '{
  "question": "do you have shoes for sports?"
}'
```
