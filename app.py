from flask import Flask, request
import scrape
import chat
import detect_intent

app = Flask(__name__)

@app.route("/scrape", methods=["POST"])
def scrapeUrl():
    json_content = request.json
    url = json_content.get("url")
    
    messages = scrape.fetch_and_persist_article(url)
    
    return {"url": url, "messages": messages}

@app.route("/ask_bot", methods=["POST"])
def askBot():
    json_content = request.json
    question = json_content.get("question")
    
    response = chat.answer_question_with_context(question)
    
    return response

@app.route("/get_intent", methods=["POST"])
def getIntent():
    json_content = request.json
    question = json_content.get("question")
    
    response = detect_intent.detect_intent_with_context(question)
    return {"question": question, "intent": response}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
