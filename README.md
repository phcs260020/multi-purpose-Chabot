# VisionChat – Interactive Image Recognition Chatbot

An interactive, multi-purpose Streamlit chatbot that accepts **both text and image inputs**.  
It uses a pre-trained **YOLOv8** model to identify objects in uploaded images and a conversational layer that answers questions about the recognition results.

## Features

- **Image upload** – drag-and-drop or sidebar uploader (JPG, PNG, BMP, WebP)
- **Object detection** – YOLOv8 (nano or small) trained on the COCO dataset (80 common object classes)
- **Annotated visualization** – bounding boxes + labels drawn on the image
- **Conversational interface** – modern chat UI with history
- **Context-aware answers** – ask questions such as:
  - “What do you see?”
  - “How many people are there?”
  - “Is there a dog?”
  - “List all objects”
  - “What is the most confident detection?”
- **General chat** – greetings, help, thanks, goodbye
- **Adjustable confidence threshold** and model size (nano / small)
- **Clear chat & image** button

## Quick Start

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The first run will automatically download the selected YOLOv8 weights (`yolov8n.pt` ≈ 6 MB).

Open the URL shown in the terminal (usually http://localhost:8501).

## How to Use

1. **Upload an image** via the sidebar or the main area.
2. Wait a moment while YOLOv8 detects objects. An annotated image and a summary appear in the chat.
3. **Ask questions** about the detected objects (or just chat).
4. Change the confidence threshold or switch to the larger `yolov8s` model in the sidebar for different results.
5. Click **Clear chat & image** to start over.

## Project Structure

```
image_chatbot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Technical Notes

- **Model**: Ultralytics YOLOv8 (pre-trained on MS COCO).  
  - `yolov8n.pt` – fastest, good for most demos  
  - `yolov8s.pt` – higher accuracy, slightly slower
- **UI**: Streamlit chat components (`st.chat_message`, `st.chat_input`)
- **State**: Session state keeps chat history, current detections and the annotated image
- **Conversation logic**: Lightweight rule-based + keyword matching that uses the current detection list. No external LLM API is required, so the app runs fully offline after the model weights are downloaded.

## Example Conversation

```
User: [uploads a photo of a street]
Bot: ✅ Image analyzed!
     I detected the following objects:
     • 3 persons (avg confidence ≈ 87%)
     • 2 cars (avg confidence ≈ 91%)
     • 1 traffic light …

User: How many cars?
Bot: I found 2 cars in the image.

User: Is there a bicycle?
Bot: No, I don't see that. The objects I found are: person, car, traffic light.
```

## Extending the App

- Swap in a larger YOLO model (`yolov8m.pt`, `yolov8l.pt`, …) or a custom-trained model.
- Replace the rule-based responder with a local LLM (e.g. via Ollama) or a vision-language model for richer answers.
- Add camera input with `st.camera_input` for live snapshots.
- Support video by iterating frames with YOLO’s tracking mode.

## License

MIT – feel free to use, modify and share.
