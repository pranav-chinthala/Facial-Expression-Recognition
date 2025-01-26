import cv2
import numpy as np
from deepface import DeepFace
import gradio as gr


emotion_dict = {
    0: {"label": "Angry", "emoji": "😠"},
    1: {"label": "Disgusted", "emoji": "🤢"},
    2: {"label": "Fearful", "emoji": "😨"},
    3: {"label": "Happy", "emoji": "😊"},
    4: {"label": "Neutral", "emoji": "😐"},
    5: {"label": "Sad", "emoji": "😢"},
    6: {"label": "Surprised", "emoji": "😲"}
}

def process_image(image):
    try:
        if image is None:
            return "Please upload an image."


        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        result = DeepFace.analyze(img,
                                actions=['emotion'],
                                enforce_detection=False)


        emotions = result[0]['emotion']
        

        output_text = "### Detected Emotions:\n\n"
        for emotion, score in emotions.items():
            # Find matching emoji from our dictionary
            emoji = next((item['emoji'] for item in emotion_dict.values() 
                         if item['label'].lower() == emotion.lower()), "")
            output_text += f"- {emoji} **{emotion}**: {score:.2f}%\n"

        return output_text

    except Exception as e:
        return f"Error analyzing image: {str(e)}"


with gr.Blocks(title="Emotion Detection", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 😊 EmotiTrack: Real-time Emotion Detection
    
    A professional tool designed to assist therapists and counselors in tracking emotional states during sessions.
    """)
    
    with gr.Tabs():
        with gr.Tab("🖼️ Upload Image"):
            with gr.Row():
                image_input = gr.Image(type="numpy", label="Upload Image")
            with gr.Row():
                output_text = gr.Markdown()
            image_button = gr.Button("Detect Emotion", variant="primary")
            
            image_button.click(
                fn=process_image,
                inputs=image_input,
                outputs=output_text
            )

        with gr.Tab("👥 About Us"):
            gr.Markdown("""
            # About EmotiTrack

            ## Our Mission
            EmotiTrack is developed to address a critical gap in mental health services - the need for objective emotion tracking during therapy and counseling sessions. Our tool helps mental health professionals monitor and analyze their clients' emotional states in real-time, enabling more informed and effective therapeutic interventions.

            ## Our Guide

            - **Ajay Chajed**
              - E-mail: 

            ## Our Team 
                       
            - **Priyal Bhusate**
              - Group Leader
              - E-mail: priyal.bhusate241@vit.edu

            - **Aditya Ulangwar**
              - Assistant Group Leader
              - E-mail: jitendra.aditya24@vit.edu

            - **Pranav Chinthala**
              - E-mail: pranav.chinthala24@vit.edu
                        
            - **Shivam Aher**
              - E-mail: shivam.aher242@vit.edu
                        
            - **Kanak Bansal**
              - E-mail: kanak.bansal24@vit.edu

            - **Urvee Bhagwat**
              - E-mail: urvee.bhagwat241@vit.edu
            
            ## Objective
            Our tool aims to:
            - Provide objective emotional state tracking during therapy sessions
            - Help therapists identify emotional patterns over time
            - Enable data-driven insights for better treatment planning
            - Support early intervention through emotion pattern recognition
            
            ## Technology
            - This model is using a Convolutional Neural Network (CNN) trained on the FER2013 dataset which contains 28,709 48x48 pixel grayscale labeled images of faces in the training set and 3,589 48x48 pixel grayscale labeled images of faces in the testing set, categorized into 7 different facial expressions (Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral).
            - Multiple Python libraries were used including:
              - Keras (TensorFlow)
              - OpenCV
              - NumPy
              - Pillow
              - Gradio                                               
            """)

    gr.Markdown("""
    ### How to use:
    1. Upload an image using the Upload Image tab
    2. Click "Detect Emotion" to analyze the image
    3. The model will detect faces and display the emotions with confidence scores
    
    ### Note:
    - Make sure faces are clearly visible and well-lit
    - The model works best with front-facing faces
    - Multiple faces can be detected in a single image
    """)

if __name__ == "__main__":
    app.launch(share=True)