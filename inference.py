import base64
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

import cv2
from openai import OpenAI


class BaseInference(ABC):
    @abstractmethod
    def create_completion(self, prompt: str) -> str:
        pass

    @abstractmethod
    def analyze_content(self, content: str, content_type: str) -> Dict[str, Any]:
        pass


class OpenAIInference(BaseInference):
    def __init__(
        self, model_name: str, temperature: float = 0.5, max_tokens: int = 3000
    ):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def encode_image(self, image_path):
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def encode_gif_frames(self, gif_path):
        """Encode GIF frames to base64."""
        video = cv2.VideoCapture(gif_path)
        base64Frames = []

        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

        video.release()
        return base64Frames

    def _chat(self, prompt: str, image_path=None):
        """Handle image-based prompts for OpenAI."""
        if not image_path:
            return self.create_completion(prompt)

        # Check if it's a GIF
        _, ext = os.path.splitext(image_path.lower())
        if ext == ".gif":
            # Handle GIF by analyzing key frames
            frames = self.encode_gif_frames(image_path)
            if not frames:
                return "Unable to process GIF file."

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *[
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{frame}"},
                            }
                            for frame in frames[0::50]
                        ],  # Sample every 50th frame
                    ],
                }
            ]
        else:
            # Handle static images
            base64_image = self.encode_image(image_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return f"Error analyzing image: {str(e)}"

    def create_completion(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def analyze_content(self, content: str, content_type: str) -> Dict[str, Any]:
        if content_type == "image":
            # For images, use the _chat method with the image path
            description = self._chat(
                "Please provide a detailed description of this image.", content
            )
            return {"category": description}

        # For text content
        prompt = f"Analyze this {content_type} content and provide categorization: {content[:1000]}"
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return {"category": response.choices[0].message.content.strip()}


class NexaWrapper(BaseInference):
    def __init__(self, nexa_model, model_type: str):
        self.model = nexa_model
        self.model_type = model_type

    def create_completion(self, prompt: str) -> str:
        # Using the correct method name from NexaTextInference
        response = self.model.create_completion(prompt)
        # Handle response format
        if isinstance(response, dict) and "choices" in response:
            return response["choices"][0]["text"].strip()
        return response

    def analyze_content(self, content: str, content_type: str) -> Dict[str, Any]:
        if self.model_type == "image":
            response = self.model.analyze_image(content)
        else:
            response = self.model.create_completion(
                content
            )  # Changed from generate to create_completion

        if isinstance(response, dict) and "choices" in response:
            return {"category": response["choices"][0]["text"].strip()}
        return {"category": str(response).strip()}

    def _chat(self, prompt: str, image_path=None) -> str:
        if self.model_type == "image":
            return self.model._chat(prompt, image_path)
        else:
            # Use create_chat_completion for text model
            response = self.model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
            if isinstance(response, dict) and "choices" in response:
                return response["choices"][0]["message"]["content"].strip()
            return str(response).strip()
