import base64
import os
import time

# Existing imports remain the same
from abc import ABC, abstractmethod
from typing import Any, Dict

import cv2
from dotenv import load_dotenv
from nexa.gguf import NexaTextInference, NexaVLMInference  # Import model classes
from openai import OpenAI

from data_processing_common import (
    compute_operations,
    execute_operations,
    process_files_by_date,
    process_files_by_type,
)
from file_utils import (
    collect_file_paths,
    display_directory_tree,
    read_file_data,
    separate_files_by_type,
)
from image_data_processing import process_image_files
from output_filter import filter_specific_output  # Import the context manager
from text_data_processing import process_text_files

# Load environment variables
load_dotenv()


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


def ensure_nltk_data():
    """Ensure that NLTK data is downloaded efficiently and quietly."""
    import nltk

    required_packages = [
        "stopwords",
        "punkt",
        "wordnet",
        "punkt_tab",  # Added this package
        "averaged_perceptron_tagger",  # Also good to have for text processing
    ]

    for package in required_packages:
        try:
            nltk.download(package, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {package}: {e}")
            # Continue anyway as some packages might be optional


# Initialize models
image_inference = None
text_inference = None


def initialize_models():
    """Initialize the models if they haven't been initialized yet."""
    global image_inference, text_inference
    if image_inference is None or text_inference is None:
        use_openai = os.getenv("USE_OPENAI", "false").lower() == "true"
        max_tokens = int(os.getenv("MAX_TOKENS", 3000))

        if use_openai:
            # Initialize OpenAI models
            image_inference = OpenAIInference(
                model_name="gpt-4o", temperature=0.3, max_tokens=max_tokens
            )
            text_inference = OpenAIInference(
                model_name="gpt-4o", temperature=0.5, max_tokens=max_tokens
            )
        else:
            # Initialize local Nexa models
            with filter_specific_output():
                # Create the base Nexa models
                nexa_image = NexaVLMInference(
                    model_path="llava-v1.6-vicuna-7b:q4_0",
                    local_path=None,
                    stop_words=[],
                    temperature=0.3,
                    max_new_tokens=max_tokens,
                    top_k=3,
                    top_p=0.2,
                    profiling=False,
                    nctx=4096,
                )

                nexa_text = NexaTextInference(
                    model_path="Llama3.2-3B-Instruct:q3_K_M",
                    local_path=None,
                    stop_words=[],
                    temperature=0.5,
                    max_new_tokens=max_tokens,
                    top_k=3,
                    top_p=0.3,
                    profiling=False,
                    nctx=8192,
                )

                # Wrap them in our common interface
                image_inference = NexaWrapper(nexa_image, "image")
                text_inference = NexaWrapper(nexa_text, "text")

        print("**----------------------------------------------**")
        print(f"**  {'OpenAI' if use_openai else 'Local'} models initialized  **")
        print("**----------------------------------------------**")


def simulate_directory_tree(operations, base_path):
    """Simulate the directory tree based on the proposed operations."""
    tree = {}
    for op in operations:
        rel_path = os.path.relpath(op["destination"], base_path)
        parts = rel_path.split(os.sep)
        current_level = tree
        for part in parts:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
    return tree


def print_simulated_tree(tree, prefix=""):
    """Print the simulated directory tree."""
    pointers = ["├── "] * (len(tree) - 1) + ["└── "] if tree else []
    for pointer, key in zip(pointers, tree):
        print(prefix + pointer + key)
        if tree[key]:  # If there are subdirectories or files
            extension = "│   " if pointer == "├── " else "    "
            print_simulated_tree(tree[key], prefix + extension)


def get_model_selection():
    """Prompt the user to select which model to use."""
    while True:
        print("\nPlease choose which model to use:")
        print("1. Local Models (Nexa)")
        print("2. OpenAI API")
        response = input("Enter 1 or 2 (or type '/exit' to exit): ").strip()
        if response == "/exit":
            print("Exiting program.")
            exit()
        elif response == "1":
            os.environ["USE_OPENAI"] = "false"
            return
        elif response == "2":
            os.environ["USE_OPENAI"] = "true"
            return
        else:
            print("Invalid selection. Please enter 1 or 2.")


def get_yes_no(prompt):
    """Prompt the user for a yes/no response."""
    while True:
        response = input(prompt).strip().lower()
        if response in ("yes", "y"):
            return True
        elif response in ("no", "n"):
            return False
        elif response == "/exit":
            print("Exiting program.")
            exit()
        else:
            print("Please enter 'yes' or 'no'. To exit, type '/exit'.")


def get_mode_selection():
    """Prompt the user to select a mode."""
    while True:
        print("Please choose the mode to organize your files:")
        print("1. By Content")
        print("2. By Date")
        print("3. By Type")
        response = input("Enter 1, 2, or 3 (or type '/exit' to exit): ").strip()
        if response == "/exit":
            print("Exiting program.")
            exit()
        elif response == "1":
            return "content"
        elif response == "2":
            return "date"
        elif response == "3":
            return "type"
        else:
            print("Invalid selection. Please enter 1, 2, or 3. To exit, type '/exit'.")


def main():
    # Ensure NLTK data is downloaded efficiently and quietly
    ensure_nltk_data()

    # Start with dry run set to True
    dry_run = True

    # Display silent mode explanation before asking
    print("-" * 50)
    print(
        "**NOTE: Silent mode logs all outputs to a text file instead of displaying them in the terminal."
    )
    silent_mode = get_yes_no("Would you like to enable silent mode? (yes/no): ")
    if silent_mode:
        log_file = "operation_log.txt"
    else:
        log_file = None

    while True:
        # Paths configuration
        if not silent_mode:
            print("-" * 50)

        # Get input and output paths once per directory
        input_path = input(
            "Enter the path of the directory you want to organize: "
        ).strip()
        while not os.path.exists(input_path):
            message = (
                f"Input path {input_path} does not exist. Please enter a valid path."
            )
            if silent_mode:
                with open(log_file, "a") as f:
                    f.write(message + "\n")
            else:
                print(message)
            input_path = input(
                "Enter the path of the directory you want to organize: "
            ).strip()

        # Confirm successful input path
        message = f"Input path successfully uploaded: {input_path}"
        if silent_mode:
            with open(log_file, "a") as f:
                f.write(message + "\n")
        else:
            print(message)
        if not silent_mode:
            print("-" * 50)

        # Default output path is a folder named "organized_folder" in the same directory as the input path
        output_path = input(
            "Enter the path to store organized files and folders (press Enter to use 'organized_folder' in the input directory): "
        ).strip()
        if not output_path:
            # Get the parent directory of the input path and append 'organized_folder'
            output_path = os.path.join(os.path.dirname(input_path), "organized_folder")

        # Confirm successful output path
        message = f"Output path successfully set to: {output_path}"
        if silent_mode:
            with open(log_file, "a") as f:
                f.write(message + "\n")
        else:
            print(message)
        if not silent_mode:
            print("-" * 50)

        # Start processing files
        start_time = time.time()
        file_paths = collect_file_paths(input_path)
        end_time = time.time()

        message = f"Time taken to load file paths: {end_time - start_time:.2f} seconds"
        if silent_mode:
            with open(log_file, "a") as f:
                f.write(message + "\n")
        else:
            print(message)
        if not silent_mode:
            print("-" * 50)
            print("Directory tree before organizing:")
            display_directory_tree(input_path)

            print("*" * 50)

        # Loop for selecting sorting methods
        while True:
            mode = get_mode_selection()

            if mode == "content":
                # Proceed with content mode
                # Initialize models once
                if not silent_mode:
                    print(
                        "Checking if the model is already downloaded. If not, downloading it now."
                    )
                get_model_selection()
                initialize_models()

                if not silent_mode:
                    print("*" * 50)
                    print(
                        "The file upload was successful. Processing may take a few minutes."
                    )
                    print("*" * 50)

                # Prepare to collect link type statistics
                link_type_counts = {"hardlink": 0, "symlink": 0}

                # Separate files by type
                image_files, text_files = separate_files_by_type(file_paths)

                # Prepare text tuples for processing
                text_tuples = []
                for fp in text_files:
                    # Use read_file_data to read the file content
                    text_content = read_file_data(fp)
                    if text_content is None:
                        message = f"Unsupported or unreadable text file format: {fp}"
                        if silent_mode:
                            with open(log_file, "a") as f:
                                f.write(message + "\n")
                        else:
                            print(message)
                        continue  # Skip unsupported or unreadable files
                    text_tuples.append((fp, text_content))

                # Process files sequentially
                data_images = process_image_files(
                    image_files,
                    image_inference,
                    text_inference,
                    silent=silent_mode,
                    log_file=log_file,
                )
                data_texts = process_text_files(
                    text_tuples, text_inference, silent=silent_mode, log_file=log_file
                )

                # Prepare for copying and renaming
                renamed_files = set()
                processed_files = set()

                # Combine all data
                all_data = data_images + data_texts

                # Compute the operations
                operations = compute_operations(
                    all_data, output_path, renamed_files, processed_files
                )

            elif mode == "date":
                # Process files by date
                operations = process_files_by_date(
                    file_paths,
                    output_path,
                    dry_run=False,
                    silent=silent_mode,
                    log_file=log_file,
                )
            elif mode == "type":
                # Process files by type
                operations = process_files_by_type(
                    file_paths,
                    output_path,
                    dry_run=False,
                    silent=silent_mode,
                    log_file=log_file,
                )
            else:
                print("Invalid mode selected.")
                return

            # Simulate and display the proposed directory tree
            print("-" * 50)
            message = "Proposed directory structure:"
            if silent_mode:
                with open(log_file, "a") as f:
                    f.write(message + "\n")
            else:
                print(message)
                print(os.path.abspath(output_path))
                simulated_tree = simulate_directory_tree(operations, output_path)
                print_simulated_tree(simulated_tree)
                print("-" * 50)

            # Ask user if they want to proceed
            proceed = get_yes_no(
                "Would you like to proceed with these changes? (yes/no): "
            )
            if proceed:
                # Create the output directory now
                os.makedirs(output_path, exist_ok=True)

                # Perform the actual file operations
                message = "Performing file operations..."
                if silent_mode:
                    with open(log_file, "a") as f:
                        f.write(message + "\n")
                else:
                    print(message)
                execute_operations(
                    operations, dry_run=False, silent=silent_mode, log_file=log_file
                )

                message = "The files have been organized successfully."
                if silent_mode:
                    with open(log_file, "a") as f:
                        f.write("-" * 50 + "\n" + message + "\n" + "-" * 50 + "\n")
                else:
                    print("-" * 50)
                    print(message)
                    print("-" * 50)
                break  # Exit the sorting method loop after successful operation
            else:
                # Ask if the user wants to try another sorting method
                another_sort = get_yes_no(
                    "Would you like to choose another sorting method? (yes/no): "
                )
                if another_sort:
                    continue  # Loop back to mode selection
                else:
                    print("Operation canceled by the user.")
                    break  # Exit the sorting method loop

        # Ask if the user wants to organize another directory
        another_directory = get_yes_no(
            "Would you like to organize another directory? (yes/no): "
        )
        if not another_directory:
            break  # Exit the main loop


if __name__ == "__main__":
    main()
