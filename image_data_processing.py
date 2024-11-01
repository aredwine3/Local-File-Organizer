import os
import re
import time

from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from data_processing_common import sanitize_filename

load_dotenv()

MAX_FILENAME_LENGTH = int(os.getenv("MAX_FILENAME_LENGTH", 50))
MAX_FOLDERNAME_LENGTH = int(os.getenv("MAX_FOLDERNAME_LENGTH", 50))


def analyze_image(image_path, image_inference, prompt):
    """Analyze image with either OpenAI or Nexa model."""
    if hasattr(image_inference, "_chat"):
        # Using Nexa model
        try:
            description_generator = image_inference._chat(prompt, image_path)
            if hasattr(description_generator, "__iter__"):
                # If it's a generator (Nexa style)
                response_text = ""
                try:
                    for response in description_generator:
                        if isinstance(response, dict):
                            choices = response.get("choices", [])
                            for choice in choices:
                                delta = choice.get("delta", {})
                                if "content" in delta:
                                    response_text += delta["content"]
                        else:
                            response_text += str(response)
                except StopIteration:
                    pass
                return response_text.strip()
            return description_generator
        except Exception as e:
            print(f"Error during image analysis: {e}")
            return None
    else:
        # Using OpenAI model
        return image_inference.analyze_content(image_path, "image")["category"]


def process_single_image(
    image_path, image_inference, text_inference, silent=False, log_file=None
):
    """Process a single image file to generate metadata."""
    start_time = time.time()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task(
            f"Processing {os.path.basename(image_path)}", total=1.0
        )
        foldername, filename, description = generate_image_metadata(
            image_path, progress, task_id, image_inference, text_inference
        )

    end_time = time.time()
    time_taken = end_time - start_time

    message = f"File: {image_path}\nTime taken: {time_taken:.2f} seconds\nDescription: {description}\nFolder name: {foldername}\nGenerated filename: {filename}\n"
    if silent:
        if log_file:
            with open(log_file, "a") as f:
                f.write(message + "\n")
    else:
        print(message)
    return {
        "file_path": image_path,
        "foldername": foldername,
        "filename": filename,
        "description": description,
    }


def process_image_files(
    image_paths, image_inference, text_inference, silent=False, log_file=None
):
    """Process image files sequentially."""
    data_list = []
    for image_path in image_paths:
        data = process_single_image(
            image_path,
            image_inference,
            text_inference,
            silent=silent,
            log_file=log_file,
        )
        data_list.append(data)
    return data_list


def generate_image_metadata(
    image_path, progress, task_id, image_inference, text_inference
):
    """Generate description, folder name, and filename for an image file."""
    total_steps = 3

    # Step 1: Generate description
    description_prompt = "Please provide a detailed description of this image, focusing on the main subject and any important details."
    description = analyze_image(image_path, image_inference, description_prompt)
    progress.update(task_id, advance=1 / total_steps)

    # Step 2: Generate filename
    filename_prompt = f"""Based on the description below, generate a specific and descriptive filename for the image.
Limit the filename to a maximum of 3 words. Use nouns and avoid starting with verbs like 'depicts', 'shows', 'presents', etc.
Do not include any data type words like 'image', 'jpg', 'png', etc. Use only letters and connect words with underscores.

Description: {description}

Example:
Description: A photo of a sunset over the mountains.
Filename: sunset_over_mountains

Now generate the filename.

Output only the filename, without any additional text.

Filename:"""

    filename = text_inference.create_completion(filename_prompt).strip()
    filename = re.sub(r"^Filename:\s*", "", filename, flags=re.IGNORECASE).strip()
    progress.update(task_id, advance=1 / total_steps)

    # Step 3: Generate folder name
    foldername_prompt = f"""Based on the description below, generate a general category or theme that best represents the main subject of this image.
This will be used as the folder name. Limit the category to a maximum of 2 words. Use nouns and avoid verbs.
Do not include specific details, words from the filename, or any generic terms like 'untitled' or 'unknown'.

Description: {description}

Examples:
1. Description: A photo of a sunset over the mountains.
   Category: landscapes

2. Description: An image of a smartphone displaying a storage app with various icons and information.
   Category: technology

Now generate the category.

Output only the category, without any additional text.

Category:"""

    foldername = text_inference.create_completion(foldername_prompt).strip()
    foldername = re.sub(r"^Category:\s*", "", foldername, flags=re.IGNORECASE).strip()
    progress.update(task_id, advance=1 / total_steps)

    unwanted_words = set(
        [
            "the",
            "and",
            "based",
            "generated",
            "this",
            "is",
            "filename",
            "file",
            "image",
            "picture",
            "photo",
            "folder",
            "category",
            "output",
            "only",
            "below",
            "text",
            "jpg",
            "png",
            "jpeg",
            "gif",
            "bmp",
            "svg",
            "logo",
            "in",
            "on",
            "of",
            "with",
            "by",
            "for",
            "to",
            "from",
            "a",
            "an",
            "as",
            "at",
            "red",
            "blue",
            "green",
            "color",
            "colors",
            "colored",
            "text",
            "graphic",
            "graphics",
            "main",
            "subject",
            "important",
            "details",
            "description",
            "depicts",
            "show",
            "shows",
            "display",
            "illustrates",
            "presents",
            "features",
            "provides",
            "covers",
            "includes",
            "demonstrates",
            "describes",
        ]
    )
    stop_words = set(stopwords.words("english"))
    all_unwanted_words = unwanted_words.union(stop_words)
    lemmatizer = WordNetLemmatizer()

    def clean_ai_output(text, max_words):
        """Clean and process AI output with error handling while maintaining original functionality."""
        try:
            # Original cleaning steps
            text = re.sub(r"\.\w{1,4}$", "", text)  # Remove file extensions
            text = re.sub(r"[^\w\s]", " ", text)  # Remove special characters
            text = re.sub(r"\d+", "", text)  # Remove digits
            text = text.strip()
            text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)  # Split camelCase

            # Tokenization with fallback
            try:
                words = word_tokenize(text)
            except (LookupError, ImportError):
                words = text.split()

            # Basic word filtering (maintained from original)
            words = [word.lower() for word in words if word.isalpha()]

            # Lemmatization with fallback
            try:
                words = [lemmatizer.lemmatize(word) for word in words]
            except (LookupError, ImportError):
                pass  # Keep original words if lemmatization fails

            # Remove unwanted words and duplicates (maintained from original)
            filtered_words = []
            seen = set()
            for word in words:
                if word not in all_unwanted_words and word not in seen:
                    filtered_words.append(word)
                    seen.add(word)

            filtered_words = filtered_words[:max_words]
            return "_".join(filtered_words)

        except Exception as e:
            # Final fallback if everything else fails
            print(f"Warning: Falling back to basic cleaning due to error: {e}")
            simple_words = text.lower().split()[:max_words]
            return "_".join(word for word in simple_words if word.isalnum())

    # Process filename
    filename = clean_ai_output(filename, max_words=MAX_FILENAME_LENGTH)
    if not filename or filename.lower() in ("untitled", ""):
        filename = clean_ai_output(description, max_words=MAX_FILENAME_LENGTH)
    if not filename:
        filename = "image_" + os.path.splitext(os.path.basename(image_path))[0]

    sanitized_filename = sanitize_filename(filename, max_words=MAX_FILENAME_LENGTH)

    # Process foldername
    foldername = clean_ai_output(foldername, max_words=MAX_FOLDERNAME_LENGTH)
    if not foldername or foldername.lower() in ("untitled", ""):
        foldername = clean_ai_output(description, max_words=MAX_FOLDERNAME_LENGTH)
        if not foldername:
            foldername = "images"

    sanitized_foldername = sanitize_filename(
        foldername, max_words=MAX_FOLDERNAME_LENGTH
    )

    return sanitized_foldername, sanitized_filename, description
