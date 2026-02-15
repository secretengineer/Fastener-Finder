import ollama
import json
from PIL import Image, ImageDraw


def _extract_json_array(raw_content: str):
    json_start = raw_content.find('[')
    json_end = raw_content.rfind(']') + 1
    if json_start == -1 or json_end == 0:
        raise ValueError("No JSON array found in model response.")
    return json.loads(raw_content[json_start:json_end])


def _normalize_box(box_2d, width, height):
    if len(box_2d) != 4:
        raise ValueError(f"Invalid box_2d length: {box_2d}")

    ymin, xmin, ymax, xmax = box_2d

    # Accept pixel coords, 0-1 normalized, or 0-1000 normalized.
    max_val = max(ymin, xmin, ymax, xmax)
    if max_val <= 1.0:
        left = xmin * width
        top = ymin * height
        right = xmax * width
        bottom = ymax * height
    elif max_val <= 1000.0:
        left = xmin * width / 1000
        top = ymin * height / 1000
        right = xmax * width / 1000
        bottom = ymax * height / 1000
    else:
        left, top, right, bottom = xmin, ymin, xmax, ymax

    # Clamp to image bounds and ensure proper ordering.
    left, right = sorted((max(0, left), min(width, right)))
    top, bottom = sorted((max(0, top), min(height, bottom)))
    return left, top, right, bottom

def run_test():
    # 1. Ask Qwen3-VL to find the hardware
    print("Analyzing image with qwen3-vl...")
    response = ollama.chat(
        model='qwen3-vl',
        messages=[{
            'role': 'user',
            'content': 'Locate all hardware items. Return a JSON list of objects. Each object must have "label" and "box_2d" [ymin, xmin, ymax, xmax].',
            'images': ['hardware.jpg']
        }]
    )

    # 2. Extract the text response
    raw_content = response['message']['content']
    print(f"Model Response: {raw_content}")

    # 3. Open the image and prepare to draw
    img = Image.open('hardware.jpg')
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # 4. Parse the JSON (Models often wrap JSON in code blocks)
    try:
        data = _extract_json_array(raw_content)

        for i, entry in enumerate(data, 1):
            if 'box_2d' not in entry or 'label' not in entry:
                raise ValueError(f"Missing fields in entry: {entry}")

            left, top, right, bottom = _normalize_box(entry['box_2d'], width, height)

            # Draw a box and a label
            draw.rectangle([left, top, right, bottom], outline="red", width=3)
            draw.text((left + 5, top + 5), f"#{i}: {entry['label']}", fill="red")

        # 5. Save and show the result
        img.save('output_labeled.jpg')
        img.show()
        print("Done! Result saved as 'output_labeled.jpg'")

    except Exception as e:
        print(f"Error parsing model output: {e}")

if __name__ == "__main__":
    run_test()