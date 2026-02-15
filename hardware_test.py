import ollama
import json
import os

from PIL import Image, ImageDraw, ImageFont

# Visual style constants
BOX_COLOR = (230, 40, 40)
BOX_WIDTH = 4
ID_BADGE_FILL = (230, 40, 40)
ID_BADGE_OUTLINE = (255, 255, 255)
ID_TEXT_COLOR = (255, 255, 255)
PANEL_BG = (18, 18, 18)
TABLE_HEADER_BG = (60, 60, 60)
TABLE_ROW_A = (24, 24, 24)
TABLE_ROW_B = (36, 36, 36)
TABLE_GRID = (90, 90, 90)
TABLE_BORDER = (130, 130, 130)
TABLE_TEXT = (255, 255, 255)
TABLE_SUBTEXT = (200, 200, 200)


def _extract_json_array(raw_content: str):
    json_start = raw_content.find('[')
    json_end = raw_content.rfind(']') + 1
    if json_start == -1 or json_end == 0:
        raise ValueError("No JSON array found in model response.")
    return json.loads(raw_content[json_start:json_end])


def _to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _load_font(size):
    candidates = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _scale_box_xyxy(x1, y1, x2, y2, width, height):
    max_val = max(x1, y1, x2, y2)
    if max_val <= 1.0:
        left = x1 * width
        top = y1 * height
        right = x2 * width
        bottom = y2 * height
    elif max_val <= 1000.0:
        left = x1 * width / 1000
        top = y1 * height / 1000
        right = x2 * width / 1000
        bottom = y2 * height / 1000
    else:
        left, top, right, bottom = x1, y1, x2, y2

    # Clamp to image bounds and ensure proper ordering.
    left, right = sorted((max(0, left), min(width, right)))
    top, bottom = sorted((max(0, top), min(height, bottom)))
    return left, top, right, bottom


def _normalize_box(entry, width, height):
    """
    Preferred format:
      "box_2d": {"x1":..., "y1":..., "x2":..., "y2":...}
    Backward compatible list format:
      "box_2d": [a, b, c, d], interpreted as xyxy by default.
      If "box_order" is present and equals "yxyx", map as [y1, x1, y2, x2].
    """
    box_2d = entry.get("box_2d")
    if box_2d is None:
        raise ValueError(f"Missing box_2d in entry: {entry}")

    if isinstance(box_2d, dict):
        needed = ("x1", "y1", "x2", "y2")
        if not all(k in box_2d for k in needed):
            raise ValueError(f"Invalid dict box_2d keys: {box_2d}")
        x1 = _to_float(box_2d["x1"])
        y1 = _to_float(box_2d["y1"])
        x2 = _to_float(box_2d["x2"])
        y2 = _to_float(box_2d["y2"])
        return _scale_box_xyxy(x1, y1, x2, y2, width, height)

    if isinstance(box_2d, list) and len(box_2d) == 4:
        a, b, c, d = [_to_float(v) for v in box_2d]
        order = str(entry.get("box_order", "xyxy")).lower()
        if order in ("yxyx", "ymin_xmin_ymax_xmax"):
            return _scale_box_xyxy(b, a, d, c, width, height)
        return _scale_box_xyxy(a, b, c, d, width, height)

    raise ValueError(f"Invalid box_2d format: {box_2d}")


def _format_bbox(box):
    left, top, right, bottom = box
    return f"({int(left)},{int(top)})-({int(right)},{int(bottom)})"


def _fit_text(draw, text, font, max_width):
    s = str(text)
    if draw.textlength(s, font=font) <= max_width:
        return s
    ellipsis = "..."
    if draw.textlength(ellipsis, font=font) > max_width:
        return ""
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = s[:mid] + ellipsis
        if draw.textlength(candidate, font=font) <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return s[:lo] + ellipsis


def _wrap_text_by_pixels(draw, text, font, max_width, max_lines):
    words = str(text).split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for w in words[1:]:
        test = f"{current} {w}"
        if draw.textlength(test, font=font) <= max_width:
            current = test
        else:
            lines.append(current)
            current = w
    lines.append(current)
    if len(lines) <= max_lines:
        return lines
    clipped = lines[:max_lines]
    clipped[-1] = _fit_text(draw, clipped[-1], font, max_width)
    return clipped


def _draw_table(draw, x_start, y_start, rows, font, header_font):
    headers = ["ID", "Label", "Specification", "Conf", "Box (px)"]
    col_widths = [70, 190, 430, 95, 340]
    row_height = 64
    table_width = sum(col_widths)
    header_h = 56
    pad = 10

    # Header band
    draw.rectangle([x_start, y_start, x_start + table_width, y_start + header_h], fill=TABLE_HEADER_BG)
    x = x_start
    for i, header in enumerate(headers):
        draw.text((x + pad, y_start + 14), header, fill=TABLE_TEXT, font=header_font)
        x += col_widths[i]
        draw.line([(x, y_start), (x, y_start + header_h + row_height * len(rows))], fill=TABLE_GRID, width=1)

    draw.rectangle([x_start, y_start, x_start + table_width, y_start + header_h + row_height * len(rows)], outline=TABLE_BORDER, width=1)
    draw.line([(x_start, y_start + header_h), (x_start + table_width, y_start + header_h)], fill=TABLE_BORDER, width=1)

    # Rows
    y = y_start + header_h
    for row in rows:
        stripe = TABLE_ROW_B if (row["id"] % 2 == 0) else TABLE_ROW_A
        draw.rectangle([x_start, y, x_start + table_width, y + row_height], fill=stripe)
        draw.line([(x_start, y + row_height), (x_start + table_width, y + row_height)], fill=(55, 55, 55), width=1)

        x = x_start
        id_text = _fit_text(draw, row["id"], font, col_widths[0] - 2 * pad)
        draw.text((x + pad, y + 18), id_text, fill=TABLE_TEXT, font=font)
        x += col_widths[0]

        label_text = _fit_text(draw, row["label"], font, col_widths[1] - 2 * pad)
        draw.text((x + pad, y + 18), label_text, fill=TABLE_TEXT, font=font)
        x += col_widths[1]

        spec_lines = _wrap_text_by_pixels(draw, row["specification"], font, col_widths[2] - 2 * pad, max_lines=2)
        draw.multiline_text((x + pad, y + 10), "\n".join(spec_lines), fill=TABLE_TEXT, font=font, spacing=4)
        x += col_widths[2]

        conf_text = _fit_text(draw, row["confidence"], font, col_widths[3] - 2 * pad)
        draw.text((x + pad, y + 18), conf_text, fill=TABLE_TEXT, font=font)
        x += col_widths[3]

        bbox_text = _fit_text(draw, row["bbox"], font, col_widths[4] - 2 * pad)
        draw.text((x + pad, y + 18), bbox_text, fill=TABLE_TEXT, font=font)
        y += row_height


def _draw_numbered_detections(img, detections, font, id_font):
    draw = ImageDraw.Draw(img)
    for i, det in enumerate(detections, 1):
        left, top, right, bottom = det["box"]
        draw.rectangle([left, top, right, bottom], outline=BOX_COLOR, width=BOX_WIDTH)
        cx, cy = int(left + 24), int(max(24, top + 24))
        txt = str(i)
        bbox = draw.textbbox((0, 0), txt, font=id_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tag_r = max(24, int(max(tw, th) / 2) + 10)
        draw.ellipse(
            [cx - tag_r, cy - tag_r, cx + tag_r, cy + tag_r],
            fill=ID_BADGE_FILL,
            outline=ID_BADGE_OUTLINE,
            width=2,
        )
        draw.text((cx - tw / 2, cy - th / 2 - 1), txt, fill=ID_TEXT_COLOR, font=id_font)


def _heuristic_specification(label, box, width, height):
    left, top, right, bottom = box
    w = max(1.0, right - left)
    h = max(1.0, bottom - top)
    long_side = max(w, h)
    short_side = min(w, h)
    rel_long = long_side / max(1.0, min(width, height))
    aspect = long_side / short_side
    label_l = str(label).lower()

    size = "small" if rel_long < 0.05 else "medium" if rel_long < 0.11 else "large"
    if "washer" in label_l:
        return f"{size} washer; outer diameter approx {int(long_side)} px"
    if "nut" in label_l:
        return f"{size} nut; across flats approx {int(long_side)} px"
    if "bolt" in label_l:
        return f"{size} bolt; shank length approx {int(long_side)} px; head style unknown"
    if "screw" in label_l:
        return f"{size} screw; length approx {int(long_side)} px; drive/head unknown"
    if "spring" in label_l:
        return f"coil spring; free length approx {int(long_side)} px; diameter approx {int(short_side)} px"
    return f"{size} fastener; footprint {int(w)}x{int(h)} px; aspect {aspect:.2f}"


def _enrich_detections_with_model(numbered_image_path, detections):
    payload = []
    for i, det in enumerate(detections, 1):
        payload.append({
            "id": i,
            "detected_label": det["label"],
            "box_px": [int(det["box"][0]), int(det["box"][1]), int(det["box"][2]), int(det["box"][3])],
        })

    response = ollama.chat(
        model='qwen3-vl',
        messages=[{
            'role': 'user',
            'content': (
                'You are validating numbered fastener detections in the attached image. '
                'Each detection already has an ID and rough label. '
                'For each ID, output best canonical hardware label and a concise practical specification. '
                'Use only what is visible. Do not invent exact standards or thread pitch unless clearly legible. '
                'If uncertain, keep generic terms and use "unknown" where needed. '
                'Return ONLY a JSON array with objects containing: '
                '"id" (int), "label" (string), "specification" (string), "confidence" (0-1). '
                f'Input detections: {json.dumps(payload)}'
            ),
            'images': [numbered_image_path]
        }]
    )

    raw = response['message']['content']
    enriched = _extract_json_array(raw)
    by_id = {}
    for row in enriched:
        if "id" not in row:
            continue
        idx = int(row["id"])
        by_id[idx] = {
            "label": str(row.get("label", "")).strip(),
            "specification": str(row.get("specification", "")).strip(),
            "confidence": min(1.0, max(0.0, _to_float(row.get("confidence", 0.0), 0.0))),
        }
    return by_id


def run_test():
    # 1. Ask Qwen3-VL to find the hardware
    print("Analyzing image with qwen3-vl...")
    response = ollama.chat(
        model='qwen3-vl',
        messages=[{
            'role': 'user',
            'content': (
                'Locate all hardware items. Return ONLY a JSON list of objects. '
                'Each object must have: '
                '"label" (hardware type), '
                '"specification" (short hardware spec or "unknown"), '
                '"box_2d" object with keys x1,y1,x2,y2 in XYXY order, '
                'normalized to 0-1000 relative to the FULL original image. '
                'Also include "box_order":"xyxy".'
            ),
            'images': ['hardware.jpg']
        }]
    )

    # 2. Extract the text response
    raw_content = response['message']['content']
    print(f"Model Response: {raw_content}")

    # 3. Open the image
    img = Image.open('hardware.jpg')
    width, height = img.size

    # 4. Parse the JSON (Models often wrap JSON in code blocks)
    try:
        data = _extract_json_array(raw_content)
        font = _load_font(28)
        header_font = _load_font(30)
        id_font = _load_font(34)
        detections = []

        for entry in data:
            if 'box_2d' not in entry or 'label' not in entry:
                raise ValueError(f"Missing fields in entry: {entry}")

            left, top, right, bottom = _normalize_box(entry, width, height)
            detections.append({
                "label": str(entry["label"]),
                "specification": str(entry.get("specification", "unknown")),
                "box": (left, top, right, bottom),
            })

        # Stable numbering by image position (top-to-bottom, left-to-right).
        detections.sort(key=lambda d: (d["box"][1], d["box"][0]))

        # Draw numbered detections first and run a second pass classifier.
        numbered_img = img.copy()
        _draw_numbered_detections(numbered_img, detections, font, id_font)
        numbered_img_path = 'output_numbered_tmp.jpg'
        numbered_img.save(numbered_img_path)

        enriched_by_id = {}
        try:
            enriched_by_id = _enrich_detections_with_model(numbered_img_path, detections)
        except Exception as enrich_error:
            print(f"Post-processing classifier warning: {enrich_error}")

        table_rows = []
        for i, det in enumerate(detections, 1):
            enriched = enriched_by_id.get(i, {})
            label = enriched.get("label", "") or det["label"]
            model_spec = enriched.get("specification", "")
            heuristic_spec = _heuristic_specification(label, det["box"], width, height)
            spec = model_spec if model_spec and model_spec.lower() != "unknown" else heuristic_spec
            conf = enriched.get("confidence", 0.0)
            if not enriched:
                conf = 0.25

            table_rows.append({
                "id": i,
                "label": label,
                "specification": spec,
                "confidence": f"{conf:.2f}",
                "bbox": _format_bbox(det["box"]),
            })

        # Build a side panel with a formatted table.
        panel_width = 1150
        output = Image.new("RGB", (width + panel_width, height), color=PANEL_BG)
        output.paste(numbered_img, (0, 0))
        panel = ImageDraw.Draw(output)
        panel.text((width + 20, 20), "Fastener Detections", fill=TABLE_TEXT, font=header_font)
        panel.text((width + 20, 60), f"Total: {len(table_rows)}", fill=TABLE_SUBTEXT, font=font)
        _draw_table(panel, width + 20, 94, table_rows, font, header_font)

        # 5. Save and show the result
        output.save('output_labeled.jpg')
        output.show()
        if os.path.exists(numbered_img_path):
            os.remove(numbered_img_path)
        print("Done! Result saved as 'output_labeled.jpg'")

    except Exception as e:
        print(f"Error parsing model output: {e}")

if __name__ == "__main__":
    run_test()
