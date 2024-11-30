from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image, ImageDraw, ImageFilter
import random
import os
import numpy as np
import base64
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    return "#{:02x}{:02x}{:02x}".format(*rgb_color)

def color_name_to_rgb(color_name):
    color_map = {
        "красный": (255, 0, 0),
        "зеленый": (0, 255, 0),
        "синий": (0, 0, 255),
        "желтый": (255, 255, 0),
        "черный": (0, 0, 0),
        "белый": (255, 255, 255),
        "оранжевый": (255, 165, 0),
        "фиолетовый": (128, 0, 128),
        "розовый": (255, 192, 203),
        "коричневый": (165, 42, 42),
        "серый": (128, 128, 128),
        "голубой": (0, 191, 255)
    }
    return color_map.get(color_name.lower(), (0, 0, 0))

def shuffle_images(images, pattern, columns, rows):
    num_images = len(images)

    if pattern == 'rows':
        shuffled_images = []
        for row in range(rows):
            row_images = images[row * columns:(row + 1) * columns]
            random.shuffle(row_images)
            shuffled_images.extend(row_images)
        return shuffled_images

    elif pattern == 'columns':
        shuffled_images = [None] * (columns * rows)
        for col in range(columns):
            col_images = [images[row * columns + col] for row in range(rows) if row * columns + col < num_images]
            random.shuffle(col_images)
            for row in range(rows):
                if row * columns + col < num_images:
                    shuffled_images[row * columns + col] = col_images.pop(0)
        return shuffled_images

    elif pattern == 'diagonals':
        diagonals = [[] for _ in range(columns + rows - 1)]
        for row in range(rows):
            for col in range(columns):
                idx = row * columns + col
                if idx < num_images:
                    diagonals[row + col].append(images[idx])

        shuffled_images = []
        for diag in diagonals:
            random.shuffle(diag)
            shuffled_images.extend(diag)
        return shuffled_images

    elif pattern == 'checkerboard':
        checkerboard = []
        for row in range(rows):
            for col in range(columns):
                idx = row * columns + col
                if idx < num_images:
                    checkerboard.append((idx, (row + col) % 2))
        random.shuffle(checkerboard)
        checkerboard.sort(key=lambda x: x[1])
        return [images[idx] for idx, _ in checkerboard]

    else:
        random.shuffle(images)
        return images

def create_gradient_background(width, height, colors, direction='horizontal', smooth_transition=False):
    image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image)

    if direction == 'horizontal':
        for x in range(width):
            segment = (len(colors) - 1) * x / width
            i = int(segment)
            fraction = segment - i

            if i < len(colors) - 1:
                r = int(colors[i][0] * (1 - fraction) + colors[i + 1][0] * fraction)
                g = int(colors[i][1] * (1 - fraction) + colors[i + 1][1] * fraction)
                b = int(colors[i][2] * (1 - fraction) + colors[i + 1][2] * fraction)
                draw.line([(x, 0), (x, height)], fill=(r, g, b))

    elif direction == 'vertical':
        for y in range(height):
            segment = (len(colors) - 1) * y / height
            i = int(segment)
            fraction = segment - i

            if i < len(colors) - 1:
                r = int(colors[i][0] * (1 - fraction) + colors[i + 1][0] * fraction)
                g = int(colors[i][1] * (1 - fraction) + colors[i + 1][1] * fraction)
                b = int(colors[i][2] * (1 - fraction) + colors[i + 1][2] * fraction)
                draw.line([(0, y), (width, y)], fill=(r, g, b))

    elif direction == 'diagonal':
        for x in range(width + height):
            segment = (len(colors) - 1) * x / (width + height)
            i = int(segment)
            fraction = segment - i

            if i < len(colors) - 1:
                r = int(colors[i][0] * (1 - fraction) + colors[i + 1][0] * fraction)
                g = int(colors[i][1] * (1 - fraction) + colors[i + 1][1] * fraction)
                b = int(colors[i][2] * (1 - fraction) + colors[i + 1][2] * fraction)
                draw.line([(max(0, x - height), min(x, height - 1)),
                          (min(x, width - 1), max(0, x - width))],
                         fill=(r, g, b))

    if smooth_transition:
        image = image.filter(ImageFilter.GaussianBlur(radius=width // 20))

    return image

def create_transparent_edges(image, sharpness):
    width, height = image.size
    alpha = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(alpha)

    for x in range(width):
        for y in range(height):
            dist_center = np.sqrt((x - width / 2) ** 2 + (y - height / 2) ** 2)
            max_dist = np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
            normalized_dist = dist_center / max_dist
            alpha_val = int(255 * (1 - normalized_dist ** (1 / sharpness)))
            alpha_val = max(0, min(255, alpha_val))
            draw.point((x, y), fill=alpha_val)

    image = image.convert("RGBA")
    image.putalpha(alpha)
    return image

def create_wrapping_paper(images, width, height, columns=1, spacing=0, gradient_colors=None,
                         gradient_direction='horizontal', shuffle=False, shuffle_pattern='random',
                         selected_images=None, smooth_transition=False, sharpness=1):
    if not images:
        raise ValueError("The image list cannot be empty.")

    if selected_images:
        images = [images[i] for i in selected_images]
    elif shuffle:
        rows = (height + spacing) // ((width - (columns - 1) * spacing) // columns + spacing)
        images = shuffle_images(images, shuffle_pattern, columns, rows)

    num_images = len(images)

    if gradient_colors and len(gradient_colors) > 1:
        new_image = create_gradient_background(width, height, gradient_colors, gradient_direction, smooth_transition)
    else:
        new_image = Image.new('RGB', (width, height), gradient_colors[0] if gradient_colors else (255, 255, 255))

    max_block_width = (width - (columns - 1) * spacing) // columns
    max_block_height = max_block_width

    rows = (height + spacing) // (max_block_height + spacing)

    resized_images = []
    for img in images:
        resized_img = img.resize((max_block_width, max_block_height))
        transparent_img = create_transparent_edges(resized_img, sharpness)
        resized_images.append(transparent_img)

    for row in range(rows):
        for col in range(columns):
            img_index = (row * columns + col) % num_images
            x_offset = col * (max_block_width + spacing) + (
                    width - columns * (max_block_width + spacing) + spacing) // 2
            y_offset = row * (max_block_height + spacing) + (
                    height - rows * (max_block_height + spacing) + spacing) // 2
            if y_offset + max_block_height <= height and x_offset + max_block_width <= width:
                new_image.paste(resized_images[img_index], (x_offset, y_offset), resized_images[img_index])

    return new_image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/download/<format>', methods=['POST'])
def download(format):
    try:
        image_data = request.json.get('image').split(',')[1]

        image_bytes = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(image_bytes))

        output = io.BytesIO()

        if format.lower() == 'jpg':
            image = image.convert('RGB')
            image.save(output, format='JPEG', quality=95)
        elif format.lower() == 'png':
            image.save(output, format='PNG')
        elif format.lower() == 'webp':
            image.save(output, format='WEBP', quality=95)
        elif format.lower() == 'bmp':
            image.save(output, format='BMP')
        else:
            return jsonify({'error': 'Неподдерживаемый формат'}), 400

        output.seek(0)

        return send_file(
            output,
            mimetype=f'image/{format.lower()}',
            as_attachment=True,
            download_name=f'wrapping_paper.{format.lower()}'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/generate', methods=['POST'])
def generate():
    try:
        files = request.files.getlist('images')
        print(f"Received files: {files}")

        width = int(request.form.get('width', 1000))
        height = int(request.form.get('height', 1000))
        columns = int(request.form.get('columns', 3))
        spacing = int(request.form.get('spacing', 0))
        gradient_colors = request.form.get('gradientColors', 'белый, черный')
        gradient_direction = request.form.get('gradientDirection', 'horizontal')
        shuffle = request.form.get('shuffle') == 'true'
        shuffle_pattern = request.form.get('shufflePattern', 'random')
        custom_order = request.form.get('customOrder') == 'true'
        selected_images = request.form.get('selectedImages', '').split(',')
        smooth_transition = request.form.get('smoothTransition') == 'true'
        sharpness = float(request.form.get('sharpness', 1))

        image_paths = []
        for file in files:
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image_paths.append(filepath)
                print(f"Saved file: {filepath}")

        images = [Image.open(path).convert("RGBA") for path in image_paths]

        if not images:
            return jsonify({'error': 'The image list is empty after opening files'}), 400

        gradient_colors_rgb = []
        for color in gradient_colors.split(','):
            color = color.strip()
            if color.startswith('#'):
                gradient_colors_rgb.append(hex_to_rgb(color))
            else:
                gradient_colors_rgb.append(color_name_to_rgb(color))

        selected_indices = None
        if custom_order and selected_images:
            selected_indices = [int(i) for i in selected_images if i]

        result_image = create_wrapping_paper(
            images, width, height, columns, spacing, gradient_colors_rgb,
            gradient_direction, shuffle, shuffle_pattern, selected_indices, smooth_transition, sharpness
        )

        output = io.BytesIO()
        result_image.save(output, format='PNG')
        output.seek(0)

        for path in image_paths:
            os.remove(path)

        encoded_image = base64.b64encode(output.getvalue()).decode('utf-8')
        return jsonify({'image': f'data:image/png;base64,{encoded_image}'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)