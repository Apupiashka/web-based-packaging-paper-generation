from flask import Flask, render_template, request, jsonify, send_file
import traceback
import requests
from PIL import Image, ImageDraw, ImageFilter
import xml.etree.ElementTree as ET
import urllib.request
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
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

MOI_OTCHETY_BASE_URL = "https://hygieia.fast-report.com/app/workspace/674982171d6ee7f62ddcd207/tasks"
MOI_OTCHETY_API_TOKEN = "z7mwiz4946jskmyxweunt31j51r61tjxu7tdpt7bmboq515iqzny"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'your@gmail.com'  # Ваша почта
app.config['MAIL_PASSWORD'] = 'your_password'  # Ваш пароль
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

# Загружаем единственный шаблон с МоиОтчёты Облако, чтобы всё время менять только его
template = "https://hygieia.fast-report.com/download/t/674b0bf81d6ee7f62ddcd7df"

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


@app.route('/send-pdf-email', methods=['POST'])
def send_pdf_email():
    try:
        email = request.form['email']
        subject = request.form['subject']
        body = request.form['body']

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "report.pdf")
        if not os.path.exists(file_path):
            return jsonify({"error": "Файл PDF для отправки не найден."}), 404

        result = send_report_to_cloud(email, subject, body, file_path)
        return jsonify({"success": True}), 200
    except Exception as e:
        print("Error occurred:", e)
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500

def send_report_to_cloud(email, subject, body, file_path):
    """
    Отправка отчёта через сервис "Мои отчёты Облако".
    """
    try:
        # Загрузка файла на сервер "Мои отчёты Облако"
        upload_url = f"{MOI_OTCHETY_BASE_URL}"
        with open(file_path, 'rb') as f:
            response = requests.post(upload_url, headers={
                'Authorization': f'Bearer {MOI_OTCHETY_API_TOKEN}'
            }, files={'file': f})
        response.raise_for_status()
        uploaded_file_id = response.json().get('file_id')

        # Отправка письма через "Мои отчёты Облако"
        send_url = f"{MOI_OTCHETY_BASE_URL}/send"
        payload = {
            "email": email,
            "subject": subject,
            "body": body,
            "file_id": uploaded_file_id
        }
        response = requests.post(send_url, headers={
            'Authorization': f'Bearer {MOI_OTCHETY_API_TOKEN}',
            'Content-Type': 'application/json'
        }, json=payload)
        response.raise_for_status()

        return {"status": "success", "message": "Отчёт успешно отправлен."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.route('/send-report', methods=['POST'])
def send_report():
    try:
        email = request.form['email']
        subject = request.form['subject']
        body = request.form['body']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "report.pdf")

        result = send_report_to_cloud(email, subject, body, file_path)
        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

def convert_image_to_pdf(image_bytes, output_filename):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)

        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter

        image.thumbnail((width, height), Image.ANTIALIAS)
        image.save("temp_image.png", "PNG")

        c.drawImage("temp_image.png", 0, 0, width, height)
        c.save()

        os.remove("temp_image.png")

        return pdf_path
    except Exception as e:
        return str(e)

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
                         selected_images=None, smooth_transition=False, sharpness=1,
                         stroke_size=0, stroke_color=(0, 0, 0)):
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
                image_to_paste = resized_images[img_index]
                if stroke_size > 0:
                    draw = ImageDraw.Draw(image_to_paste)
                    draw.rectangle([0, 0, max_block_width, max_block_height], outline=stroke_color, width=stroke_size)
                new_image.paste(image_to_paste, (x_offset, y_offset), image_to_paste)

    return new_image


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download-external', methods=['POST'])
def download_external():
    try:
        global template
        external_url = request.form.get('externalUrl', template)
        if not external_url:
            return jsonify({'error': 'No external URL provided'}), 400

        response = urllib.request.urlopen(external_url)
        file_data = response.read()

        filename = secure_filename(os.path.basename(external_url))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(file_data)

        image = Image.open(filepath).convert("RGBA")

        os.remove(filepath)

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({'image': f'data:image/png;base64,{encoded_image}', 'filename': filename})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


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

        width = int(request.form.get('width', 1240))
        height = int(request.form.get('height', 1754))
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
        stroke_size = int(request.form.get('strokeSize', 0))
        stroke_color_input = request.form.get('strokeColor', '#000000')

        if stroke_color_input.startswith('#'):
            stroke_color = hex_to_rgb(stroke_color_input)
        else:
            stroke_color = color_name_to_rgb(stroke_color_input)

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
            gradient_direction, shuffle, shuffle_pattern, selected_indices, smooth_transition, sharpness,
            stroke_size, stroke_color
        )

        output = io.BytesIO()
        result_image.save(output, format='PNG')
        output.seek(0)

        for path in image_paths:
            os.remove(path)

        encoded_image = base64.b64encode(output.getvalue()).decode('utf-8')
        return jsonify({'image': f'data:image/png;base64,{encoded_image}'})

        output = io.BytesIO()
        result_image.save(output, format='PNG')
        image_data = output.getvalue()
        output.seek(0)
        encoded_image = base64.b64encode(image_data).decode('utf-8')

        send_email_request = request.form.get('sendEmail') == 'true'
        if send_email_request:
            recipient_email = request.form.get('email')
            if not recipient_email:
                return jsonify(
                    {'error': 'Email получателя не указан', 'image': f'data:image/png;base64,{encoded_image}'}), 400

            email_subject = request.form.get('emailSubject', 'Сгенерированное изображение')
            email_body = request.form.get('emailBody', 'Ваше изображение во вложении.')
            if send_email(recipient_email, email_subject, email_body, image_data):
                return jsonify({'message': 'Изображение успешно сгенерировано и отправлено на почту',
                                'image': f'data:image/png;base64,{encoded_image}'})
            else:
                return jsonify(
                    {'error': 'Ошибка при отправке email', 'image': f'data:image/png;base64,{encoded_image}'}), 500
        else:
            return jsonify({'image': f'data:image/png;base64,{encoded_image}'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
