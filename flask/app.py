import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
import cv2
import uuid
import tempfile
import traceback
from contextlib import contextmanager
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 允许跨域请求
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {
    'image': {'png', 'jpg', 'jpeg'},
    'video': {'mp4', 'mov', 'avi', 'webm'}
}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# 确保文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

model = YOLO('best.pt')  # 请替换为你的模型路径

@contextmanager
def video_capture_context(path):
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        yield cap
    finally:
        if cap is not None:
            cap.release()

@contextmanager
def video_writer_context(path, fourcc, fps, frame_size):
    out = None
    try:
        out = cv2.VideoWriter(path, fourcc, fps, frame_size)
        yield out
    finally:
        if out is not None:
            out.release()

def allowed_file(filename, mode):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS'][mode]

def process_video(input_path, output_path):
    cap = None
    out = None
    try:
        # 使用更兼容的视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 更兼容H.264编码

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 确保分辨率是偶数（某些编码器要求）
        width = width if width % 2 == 0 else width - 1
        height = height if height % 2 == 0 else height - 1

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            im_array = results[0].plot()
            out.write(im_array)

        return {
            'resolution': f"{width}x{height}",
            'fps': fps
        }
    except Exception as e:
        app.logger.error(f"视频处理错误: {str(e)}")
        return False
    finally:
        if out is not None:
            out.release()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        mode = request.form.get('mode', 'image')  # 默认为图片模式

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not file or not allowed_file(file.filename, mode):
            return jsonify({'error': f'Invalid file type for {mode} mode'}), 400

        # 保存上传文件
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{str(uuid.uuid4())}.{ext}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)

        result_data = {}
        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

        try:
            if mode == 'image':
                # 处理图片
                results = model(upload_path)

                # 使用OpenCV保存带标注的图像
                im_array = results[0].plot()
                cv2.imwrite(result_path, im_array)

                # 收集检测到的对象
                detected_objects = set()
                for box in results[0].boxes:
                    detected_objects.add(results[0].names[int(box.cls)])

                result_data = {
                    'type': 'image',
                    'original': f'/static/uploads/{filename}',
                    'result': f'/static/results/{result_filename}',
                    'stats': {
                        'inference_time': results[0].speed['inference'],
                        'resolution': f"{results[0].orig_shape[1]}x{results[0].orig_shape[0]}",
                        'detected_objects': list(detected_objects)
                    }
                }
            else:
                # 处理视频
                video_stats = process_video(upload_path, result_path)

                result_data = {
                    'type': 'video',
                    'original': f'/static/uploads/{filename}',
                    'result': f'/static/results/{result_filename}',
                    'stats': {
                        'resolution': video_stats['resolution'],
                        'fps': video_stats['fps'],
                        'detected_objects': []  # 视频检测暂不支持对象统计
                    }
                }

            return jsonify(result_data)
        except Exception as e:
            app.logger.error(f"Error processing file: {str(e)}")
            app.logger.error(traceback.format_exc())
            # 清理可能生成的部分结果文件
            if os.path.exists(result_path):
                os.remove(result_path)
            return jsonify({'error': 'Error processing file', 'details': str(e)}), 500
        finally:
            # 确保释放所有资源
            cv2.destroyAllWindows()
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/static/<path:folder>/<path:filename>')
def serve_static(folder, filename):
    return send_from_directory(f'static/{folder}', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)