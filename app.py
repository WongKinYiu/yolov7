from flask import Flask, render_template, request, send_file
import os
import subprocess

app = Flask(__name__)

uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/opencam", methods=['GET'])
def opencam():
    print("here")
    subprocess.run(['python', '/workspace/yolov7/detect.py', '--weights', '/workspace/asl-volov7-model/yolov7.pt', '--source', '0'])
    return "done"

@app.route('/return-files', methods=['GET'])
def return_file():
    obj = request.args.get('obj')
    loc = os.path.join("runs/detect", obj)
    print(loc)
    try:
        return send_file(os.path.join("runs/detect", obj), attachment_filename=obj)
        # return send_from_directory(loc, obj)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    # starting app
    app.run(debug=True,host='0.0.0.0')
