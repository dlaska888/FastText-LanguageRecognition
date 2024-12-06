from flask import Flask, request, jsonify, render_template
import fasttext

app = Flask(__name__)

model_txt = fasttext.train_supervised('model.txt',
                                      epoch=100,
                                      lr=0.1,
                                      dim=720,
                                      ws=5,
                                      loss='softmax')
model_lid = fasttext.load_model('lid.176.bin')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/identify_language', methods=['POST'])
def identify_language():
    data = request.get_json()
    text = data.get('text', '')
    use_lid_model = data.get('useLidModel', False)
    
    if not text:
        return jsonify({"error": "Text empty"}), 400
    
    model = model_lid if use_lid_model else model_txt
    
    try:
        predictions = model.predict(text)
        lang_code = predictions[0][0].replace("__label__", "")
        return jsonify({"language": lang_code})
    except ValueError as e:
        return jsonify({"error": f"Error: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)