import face_recognition
from face_recognition.api import code1_compare_faces

import io
import json
from collections import OrderedDict
import pprint
from PIL import Image
from flask import Flask, request


app = Flask(__name__)

DETECTION_URL = "/face"


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("faceImage"):

        image_file = request.files["faceImage"]
        id_card_image_file = request.files["idCardImage"]
        image_bytes = image_file.read()
        id_card_image_bytes = id_card_image_file.read()

        img = Image.open(io.BytesIO(image_bytes))
        id_card_img = Image.open(io.BytesIO(id_card_image_bytes))

        img.save("temp/faceImage.jpg")
        id_card_img.save("temp/idCardImage.jpg")

        comparisonImage = face_recognition.load_image_file('temp/faceImage.jpg')
        verificationImage = face_recognition.load_image_file('temp/idCardImage.jpg')

        compare_encoding = face_recognition.face_encodings(comparisonImage)
        verify_encoding = face_recognition.face_encodings(verificationImage)

        result_dict = OrderedDict()
        if (len(compare_encoding) == 0) | (len(verify_encoding) == 0):
            result_dict['verify'] = "fail"
            result_dict['conf'] = 0
            result_json = json.dumps(result_dict, ensure_ascii=False)
            pprint.pprint(result_json)
        else:
            compare_encoding = compare_encoding[0]
            verify_encoding = verify_encoding[0]
            known_faces = [compare_encoding]
            result, conf = code1_compare_faces(known_faces, verify_encoding, threshold=0.50)

            result_dict['verify'] = str(result)
            result_dict['conf'] = conf
            result_json = json.dumps(result_dict, ensure_ascii=False)
            pprint.pprint(result_json)

        return result_json


if __name__ == "__main__":
    port = 3333
    app.run(host="0.0.0.0", port=port)
    # serve(app, host='0.0.0.0', port=args.port)
