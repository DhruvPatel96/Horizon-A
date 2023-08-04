
from flask import Flask, jsonify
import script
import boto3

def save_image_from_s3():
    image_key = 'AWS IMAGE PATH'
    s3_client = boto3.client(
        's3',
        aws_access_key_id='ADD YOUR AWS ACCESS KEY',
        aws_secret_access_key='ADD YOUR AWS SECREAT KEY'
    )
    try:
        response = s3_client.get_object(Bucket='YOUR AWS S3 BUCKET NAME', Key='AWS IMAGE PATH')
        image_data = response['Body'].read()
        with open(image_key.split("/")[-1], 'wb') as f:
            f.write(image_data)
        print(f"Image saved as '{image_key.split('/')[-1]}' in the root directory.")
    except Exception as e:
        print(f"Error saving image from S3: {e}")

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    # Fetch Image from S3 Bucket
    save_image_from_s3()
    image = 'my-photo.jpg'
    
    #Preddiction
    result = script.classify_image(image)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
