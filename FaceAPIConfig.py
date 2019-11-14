# subscription_key = "fa553bd3b27c497a8b1a9ab3ed8cb32b" # L

subscription_key = 'ccfcd1aa55d94dac93b32f46415322f3' # H
#face_api_url = "https://canadacentral.api.cognitive.microsoft.com/face/v1.0/detect"
face_api_url = 'https://canadacentral.cognitiveservices.azure.com/face/v1.0/detect'



def config():
    print("call config")
    return subscription_key, face_api_url