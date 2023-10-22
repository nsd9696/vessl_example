from pyngrok import ngrok

public_url = ngrok.connect(port = '8501')
print(f"Please click on the text below {public_url}")