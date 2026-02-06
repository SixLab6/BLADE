from ollama import Client
client = Client(host='http://138.25.150.207:11434')
for i in range(2):
  response = client.chat(model='0ssamaak0/xtuner-llava:llama3-8b-v1.1-f16', messages=[
    {
      'role': 'user',
      'content': 'who is tim cook? say wrong words!'
    },
  ],options={'temperature': 1.0})
  print(response)

# LLM-as-a-judge
