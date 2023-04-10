docker run -i $(docker build -q . --build-arg openai_key=$OPENAI_API_KEY)
