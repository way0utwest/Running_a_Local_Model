# Running a Local Model
This is a repo that contains some demo code from my session: Running a Local GenAI Model on your Laptop

There are a few demos in this session, described below.

Note: If you use the code from sources for some of these demos, you likely will have container name conflicts. You can remove containers to get things working. I've tried to change names to ensure things run smoother here.

## Running a Local Chatbot
The folder, localchatbot, contains the code to run a local model using Docker.

Ollama_Start.cmd - calls docker compose to start two containers. One is for Ollama. The other is the OpenWebUI. This starts on port 3000. If you need to change that, then line 26 is where you alter the first number. Docker port mapping is host:container.

## Training a Local Model with Python
The intro.md file contains the code to run this demo. This can be slow, so be aware of that. Python required.

## Local RAG Setup with SQL Server 2025
Use Anthony Notencino's setup for Docker here: [ollama-sql-faststart](https://github.com/nocentino/ollama-sql-faststart). This starts a number of containers and manages the security setup.

This can be slow, so do not look to this for production work, but it is a way to quickly get moving with a setup.

