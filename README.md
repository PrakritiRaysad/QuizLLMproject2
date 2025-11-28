# QuizLLMproject2
Repo for my LLM analysis task - project 2 - IITM - Data Science and Applications

This project is a local quiz/assignment solver that runs entirely on your machine using FastAPI and a Groq LLM–powered universal solver.
I exposed my server using ngrok, so no deployment (Railway/Render) is required.

Installation

1. Clone the repo
git clone https://github.com/<your-username>/QuizLLMproject2.git
cd QuizLLMproject2
2. Create a virtual environment (Windows)
python -m venv venv
venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt


Environment Variables

This project requires:
STUDENT_SECRET — the key required for /task execution
GROQ_API_KEY — your Groq LLM key
You can set them manually in CMD:

set STUDENT_SECRET=your_secret
set GROQ_API_KEY=your_groq_key

Or just run your included setup.bat.

No .env file is required since this project runs locally.

Running the Server

Option A: Run the batch file
setup.bat

Option B: Run manually
uvicorn main:app --host 0.0.0.0 --port 8000


Server will be available at:

http://127.0.0.1:8000

Exposing via Ngrok

In a separate terminal:
ngrok http 8000


You'll get a public HTTPS URL like:
https://abc123.ngrok-free.app

Use this URL to access your /task endpoint from anywhere.


Submitting a Quiz Task
Use the following cURL command (Windows PowerShell/CMD):

curl -X POST "<NGROK_URL>/task" ^
  -H "Content-Type: application/json" ^
  -d "{\"email\":\"YOUR_EMAIL\",\"secret\":\"STUDENT_SECRET\",\"url\":\"QUIZ_URL\"}"


Example:

curl -X POST "https://abc123.ngrok-free.app/task" ^
  -H "Content-Type: application/json" ^
  -d "{\"email\":\"24f2006582@ds.study.iitm.ac.in\", \"secret\":\"prakriti-tds\", \"url\":\"h

Expected Output

{
  "status": "done",
  "steps": 3,
  "time_s": 12
}


the solver will follow redirects, compute answers using either:
demo handlers, or universal LLM reasoning, and then submit the solution back automatically.
  

This project is licensed under the MIT License.

