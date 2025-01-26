# mann-vs-machine
Machine Learning project about distinguishing between AI-generated and human made Tweets

How to run:
Suggested version of Python: 3.11.9

Install packages from requirements.txt
In CMD type uvicorn src.frontend_backend.app:app --reload
In the browser go to http://127.0.0.1:8000/

Other scripts should be ran from the root of the project as modules, for example:
python -m src.frontend_backend.utils_and_tests.testLogReg