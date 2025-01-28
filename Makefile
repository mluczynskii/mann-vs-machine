run:
	python3 -m pip install -r requirements.txt
	uvicorn src.frontend_backend.app:app --reload