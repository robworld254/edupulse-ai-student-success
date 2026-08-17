setup:
	python -m pip install -r requirements.txt

fetch:
	python -m scripts.fetch_data

train:
	python -m scripts.train

run:
	streamlit run app.py

test:
	python -m pytest -q

lint:
	python -m ruff check app.py src views scripts tests
