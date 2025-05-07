all: run

run:
	python src/pipeline.py train

data:
	python src/data.py

batch:
	python src/batch.py

install:
	uv sync
