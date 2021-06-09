style:
	black .
	isort .

lint:
	flake8 .
	mypy --strict .

test:
	pytest .

qa: style lint test

clean:
	rm -r data cache splits plots results
