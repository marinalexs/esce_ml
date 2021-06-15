style:
	black esce tests
	isort esce tests

lint:
	flake8 esce tests
	mypy --strict esce tests

test:
	pytest tests

qa: style lint test

clean:
	rm -r data cache splits plots results .mypy_cache .pytest_cache
