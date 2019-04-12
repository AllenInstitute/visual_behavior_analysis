test-assets:
	cp /allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/702013508_363887_20180524142941_stim.pkl ./tests/assets/issue_467.pkl

test-env :
	python -m pip install pip -U
	python -m pip install pipenv
	python -m pipenv install .

init: test-assets test-env

test :
	python -m pipenv run python -m pytest
