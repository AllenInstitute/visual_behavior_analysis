test-assets:
	cp /allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/702013508_363887_20180524142941_stim.pkl ./tests/assets/issue_467.pkl
	cp /allen/programs/braintv/production/visualbehavior/prod0/specimen_789113132/behavior_session_815348761/815160769.pkl ./tests/assets/issue_482.pkl

test-env :
	python -m pip install pip -U
	python -m pip install pipenv
	python -m pipenv install .

init: test-assets test-env

test :
	python -m pipenv run python -m pytest
