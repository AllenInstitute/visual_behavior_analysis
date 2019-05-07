test-assets:
	cp /allen/programs/braintv/production/visualbehavior/prod0/specimen_765992005/behavior_session_808500887/808387448.pkl ./tests/assets/issue_467.pkl
	cp /allen/programs/braintv/production/visualbehavior/prod0/specimen_789113132/behavior_session_815348761/815160769.pkl ./tests/assets/issue_482.pkl

test-env :
	python -m pip install pip -U
	python -m pip install pipenv
	python -m pipenv install .

init: test-assets test-env

test :
	python -m pipenv run python -m pytest

clean:
	rm ./tests//assets/*.pkl