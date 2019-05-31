test-assets:
	cp /allen/programs/braintv/production/neuralcoding/prod0/specimen_652073919/ophys_session_702013508/702013508_363887_20180524142941_stim.pkl ./tests/assets/issue_467.pkl
	cp /allen/programs/braintv/production/visualbehavior/prod0/specimen_789113132/behavior_session_815348761/815160769.pkl ./tests/assets/issue_482.pkl
	cp /allen/programs/braintv/workgroups/nc-ophys/vba-assets/180730174647_385533_be539022-b792-4484-817d-cb9c1379307f.pkl ./tests/assets/issue_572_0.pkl
	cp /allen/programs/braintv/workgroups/nc-ophys/vba-assets/180725165745_385533_b329fdcb-2c26-4ef6-881a-b612dd0094a1.pkl ./tests/assets/issue_572_1.pkl
	cp /allen/programs/braintv/workgroups/nc-ophys/vba-assets/190404162006_408528_518e6a25-d5cc-405c-a40d-97419ce377ab.pkl ./tests/assets/issue_572_2.pkl

test-env :
	python -m pip install pip -U
	python -m pip install pipenv
	python -m pipenv install .

init: test-assets test-env

test :
	python -m pipenv run python -m pytest
