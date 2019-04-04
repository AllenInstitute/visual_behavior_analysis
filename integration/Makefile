assets :
	docker -d \
		--name make-assets \
		-f Assets.Dockerfile \
		-v vba-assets:/
	docker container stop make-assets
	docker container rm make-assets

clean :
	docker container stop make-assets
	docker container rm make-assets
	docker volume rm vba-assets

test-env :
	python -m pip install pip -U
	python -m pip install pipenv
	python -m pipenv install .

test :
	python -m pipenv run python -m pytest

auto-test : test-env
	python -m pipenv run python -m integrator.py
