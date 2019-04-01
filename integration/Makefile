assets:
	docker -d \
		--name make-assets \
		-f Assets.Dockerfile \
		-v vba-assets:/
	docker container stop make-assets
	docker container rm make-assets

clean:
	docker container stop make-assets
	docker container rm make-assets
	docker volume rm vba-assets

test-env:
	pip install pip -U
	pip install pipenv
	pipenv install .

test:
	cd ./tests && python -m pipenv run pytest
	cd ..
