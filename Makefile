VENV_NAME = "needle_in_haystack_venv"

setup: create_venv
	@echo "Activate the venv with: \`source ./$(VENV_NAME)/bin/activate\`"

create_venv:
	python3 -m venv ./$(VENV_NAME) ;\
	source ./$(VENV_NAME)/bin/activate ;\

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +

destroy: clean
	rm -rf ./$(VENV_NAME)